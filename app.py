"""FastAPI routes for the SEC Value Screen app. All the actual logic lives in
the sibling modules (screening.py, exchange_filter.py, market_data.py,
data_ingestion.py, formatting.py, pipeline_imports.py) -- this file just wires
HTTP requests to that logic and renders templates.
"""

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.paths import TEMPLATES_DIR, BASE_DIR
from core.types import IngestState, RowContext
from providers.pipeline_imports import sec_screen, require_pipeline
from core.fundamentals.data_ingestion import resolve_data_sources, get_db_connection, list_available_data_files, clear_cached_data
from core.fundamentals.exchange_filter import load_exchange_map, get_allowed_cik_set
from core.fundamentals.screening import (
    PAGE_SIZE, parse_thresholds, build_query_string, base_query_string,
    load_cached_ratios, load_history_for_cik, screen_data_from_cache,
    paginate_frame, build_row_context, coerce_row_values,
    get_ingest_summary, get_ingest_log,
    load_watchlist, toggle_watchlist, load_watchlist_data,
    _invalidate_cache, save_market_prices_to_db,
)
from providers.market_data import join_market_data
from providers.schwab.auth import (
    get_connection_status, exchange_code_for_tokens, SchwabAuthError,
    build_authorization_url, save_pending_credentials, load_pending_credentials,
)
from core.fundamentals.valuation import compute_margin_of_safety
from core.fundamentals.company_analysis import compute_red_flags, build_trend_table, scan_footnotes_for_red_flags
from core.fundamentals.notes_ingestion import ingest_notes_txt, get_footnotes_for_cik, has_footnote_data
import core.insider.insider_analysis as insider
import core.formatting as formatting
import config

app = FastAPI(title="SEC Value Screen")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["fmt_pct"] = formatting.fmt_pct
templates.env.filters["fmt_number"] = formatting.fmt_number
templates.env.filters["fmt_currency"] = formatting.fmt_currency

# ---------------------------------------------------------------------------
# Ingest progress tracker (in-memory, thread-safe)
# ---------------------------------------------------------------------------
_ingest_lock = threading.Lock()
_ingest_state: IngestState = {
    "running": False,
    "started_at": None,  # epoch seconds — used to detect hung ingests
    "total_sources": 0,
    "current_source": "",
    "completed_sources": 0,
    "total_filings": 0,
    "sources_done": [],   # list of {"name": str, "filings": int}
    "error": None,
    "complete": False,
}

# If a background ingest thread hasn't reported completion or error within this
# many seconds, treat it as hung and allow a new ingest to start.
INGEST_HUNG_TIMEOUT_SECONDS = 30 * 60  # 30 minutes


def _reset_ingest_state(total: int) -> None:
    with _ingest_lock:
        _ingest_state.update(
            running=True, started_at=time.time(), total_sources=total,
            current_source="", completed_sources=0, total_filings=0,
            sources_done=[], error=None, complete=False,
        )


def _update_ingest_progress(source_name: str = "", filings: int = 0,
                            completed: bool = False) -> None:
    with _ingest_lock:
        if source_name and filings:
            _ingest_state["current_source"] = source_name
            _ingest_state["completed_sources"] += 1
            _ingest_state["total_filings"] += filings
            _ingest_state["sources_done"].append(
                {"name": source_name, "filings": filings}
            )
        if completed:
            _ingest_state["running"] = False
            _ingest_state["started_at"] = None
            _ingest_state["complete"] = True


def _set_ingest_error(error: str) -> None:
    with _ingest_lock:
        _ingest_state["running"] = False
        _ingest_state["started_at"] = None
        _ingest_state["error"] = error
        _ingest_state["complete"] = True


def _get_ingest_state() -> dict:
    with _ingest_lock:
        return dict(_ingest_state)


def _run_ingest(data_dir: str, filter_major_exchanges_at_ingest: str) -> None:
    """Runs the full ingest pipeline in a background thread, updating
    _ingest_state as each source completes."""
    try:
        require_pipeline()
        data_sources = resolve_data_sources(data_dir)
    except Exception as exc:
        _set_ingest_error(str(exc))
        return

    _reset_ingest_state(len(data_sources))

    allowed_ciks = None
    if filter_major_exchanges_at_ingest in {"on", "true", "1"}:
        exchange_map, exch_error = load_exchange_map()
        if exchange_map:
            allowed_ciks = get_allowed_cik_set(exchange_map)

    con = get_db_connection()
    try:
        for source_path in data_sources:
            _update_ingest_progress(source_name=source_path.name)
            try:
                sec_screen.load_data_filtered(
                    con, str(source_path),
                    form_types={"10-K", "10-Q"},
                    allowed_ciks=allowed_ciks,
                )
                sec_screen.filter_relevant_submissions(con)
                matched = con.execute(
                    "SELECT COUNT(*) FROM sub_filtered"
                ).fetchone()[0]
                if matched == 0:
                    continue

                per_filing = sec_screen.build_filing_fundamentals(con)
                if per_filing.empty:
                    continue

                sec_screen.accumulate_fundamentals_history(con, per_filing)
                sec_screen.log_ingest(con, source_path.name, len(per_filing))

                # --- Footnote notes ingestion (optional) ---
                # If this data directory also has txt.txt (from the SEC
                # "Notes Data Sets"), ingest qualitative footnote text blocks.
                try:
                    notes_count = ingest_notes_txt(con, str(source_path))
                except Exception as exc:
                    notes_count = 0  # notes are optional — never fail an ingest
                    print(f"[notes] Optional notes ingest skipped: {exc}")

                _update_ingest_progress(
                    source_name=source_path.name, filings=len(per_filing),
                )
            except Exception as exc:
                _set_ingest_error(f"{source_path.name}: {exc}")
                return

        con.commit()

        # --- Insider trading ingestion (optional) ---
        # Try the user's data_dir first, then fall back to the sibling
        # data/insider_trading directory (they're separate SEC datasets).
        # A missing or broken insider ZIP never blocks the main ingest.
        insider_dirs = [data_dir]
        sibling_insider = Path(data_dir).parent / "insider_trading"
        if sibling_insider.is_dir():
            insider_dirs.append(str(sibling_insider))
        for insider_dir in insider_dirs:
            try:
                insider_df = insider.load_and_process_data(
                    insider_dir, start_year=2016, end_year=2026,
                )
                if not insider_df.empty:
                    classified = insider.classify_insiders(insider_df)
                    new_rows = insider.persist_insider_trades(con, classified)
                    if new_rows > 0:
                        print(
                            f"Insider ingest ({insider_dir}): {new_rows:,} "
                            f"new trades classified and stored."
                        )
                        break  # found data, stop looking
            except Exception as exc:
                print(f"[insider] {insider_dir}: skipped ({exc})")

        history = sec_screen.load_fundamentals_history(con)
        if history.empty:
            _set_ingest_error(
                "No usable filings were found across any of the ingested sources."
            )
            return

        n_companies = history["cik"].nunique()
        n_10k = int((history["form"].str.upper() == "10-K").sum())
        n_10q = int((history["form"].str.upper() == "10-Q").sum())
        print(
            f"Ingest complete: {len(history)} total filing(s) across "
            f"{n_companies} companies ({n_10k} 10-K, {n_10q} 10-Q) "
            f"accumulated in history."
        )
    except Exception as exc:
        _set_ingest_error(str(exc))
        return
    finally:
        con.close()

    _update_ingest_progress(completed=True)


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request) -> HTMLResponse:
    summary = get_ingest_summary()
    ingest_log = get_ingest_log()
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "schwab_status": get_connection_status(),
            "summary": summary,
            "ingest_log": ingest_log,
        },
    )


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request) -> HTMLResponse:
    data_dir = request.query_params.get("data_dir", "./data")
    available_files = list_available_data_files(data_dir)
    summary = get_ingest_summary()
    ingest_log = get_ingest_log()
    return templates.TemplateResponse(
        request,
        "setup.html",
        {"error": None, "data_dir": data_dir, "form_type": "", "available_files": available_files,
         "schwab_status": get_connection_status(), "summary": summary, "ingest_log": ingest_log,
         "schwab_client_id": os.environ.get("SCHWAB_CLIENT_ID", ""),
         "schwab_client_secret": os.environ.get("SCHWAB_CLIENT_SECRET", ""),
         "schwab_redirect_uri": os.environ.get("SCHWAB_REDIRECT_URI",
                                               "http://127.0.0.1:8000/schwab/callback")},
    )


@app.post("/ingest", response_class=HTMLResponse)
async def ingest_data(
    request: Request,
    data_dir: str = Form(...),
    form_type: str = Form(default=""),
    filter_major_exchanges_at_ingest: str = Form(default=""),
) -> HTMLResponse:
    """Launch ingest in a background thread and return a progress bar that
    polls /ingest/progress until completion."""
    # Quick validation before launching the thread
    if not data_dir.strip():
        return templates.TemplateResponse(
            request, "_ingest_progress.html",
            {"request": request, "error": "No data directory provided.",
             "state": _get_ingest_state()},
        )

    state = _get_ingest_state()
    if state["running"]:
        started = state.get("started_at")
        if started is not None and (time.time() - started) > INGEST_HUNG_TIMEOUT_SECONDS:
            # Ingest thread appears hung — auto-reset so another can start.
            _set_ingest_error(
                "Previous ingest timed out after "
                f"{INGEST_HUNG_TIMEOUT_SECONDS // 60} minutes without completing — "
                "the background thread may have hung. You can try again."
            )
            state = _get_ingest_state()
        else:
            return templates.TemplateResponse(
                request, "_ingest_progress.html",
                {"request": request, "error": None, "state": state},
            )

    thread = threading.Thread(
        target=_run_ingest,
        args=(data_dir, filter_major_exchanges_at_ingest),
        daemon=True,
    )
    thread.start()
    # Give the thread a moment to parse sources and populate total count
    time.sleep(0.1)

    return templates.TemplateResponse(
        request, "_ingest_progress.html",
        {"request": request, "error": None, "state": _get_ingest_state()},
    )


@app.get("/ingest/progress", response_class=HTMLResponse)
async def ingest_progress(request: Request) -> HTMLResponse:
    """HTMX polling endpoint -- returns the progress bar HTML snippet."""
    return templates.TemplateResponse(
        request, "_ingest_progress.html",
        {"request": request, "error": None, "state": _get_ingest_state()},
    )


@app.post("/setup/reset", response_class=HTMLResponse)
async def reset_data(request: Request) -> HTMLResponse:
    """Clear ALL cached data (fundamentals, market prices, watchlist, logs)
    and invalidate the in-memory cache.  Use this when you need to re-ingest
    from scratch — e.g. after adding new XBRL tags to TAG_MAP that require
    re-parsing the raw SEC files.
    """
    try:
        cleared = clear_cached_data()
        _invalidate_cache()
        msg = "Cleared: " + ", ".join(cleared) if cleared else "Nothing to clear — cache was already empty."
    except Exception as exc:
        msg = f"Error clearing cache: {exc}"

    return templates.TemplateResponse(
        request, "_reset_result.html",
        {"request": request, "message": msg},
    )


@app.get("/screen", response_class=HTMLResponse)
async def screen_page(request: Request) -> HTMLResponse:
    params = dict(request.query_params)
    thresholds = parse_thresholds(params)
    try:
        ranked_df, info = screen_data_from_cache(params, include_market_data=False)
        page_df, pagination = paginate_frame(ranked_df, params)
        rows = [build_row_context(row, params) for _, row in page_df.iterrows()]
        context = {
            "request": request,
            "rows": rows,
            "thresholds": thresholds,
            "pagination": pagination,
            "query_string": request.url.query,
            "base_query_string": base_query_string(params),
            "sort": params.get("sort", "roic"),
            "order": params.get("order", "asc"),
            "error": None,
            "diagnostics": info.get("diagnostics", {}) if not rows else None,
            "summary": info.get("summary"),
            "market_errors": info.get("errors", []),
            "schwab_status": get_connection_status(),
            "market_data_provider": config.MARKET_DATA_PROVIDER,
            "watchlist": load_watchlist(),
        }
        template_name = "_results_table.html" if request.headers.get("hx-request") == "true" else "screen.html"
        return templates.TemplateResponse(request, template_name, context)
    except Exception as exc:
        context = {
            "request": request,
            "rows": [],
            "thresholds": thresholds,
            "pagination": {"page": 1, "total_pages": 1, "page_size": PAGE_SIZE, "total_rows": 0},
            "query_string": request.url.query,
            "base_query_string": base_query_string(params),
            "sort": params.get("sort", "roic"),
            "order": params.get("order", "asc"),
            "error": str(exc),
            "diagnostics": None,
            "summary": None,
            "market_errors": [],
            "schwab_status": get_connection_status(),
            "market_data_provider": config.MARKET_DATA_PROVIDER,
            "watchlist": load_watchlist(),
        }
        template_name = "_results_table.html" if request.headers.get("hx-request") == "true" else "screen.html"
        return templates.TemplateResponse(request, template_name, context)


@app.post("/market-data/refresh", response_class=HTMLResponse)
async def refresh_market_data(request: Request) -> HTMLResponse:
    params = dict(request.query_params)
    thresholds = parse_thresholds(params)
    scope = params.pop("scope", "page")  # "page" (fast, default) or "all" (slow, explicit opt-in)
    try:
        # Get the filtered/ranked set WITHOUT market data first (fast), then
        # paginate, THEN join market data only onto the rows we're actually
        # about to display -- unless the user explicitly asked for the full
        # filtered universe via scope=all, which is intentionally slow (one
        # live network call per company) and should be a deliberate choice,
        # not the default button behavior.
        ranked_df, info = screen_data_from_cache(params, include_market_data=False)
        page_df, pagination = paginate_frame(ranked_df, params)

        enrich_target = ranked_df if scope == "all" else page_df
        enriched, errors = (
            join_market_data(enrich_target, overall_timeout_seconds=(90.0 if scope == "all" else 20.0))
            if not enrich_target.empty else (enrich_target, [])
        )

        # Persist fetched prices so they survive page reloads, and compute
        # margin-of-safety against those prices.
        save_market_prices_to_db(enriched)
        if "Price" in enriched.columns and enriched["Price"].notna().any():
            enriched = compute_margin_of_safety(enriched)

        if scope == "all":
            # Re-paginate the now-enriched, full-universe-ranked frame so the
            # displayed page reflects the completed Magic Formula ranking.
            enriched = sec_screen.rank_magic_formula(enriched) if "ROIC" in enriched.columns else enriched
            page_df, pagination = paginate_frame(enriched, params)
        else:
            page_df = enriched

        rows = [build_row_context(row, params) for _, row in page_df.iterrows()]
        context = {
            "request": request,
            "rows": rows,
            "thresholds": thresholds,
            "pagination": pagination,
            "query_string": request.url.query,
            "base_query_string": base_query_string(params),
            "sort": params.get("sort", "roic"),
            "order": params.get("order", "asc"),
            "error": None,
            "market_errors": errors,
            "market_scope": scope,
            "summary": info.get("summary"),
            "diagnostics": None,
            "schwab_status": get_connection_status(),
            "market_data_provider": config.MARKET_DATA_PROVIDER,
            "watchlist": load_watchlist(),
        }
        return templates.TemplateResponse(request, "_results_table.html", context)
    except Exception as exc:
        context = {
            "request": request,
            "rows": [],
            "thresholds": thresholds,
            "pagination": {"page": 1, "total_pages": 1, "page_size": PAGE_SIZE, "total_rows": 0},
            "query_string": request.url.query,
            "base_query_string": base_query_string(params),
            "sort": params.get("sort", "roic"),
            "order": params.get("order", "asc"),
            "error": str(exc),
            "market_errors": [],
            "market_scope": scope,
            "summary": None,
            "diagnostics": None,
            "schwab_status": get_connection_status(),
            "market_data_provider": config.MARKET_DATA_PROVIDER,
            "watchlist": load_watchlist(),
        }
        return templates.TemplateResponse(request, "_results_table.html", context)


@app.get("/export")
async def export_results(request: Request) -> StreamingResponse:
    params = dict(request.query_params)
    try:
        ranked_df, _ = screen_data_from_cache(params, include_market_data=False)
        csv_body = ranked_df.to_csv(index=False)
        return StreamingResponse(iter([csv_body]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=screen_results.csv"})
    except Exception as exc:
        return StreamingResponse(iter([f"error,{exc}\n"]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=screen_results.csv"})


@app.get("/company/{cik}", response_class=HTMLResponse)
async def company_detail(request: Request, cik: str) -> HTMLResponse:
    try:
        df = load_cached_ratios()
        cik_col = "cik" if "cik" in df.columns else "CIK"
        mask = df[cik_col].astype(str) == str(cik)
        matches = df[mask]
        if matches.empty:
            raise ValueError("No company with that CIK was found in the cached data.")
        match = matches.iloc[0]

        # Hydrate persisted market prices for this single company so the
        # detail page shows valuation/market data without a live refresh.
        from core.fundamentals.screening import _hydrate_prices_from_db
        single_df = _hydrate_prices_from_db(pd.DataFrame([match]))
        match = single_df.iloc[0]

        values = coerce_row_values(match)
        grouped: Dict[str, List[Tuple[str, str]]] = {g: [] for g in formatting.GROUP_ORDER}
        for column, value in values.items():
            if column in {"CIK", "cik", "Name", "name", "SIC", "sic"}:
                continue
            group = formatting.GROUP_MAP.get(column, "Other Fundamentals")
            display_value = formatting.format_display_value(column, value)
            grouped[group].append((column, display_value))
        grouped = {g: items for g, items in grouped.items() if items}

        # Load full filing history — used for trend table (10-K only, year-over-
        # year comparison) and red-flag checks (also 10-K only, to avoid false
        # positives from mixing annual and quarterly figures).
        history_df = load_history_for_cik(cik)
        annuals_df = history_df[history_df["form"].str.strip().str.upper() == "10-K"] if not history_df.empty and "form" in history_df.columns else history_df

        # Red flags — computed from 10-K (annual) data only so quarter-to-quarter
        # noise and FY-vs-Q1 magnitude differences don't trigger spurious warnings.
        red_flags = compute_red_flags(annuals_df)

        # TTM basis — explains exactly how TTM was constructed for this company
        ttm_basis = values.get("ttm_basis")

        # Sector label
        from core.formatting import sic_to_sector
        sic_val = values.get("SIC") or values.get("sic")
        sector_label = sic_to_sector(sic_val) if sic_val is not None else None

        # Multi-year trend table — 10-K filings only for clean year-over-year
        # comparison.  Falls back to mixed history if fewer than two 10-Ks exist.
        trend_metrics = ["Revenues", "OperatingIncomeLoss", "NetIncomeLoss",
                         "ROIC", "OperatingMargin", "DebtToEquity", "FCF"]
        source_for_trend = annuals_df if len(annuals_df) >= 2 else history_df
        trend_rows = build_trend_table(source_for_trend, trend_metrics)

        context = {
            "request": request,
            "company": values,
            "groups": grouped,
            "red_flags": red_flags,
            "ttm_basis": ttm_basis,
            "sector_label": sector_label,
            "trend_metrics": trend_metrics,
            "trend_rows": trend_rows,
            "back_link": f"/screen?{request.url.query}" if request.url.query else "/screen",
            "schwab_status": get_connection_status(),
            "on_watchlist": str(cik) in load_watchlist(),
        }
        return templates.TemplateResponse(request, "company.html", context)
    except Exception as exc:
        return templates.TemplateResponse(request, "company.html", {
            "request": request, "company": {}, "groups": {},
            "red_flags": [], "ttm_basis": None,
            "trend_metrics": [], "trend_rows": [],
            "back_link": "/screen", "error": str(exc),
            "schwab_status": get_connection_status(),
        })


@app.get("/company/{cik}/footnotes", response_class=HTMLResponse)
async def company_footnotes(request: Request, cik: str) -> HTMLResponse:
    """HTMX lazy-load endpoint — returns footnote text blocks and qualitative
    red-flag scan results for a single company.  Called by company.html after
    the main page renders so footnote data (which can be large) doesn't block
    the initial page load."""
    try:
        if not has_footnote_data():
            return templates.TemplateResponse(
                request, "_company_footnotes.html",
                {"request": request, "footnotes": [], "qual_flags": [],
                 "error": "No footnote data has been ingested yet. "
                          "Download SEC Notes Data Sets and re-ingest."},
            )

        df = get_footnotes_for_cik(cik)
        if df.empty:
            return templates.TemplateResponse(
                request, "_company_footnotes.html",
                {"request": request, "footnotes": [], "qual_flags": [], "error": None},
            )

        # Qualitative red-flag scan
        qual_flags = scan_footnotes_for_red_flags(df)

        # Build display list for the template
        from core.formatting import FOOTNOTE_TAG_LABELS
        footnotes = []
        for _, row in df.iterrows():
            tag = row.get("tag") or ""
            text = row.get("txt_value") or ""
            ddate = row.get("ddate") or ""
            fy = row.get("fy") or ""
            fp = row.get("fp") or ""

            # Truncate very long text blocks for display
            display_text = text[:5000]
            if len(text) > 5000:
                display_text += f"\n\n… [truncated — {len(text):,} chars total]"

            footnotes.append({
                "label": FOOTNOTE_TAG_LABELS.get(tag, tag),
                "text": display_text,
                "ddate": formatting.format_display_value("period", str(ddate)),
                "form_info": f"FY{fy} {fp}" if fy else "",
            })

        return templates.TemplateResponse(
            request, "_company_footnotes.html",
            {"request": request, "footnotes": footnotes,
             "qual_flags": qual_flags, "error": None},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request, "_company_footnotes.html",
            {"request": request, "footnotes": [], "qual_flags": [],
             "error": str(exc)},
        )


@app.get("/company/{cik}/insider-trades", response_class=HTMLResponse)
async def company_insider_trades(request: Request, cik: str) -> HTMLResponse:
    """HTMX lazy-load endpoint — returns recent insider trades and summary
    for a single company.  Requires insider trading data to have been
    ingested first."""
    con = get_db_connection()
    try:
        if not insider.has_insider_data(con):
            return templates.TemplateResponse(
                request, "_company_insider_trades.html",
                {"request": request, "cik": cik, "trades": [],
                 "summary": {"total_trades": 0},
                 "error": "No insider trading data has been ingested yet."},
            )

        df = insider.get_insider_trades_for_cik(cik, con, limit=30)
        summary = insider.get_insider_summary_for_cik(cik, con)

        # Format for display
        trades = []
        for _, row in df.iterrows():
            # Resolve company name from collected data or fall back to ticker
            company_name = (
                row.get("COMPANY_NAME")
                or row.get("issuer_trading_symbol")
                or f"CIK {cik}"
            )
            trade_date = row.get("trans_date")
            if hasattr(trade_date, "strftime"):
                trade_date = trade_date.strftime("%b %d, %Y")

            trades.append({
                "date": trade_date or "",
                "insider_name": row.get("rpt_owner_name") or "—",
                "insider_role": row.get("rpt_owner_relationship") or "",
                "code": "Buy" if str(row.get("transaction_code") or "").strip().upper() == "P" else "Sell",
                "shares": formatting.fmt_number(row.get("trans_shares")),
                "price": formatting.fmt_currency(row.get("trans_price_per_share")),
                "trade_type": row.get("trade_type", "").title(),
                "routine_years": int(row.get("routine_years") or 0),
                "company_name": company_name,
            })

        return templates.TemplateResponse(
            request, "_company_insider_trades.html",
            {"request": request, "cik": cik, "trades": trades,
             "summary": summary, "error": None},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request, "_company_insider_trades.html",
            {"request": request, "cik": cik, "trades": [],
             "summary": {"total_trades": 0}, "error": str(exc)},
        )
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Scuttlebutt & Idea Pipeline
# ---------------------------------------------------------------------------

@app.get("/scuttlebutt", response_class=HTMLResponse)
async def scuttlebutt_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "strategies/scuttlebutt.html",
        {"request": request, "schwab_status": get_connection_status()},
    )


@app.get("/ideas", response_class=HTMLResponse)
async def idea_pipeline_page(request: Request) -> HTMLResponse:
    """Triangulation page — rank companies by combined Value + Insider +
    Scuttlebutt signals using the live fundamentals and insider data."""
    from triangulation.ranker import rank_ideas

    ideas = []
    con = get_db_connection()
    try:
        # Value data: per-company passes from the screening engine
        from core.fundamentals.screening import load_cached_ratios
        try:
            df = load_cached_ratios()
            value_data = {}
            if not df.empty:
                cik_col = "cik" if "cik" in df.columns else "CIK"
                name_col = "name" if "name" in df.columns else "Name"
                for _, row in df.iterrows():
                    cik_str = str(row.get(cik_col) or "").strip()
                    if not cik_str:
                        continue
                    value_data[cik_str] = {
                        "name": str(row.get(name_col, "")),
                        "median_roic": float(row.get("ROIC") or 0),
                        "total_passed": 1,
                    }
        except Exception:
            value_data = {}

        # Insider data: per-company summary
        insider_data = {}
        if hasattr(insider, 'has_insider_data') and insider.has_insider_data(con):
            try:
                all_ciks = con.execute(
                    "SELECT DISTINCT issuer_cik FROM insider_trades"
                ).fetchall()
                for (issuer_cik,) in all_ciks:
                    cik_str = str(issuer_cik).strip()
                    if not cik_str:
                        continue
                    summ = insider.get_insider_summary_for_cik(cik_str, con)
                    if summ.get("total_trades", 0) > 0:
                        insider_data[cik_str] = summ
            except Exception:
                pass

        # Rank ideas
        ideas = rank_ideas(value_data, insider_data, {}, top_n=50)
    finally:
        con.close()

    return templates.TemplateResponse(
        request, "strategies/idea_pipeline.html",
        {"request": request, "ideas": ideas,
         "schwab_status": get_connection_status()},
    )


# ---------------------------------------------------------------------------
# Insider Trading — standalone page
# ---------------------------------------------------------------------------

@app.get("/insider", response_class=HTMLResponse)
async def insider_page(request: Request) -> HTMLResponse:
    """Standalone insider trading analysis page with aggregate stats,
    top companies, and searchable trade history."""
    con = get_db_connection()
    try:
        if not insider.has_insider_data(con):
            return templates.TemplateResponse(
                request, "insider.html",
                {"request": request, "summary": None, "top_companies": [],
                 "trades": [], "schwab_status": get_connection_status(),
                 "search": "", "trade_type": "", "code": "", "page": 1,
                 "error": "No insider trading data has been ingested yet. "
                          "Place SEC EDGAR insider ZIP files (containing "
                          "NONDERIV_TRANS.txt, SUBMISSION.txt, "
                          "REPORTINGOWNER.txt) in your data directory "
                          "and re-run the main ingest from Setup."},
            )

        major_only = request.query_params.get("major_exchanges_only", "1")
        major_only = major_only in {"1", "on", "true", ""}

        summary = insider.get_aggregate_summary(con)
        top = insider.get_top_companies(
            con, by="opp_trades", limit=10, major_exchanges_only=major_only,
        )
        trades_df = insider.search_insider_trades(
            con, limit=50, major_exchanges_only=major_only,
        )
        trades = _format_insider_trade_rows(trades_df)
        trades = insider.enrich_trades_with_prices(trades, con)
        perf = insider.compute_performance_summary(trades)

        return templates.TemplateResponse(
            request, "insider.html",
            {"request": request, "summary": summary,
             "top_companies": top.to_dict(orient="records"),
             "trades": trades, "performance": perf,
             "schwab_status": get_connection_status(),
             "error": None, "search": "", "trade_type": "", "code": "",
             "page": 1, "major_exchanges_only": major_only},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request, "insider.html",
            {"request": request, "summary": None, "top_companies": [],
             "trades": [], "schwab_status": get_connection_status(),
             "error": str(exc)},
        )
    finally:
        con.close()


@app.get("/insider/search", response_class=HTMLResponse)
async def insider_search(request: Request) -> HTMLResponse:
    """HTMX endpoint — returns filtered trade rows for the insider page."""
    search = request.query_params.get("search", "")
    trade_type = request.query_params.get("trade_type", "")
    code = request.query_params.get("code", "")
    major_only = request.query_params.get("major_exchanges_only", "1")
    major_only = major_only in {"1", "on", "true", ""}
    page = max(1, int(request.query_params.get("page", 1)))
    limit = 50
    offset = (page - 1) * limit

    con = get_db_connection()
    try:
        if not insider.has_insider_data(con):
            return HTMLResponse(
                '<tr><td colspan="7" class="text-sm text-surface-500 italic py-4">'
                'No insider trading data available.</td></tr>'
            )
        df = insider.search_insider_trades(
            con, search=search, trade_type=trade_type, code=code,
            limit=limit, offset=offset,
            major_exchanges_only=major_only,
        )
        trades = _format_insider_trade_rows(df)
        # Enrich with cached prices only (don't hit Schwab on every search)
        trades = insider.enrich_trades_with_prices(trades, con)
    finally:
        con.close()

    return templates.TemplateResponse(
        request, "_insider_trade_rows.html",
        {"request": request, "trades": trades, "page": page,
         "search": search, "trade_type": trade_type, "code": code,
         "major_exchanges_only": major_only},
    )


@app.post("/insider/ingest", response_class=HTMLResponse)
async def insider_ingest(
    request: Request,
    data_dir: str = Form(...),
) -> HTMLResponse:
    """Standalone insider data ingestion — processes SEC EDGAR ZIP files
    (named like 2016q1_form345.zip) and persists classified trades."""
    try:
        df = insider.load_and_process_data(
            str(data_dir), start_year=2016, end_year=2026,
        )
        if df.empty:
            return HTMLResponse(
                '<div class="rounded-lg border border-amber-200 bg-amber-50 '
                'px-4 py-3 text-[13px] text-amber-700">'
                'No insider transactions found. Ensure the directory contains '
                'ZIP files with NONDERIV_TRANS.txt, SUBMISSION.txt, and '
                'REPORTINGOWNER.txt inside.</div>'
            )
        classified = insider.classify_insiders(df)
        con = get_db_connection()
        try:
            new_rows = insider.persist_insider_trades(con, classified)
        finally:
            con.close()
        stats = insider.get_summary_stats(classified)
        return HTMLResponse(
            '<div class="rounded-lg border border-positive-200 bg-positive-50 '
            'px-4 py-3 text-[13px] text-positive-700">'
            f'✓ Ingested {new_rows:,} trades. '
            f'{stats["pct_opportunistic"]}% opportunistic, '
            f'{stats["pct_routine"]}% routine across '
            f'{stats["unique_insiders"]:,} insiders at '
            f'{stats["unique_firms"]:,} companies. '
            f'<a href="/insider" class="underline font-semibold">'
            f'Reload page</a> to see results.'
            '</div>'
        )
    except Exception as exc:
        return HTMLResponse(
            '<div class="rounded-lg border border-negative-200 bg-negative-50 '
            f'px-4 py-3 text-[13px] text-negative-700">{exc}</div>'
        )


def _format_insider_trade_rows(df: pd.DataFrame) -> list:
    """Convert a DataFrame of insider trades into display-ready dicts."""
    trades = []
    for _, row in df.iterrows():
        trade_date = row.get("trans_date")
        if hasattr(trade_date, "strftime"):
            trade_date = trade_date.strftime("%b %d, %Y")
        trades.append({
            "date": trade_date or "",
            "company_name": row.get("company_name") or "—",
            "ticker": row.get("issuer_trading_symbol") or "",
            "insider_name": row.get("rpt_owner_name") or "—",
            "insider_role": row.get("rpt_owner_relationship") or "",
            "code": "Buy" if str(row.get("transaction_code") or "").strip().upper() == "P" else "Sell",
            "shares": formatting.fmt_number(row.get("trans_shares")),
            "price": formatting.fmt_currency(row.get("trans_price_per_share")),
            "trade_type": str(row.get("trade_type") or "").title(),
            "routine_years": int(row.get("routine_years") or 0),
            "issuer_cik": str(row.get("issuer_cik") or ""),
        })
    return trades


# ---------------------------------------------------------------------------
# Watchlist routes
# ---------------------------------------------------------------------------

@app.get("/watchlist", response_class=HTMLResponse)
async def watchlist_page(request: Request) -> HTMLResponse:
    df = load_watchlist_data()
    rows = []
    if not df.empty:
        # Add sector labels
        from core.formatting import sic_to_sector
        sic_col = "sic" if "sic" in df.columns else "SIC"
        if sic_col in df.columns:
            df["Sector"] = df[sic_col].apply(sic_to_sector)
        # Sort by Magic Formula rank if available
        if "MagicFormulaRank" in df.columns:
            df = df.sort_values("MagicFormulaRank")
        for _, row_data in df.iterrows():
            ctx = build_row_context(row_data, {})
            # Watchlist-specific fields: entry date, entry price, current
            # price, and gain/loss delta.
            ctx["added_at"] = row_data.get("AddedAt")
            ctx["added_price"] = row_data.get("AddedPrice")
            ctx["current_price"] = ctx.get("price")  # from build_row_context
            if ctx["added_price"] is not None and ctx["current_price"] is not None:
                try:
                    ctx["price_change"] = float(ctx["current_price"]) - float(ctx["added_price"])
                    ctx["price_change_pct"] = (float(ctx["current_price"]) / float(ctx["added_price"])) - 1
                except (TypeError, ValueError, ZeroDivisionError):
                    ctx["price_change"] = None
                    ctx["price_change_pct"] = None
            else:
                ctx["price_change"] = None
                ctx["price_change_pct"] = None
            rows.append(ctx)
    return templates.TemplateResponse(request, "watchlist.html", {
        "request": request, "rows": rows,
        "schwab_status": get_connection_status(),
    })


@app.post("/watchlist/toggle/{cik}")
async def watchlist_toggle(cik: str):
    """Toggle a CIK on/off the watchlist.  Returns an HTML star button
    (not JSON) so HTMX's hx-swap=\"outerHTML\" replaces the clicked button
    in-place with the new state."""
    result = toggle_watchlist(cik)
    on_watchlist = result.get("on_watchlist", False)
    star_class = "btn-star--on" if on_watchlist else "btn-star--off"
    title = "Remove from watchlist" if on_watchlist else "Add to watchlist"
    pressed = "true" if on_watchlist else "false"
    return HTMLResponse(
        f'<button class="btn-star {star_class}" '
        f'hx-post="/watchlist/toggle/{cik}" hx-swap="outerHTML" '
        f'title="{title}" aria-pressed="{pressed}" '
        f'aria-label="{title}">★</button>'
    )


# ---------------------------------------------------------------------------
# Schwab OAuth — web-based connect / reconnect flow
# ---------------------------------------------------------------------------

@app.post("/schwab/authorize")
async def schwab_authorize(
    request: Request,
    client_id: str = Form(default=""),
    client_secret: str = Form(default=""),
    redirect_uri: str = Form(default=""),
):
    """Store the user's Schwab app credentials and redirect them to Schwab's
    OAuth authorization page.  Falls back to SCHWAB_CLIENT_ID /
    SCHWAB_CLIENT_SECRET / SCHWAB_REDIRECT_URI from the .env file when
    form fields are left empty."""
    cid = client_id.strip() or os.environ.get("SCHWAB_CLIENT_ID", "")
    csec = client_secret.strip() or os.environ.get("SCHWAB_CLIENT_SECRET", "")
    ruri = redirect_uri.strip() or os.environ.get(
        "SCHWAB_REDIRECT_URI", "http://127.0.0.1:8000/schwab/callback",
    )
    if not cid or not csec:
        return templates.TemplateResponse(
            request, "_schwab_result.html",
            {"request": request, "success": False,
             "message": "Client ID and Client Secret are required. "
                        "Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET in .env "
                        "or enter them in the form."},
        )
    state = save_pending_credentials(cid, csec, ruri)
    auth_url = build_authorization_url(cid, ruri)
    # Append state so we can look up credentials on callback
    separator = "&" if "?" in auth_url else "?"
    return RedirectResponse(f"{auth_url}{separator}state={state}", status_code=302)


@app.get("/schwab/callback")
async def schwab_callback(request: Request):
    """Handle the redirect from Schwab's OAuth authorization page.
    Exchanges the authorization code for tokens and saves them."""
    code = request.query_params.get("code")
    state = request.query_params.get("state", "")
    error = request.query_params.get("error")

    if error:
        return templates.TemplateResponse(
            request, "_schwab_result.html",
            {"request": request, "success": False,
             "message": f"Schwab returned an error: {error}"},
        )

    if not code:
        return templates.TemplateResponse(
            request, "_schwab_result.html",
            {"request": request, "success": False,
             "message": "No authorization code received from Schwab."},
        )

    pending = load_pending_credentials(state)
    if not pending:
        return templates.TemplateResponse(
            request, "_schwab_result.html",
            {"request": request, "success": False,
             "message": "Session expired or invalid. Please try connecting again from the Setup page."},
        )

    try:
        exchange_code_for_tokens(
            pending["client_id"], pending["client_secret"],
            pending["redirect_uri"], code,
        )
    except SchwabAuthError as exc:
        return templates.TemplateResponse(
            request, "_schwab_result.html",
            {"request": request, "success": False,
             "message": f"Token exchange failed: {exc}"},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request, "_schwab_result.html",
            {"request": request, "success": False,
             "message": f"Unexpected error: {exc}"},
        )

    return templates.TemplateResponse(
        request, "_schwab_result.html",
        {"request": request, "success": True,
         "message": "Schwab connected! Your access token will refresh automatically."},
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}