# SEC Value Screen

A local-first web application for value-investing screening, powered by SEC DERA XBRL fundamental data. Think "Bloomberg terminal lite for value investors, running entirely on your laptop."

## What it does

1. **Ingests SEC quarterly data dumps** — downloads or reads ZIP archives containing `sub.txt`, `num.txt`, `tag.txt`, `pre.txt` published quarterly by the SEC.
2. **Computes TTM fundamentals** — trailing-twelve-month figures using the standard formula (most recent 10-K + latest 10-Q YTD − prior-year same-quarter 10-Q YTD), correctly handling cumulative year-to-date flow figures.
3. **Calculates 30+ financial ratios** — ROIC (Greenblatt-style), margins, leverage, liquidity, coverage, earnings quality (accrual ratio, CFO/NI), and growth metrics.
4. **Applies value-investing screens** — Buffett/Pabrai-style quality filters with adjustable thresholds (ROIC, operating margin, debt/equity, interest coverage, CFO/NI, revenue growth, Piotroski F-Score).
5. **Fetches live market data** — from yfinance (zero setup) or Schwab API (richer data, batch quotes). Computes Enterprise Value, Earnings Yield, P/E, P/B, EV/EBIT, P/FCF.
6. **Ranks by Magic Formula** — Greenblatt's combined ROIC + Earnings Yield rank.
7. **Computes intrinsic valuation** — Graham Number and two-stage Discounted Cash Flow with user-adjustable assumptions.
8. **Shows company deep-dives** — multi-year trend tables, red flags (revenue declines, DSO ballooning, negative FCF, rising leverage), sector context, and margin of safety.
9. **Watchlist tracking** — save companies with entry price, see gain/loss deltas against current market prices.

## Investment methodology

The app implements several well-known value-investing frameworks:

| Framework | What it measures | Implementation |
|---|---|---|
| **Greenblatt Magic Formula** | ROIC (quality) + Earnings Yield (cheapness) | Combined rank across all screened companies |
| **Buffett/Pabrai Quality** | Durable competitive advantage | ROIC > 15%, margins > 10%, low leverage, cash-backed earnings |
| **Piotroski F-Score** | Financial health (0–9) | All 9 components from adjacent 10-K pairs |
| **Graham Number** | Defensive investor's max price | √(22.5 × EPS × Book Value per Share) |
| **DCF Intrinsic Value** | Present value of future cash flows | Two-stage model: projection period + perpetuity terminal value |
| **Earnings Quality** | Accrual accounting vs cash reality | Accrual ratio, CFO/NI coverage, DSO trend analysis |

### TTM construction

XBRL flow figures (income statement, cash flow) in 10-Q filings are reported as **cumulative year-to-date**, not clean single-quarter numbers. The app handles this correctly:

```
TTM = most recent 10-K (FY)
    + latest 10-Q YTD (e.g. Q3)
    − prior-year same-quarter 10-Q YTD (e.g. prior Q3)
```

Balance sheet items always come from the most recent filing regardless of form type — they're point-in-time snapshots, not flows.

### Red flags

The company detail page automatically checks for:

- Revenue declining in a majority of recent periods
- Net income or free cash flow negative in the most recent filing
- Debt/Equity ratio increasing >50% across available filings
- **DSO (Days Sales Outstanding) ballooning >20% while revenue grows** — a channel-stuffing or collection-deterioration signal
- Operating leases capitalized into total debt for lease-adjusted leverage (ASC 842)

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI + Jinja2 + HTMX                │
│                       (app.py)                            │
├──────────────────────────────────────────────────────────┤
│  core/                     │  providers/                  │
│  ├── sec_value_screen.py   │  ├── market_data.py          │
│  │   XBRL ingestion, TTM,  │  │   provider-agnostic join   │
│  │   ratios, Magic Formula  │  ├── yfinance_provider.py   │
│  ├── screening.py          │  └── schwab/                 │
│  │   thresholds, pagination│      OAuth2, batch quotes    │
│  │   search, watchlist      │                             │
│  ├── valuation.py          │                              │
│  │   Graham Number, DCF     │                              │
│  ├── company_analysis.py   │                              │
│  │   red flags, trend table │                              │
│  ├── formatting.py         │                              │
│  │   SIC→sector, Jinja      │                              │
│  ├── exchange_filter.py    │                              │
│  └── types.py              │                              │
├──────────────────────────────────────────────────────────┤
│                    DuckDB (cache/screen.duckdb)           │
│  fundamentals_history · market_prices · watchlist · logs  │
└──────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **DuckDB over pandas for ingestion** — SEC quarterly data files are hundreds of MB. DuckDB queries them directly via `read_csv()` without loading everything into memory, then runs full SQL for joins/pivots.
- **In-memory TTM cache with row-count invalidation** — the enriched TTM DataFrame (groupby over 45K+ rows) is cached at module level and only recomputed when new data is ingested.
- **Market prices persisted in DuckDB** — prices survive page reloads so margin-of-safety and multiples render without a live refresh every time.
- **Two market data providers** — yfinance (zero setup, one call per ticker, rate-limited) or Schwab (requires OAuth setup, one batch call for all tickers).
- **Graceful degradation** — exchange filter, ticker map, and market data all fail soft with warnings rather than hard errors. If the SEC endpoint is unreachable, you still get a screen — just unfiltered.
- **Server-side search** — the company search filters across ALL companies that passed the quality screen, not just the 20 on the current page.

## Project structure

```
.
├── app.py                    # FastAPI routes (thin wiring layer)
├── config.py                 # MARKET_DATA_PROVIDER env var
├── core/
│   ├── sec_value_screen.py   # XBRL ingestion, TTM, ratio pipeline (the engine)
│   ├── screening.py          # thresholds, quality screen, pagination, search, watchlist
│   ├── valuation.py          # Graham Number, DCF intrinsic value, margin of safety
│   ├── company_analysis.py   # red flags, forensic checks, trend tables
│   ├── formatting.py         # display formatting, SIC→sector mapping, Jinja filters
│   ├── exchange_filter.py    # NASDAQ/NYSE filter via SEC exchange listing
│   ├── data_ingestion.py     # ZIP extraction, directory resolution, DB connection
│   ├── types.py              # shared TypedDicts
│   └── paths.py              # shared filesystem paths
├── providers/
│   ├── pipeline_imports.py   # single source of truth for core imports
│   ├── market_data.py        # provider-agnostic price/EV/multiples join logic
│   ├── yfinance_provider.py  # yfinance-based price + CIK→ticker mapping
│   └── schwab/
│       ├── auth.py           # OAuth2 token management (auto-refresh)
│       ├── market_data.py    # batch quote fetching
│       └── setup.py          # one-time interactive OAuth bootstrap
├── templates/                # Jinja2 + HTMX templates
│   ├── base.html             # shell: nav, fonts, Tailwind config
│   ├── screen.html           # main screening page (sliders + results)
│   ├── _results_table.html   # HTMX partial: table, pagination, diagnostic rows
│   ├── _ingest_progress.html # HTMX partial: polling progress bar
│   ├── company.html          # company deep-dive (trends, red flags, ratios)
│   ├── watchlist.html        # saved companies with entry vs current prices
│   ├── home.html             # dashboard with ingest summary
│   └── setup.html            # data ingestion UI
├── static/                   # CSS
├── tests/                    # pytest suite (122 tests)
├── scripts/                  # CLI diagnostic tools
├── cache/                    # runtime-generated (gitignored)
│   ├── screen.duckdb         # DuckDB database (fundamentals, prices, watchlist)
│   ├── extracted/            # unzipped SEC data
│   ├── ticker_map.json       # cached CIK→ticker mapping
│   └── exchange_map.json     # cached CIK→exchange mapping
└── data/                     # user's downloaded SEC quarterly zips (gitignored)
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000/.

### First-time setup

1. Download one or more quarterly ZIP files from the [SEC DERA Financial Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets) page.
2. Place them in `./data/` (or any directory).
3. Go to http://127.0.0.1:8000/setup, point the data directory at your ZIP file(s), and click **Ingest**.
4. After ingestion completes, go to http://127.0.0.1:8000/screen to start screening.

### Schwab API (optional)

If you have Schwab developer access:

```bash
python3 -m providers.schwab.setup --client-id YOUR_CLIENT_ID --client-secret YOUR_SECRET --redirect-uri https://127.0.0.1
```

Then set the environment variable:

```bash
export MARKET_DATA_PROVIDER=schwab
```

Schwab provides batch quotes (one API call for all tickers instead of one per ticker), plus pre-computed market cap and fundamental ratios for cross-checking.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

## How to get the data

The [SEC DERA Financial Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets) page publishes quarterly ZIP archives containing:

| File | Contents |
|---|---|
| `sub.txt` | Submission metadata — CIK, company name, form type (10-K/10-Q), filing date, fiscal period, SIC code |
| `num.txt` | Numeric XBRL facts — tag name, value, quarter count (`qtrs`), coregistrant, reporting date (`ddate`) |
| `tag.txt` | Tag definitions — label, description, whether it's a custom/extension tag |
| `pre.txt` | Presentation structure — how tags are grouped/ordered in financial statements |

The app reads `sub.txt` and `num.txt` directly via DuckDB's CSV reader, standardizes XBRL tags across different naming conventions (e.g. `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`), and pivots into a wide fundamentals table.

## Limitations

- **Single-quarter data dumps** reflect filings *accepted* in that quarter, not necessarily fiscal periods ending in that quarter. Companies with off-cycle fiscal years, restatements, or late filings may be missing from any single dump. Ingesting multiple consecutive quarterly dumps (which this app supports) mitigates this.
- **Tag standardization** covers the common cases. Some filers use custom extension tags that won't match — those show up as NULL. Check the `tag` table (`custom=1`) if a company you care about has missing data.
- **No qualitative/footnote data** — lease commitments buried in footnotes, contingent liabilities, or segment-level detail are not captured. This is a quantitative first-pass screen, not a substitute for reading the 10-K.
- **ROIC uses a simplified Greenblatt invested-capital proxy** (NWC + Net Fixed Assets). The original Greenblatt formula also excludes excess cash and non-interest-bearing current liabilities more precisely. Operating lease liabilities (ASC 842) are included in Total Debt when the tag is present.
- **Price is live, fundamentals are as-of the filing period** — there's an inherent lag. Greenblatt's original methodology has the same property.
- **Financial sector companies** (SIC 6000–6799) have fundamentally different balance sheets — NWC and Net PPE are not meaningful for them, so ROIC is nonsensical. Use the "Exclude financials" toggle to filter them out.
