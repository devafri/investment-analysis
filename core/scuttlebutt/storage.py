"""DuckDB persistence for scuttlebutt research notes."""

import json
import uuid
from typing import List, Optional

from core.fundamentals.data_ingestion import get_db_connection
from core.scuttlebutt.models import ScuttlebuttNote, CompanyAssessment, Conviction


def _ensure_tables(con) -> None:
    """Create scuttlebutt tables if they don't exist yet."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS scuttlebutt_notes (
            id VARCHAR PRIMARY KEY,
            company_cik VARCHAR,
            company_name VARCHAR,
            category VARCHAR,
            conviction VARCHAR,
            title VARCHAR,
            body TEXT,
            source_type VARCHAR,
            source_quality INTEGER,
            tags VARCHAR,          -- JSON array stored as string
            created_at VARCHAR,
            updated_at VARCHAR
        )
    """)


def save_note(note: ScuttlebuttNote) -> str:
    """Persist a single scuttlebutt note.  Returns the note ID."""
    if not note.id:
        note.id = str(uuid.uuid4())
    con = get_db_connection()
    try:
        _ensure_tables(con)
        con.execute(
            """INSERT OR REPLACE INTO scuttlebutt_notes
               (id, company_cik, company_name, category, conviction,
                title, body, source_type, source_quality, tags,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                note.id, note.company_cik, note.company_name,
                note.category.value, note.conviction.value,
                note.title, note.body, note.source_type,
                note.source_quality, json.dumps(note.tags or []),
                note.created_at, note.updated_at,
            ],
        )
        con.commit()
    finally:
        con.close()
    return note.id


def get_notes_for_cik(cik: str, limit: int = 50) -> List[ScuttlebuttNote]:
    """Return all scuttlebutt notes for a given company CIK."""
    con = get_db_connection()
    try:
        _ensure_tables(con)
        rows = con.execute(
            "SELECT * FROM scuttlebutt_notes "
            "WHERE company_cik = ? "
            "ORDER BY created_at DESC LIMIT ?",
            [str(cik), limit],
        ).fetchall()
    finally:
        con.close()

    notes = []
    for r in rows:
        try:
            tags = json.loads(r[9]) if r[9] else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        notes.append(ScuttlebuttNote(
            id=r[0], company_cik=r[1], company_name=r[2],
            category=r[3], conviction=r[4], title=r[5], body=r[6],
            source_type=r[7] or "", source_quality=r[8] or 3,
            tags=tags, created_at=r[10] or "", updated_at=r[11] or "",
        ))
    return notes


def get_all_notes(limit: int = 100) -> List[ScuttlebuttNote]:
    """Return recent scuttlebutt notes across all companies."""
    con = get_db_connection()
    try:
        _ensure_tables(con)
        rows = con.execute(
            "SELECT * FROM scuttlebutt_notes "
            "ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
    finally:
        con.close()

    notes = []
    for r in rows:
        try:
            tags = json.loads(r[9]) if r[9] else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        notes.append(ScuttlebuttNote(
            id=r[0], company_cik=r[1], company_name=r[2],
            category=r[3], conviction=r[4], title=r[5], body=r[6],
            source_type=r[7] or "", source_quality=r[8] or 3,
            tags=tags, created_at=r[10] or "", updated_at=r[11] or "",
        ))
    return notes


def get_company_assessment(cik: str) -> Optional[CompanyAssessment]:
    """Aggregate all notes for a company into an assessment."""
    notes = get_notes_for_cik(cik, limit=500)
    if not notes:
        return None

    bullish = sum(1 for n in notes if n.conviction == Conviction.BULLISH)
    bearish = sum(1 for n in notes if n.conviction == Conviction.BEARISH)
    neutral = sum(1 for n in notes if n.conviction == Conviction.NEUTRAL)
    avg_quality = (
        sum(n.source_quality for n in notes) / len(notes) if notes else 0
    )

    # Top tags
    tag_counts: dict = {}
    for n in notes:
        for t in n.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:10]

    return CompanyAssessment(
        company_cik=cik,
        company_name=notes[0].company_name or f"CIK {cik}",
        bullish_notes=bullish,
        bearish_note=bearish,
        neutral_notes=neutral,
        avg_source_quality=round(avg_quality, 1),
        top_tags=top_tags,
        last_updated=max(n.updated_at or n.created_at for n in notes),
    )
