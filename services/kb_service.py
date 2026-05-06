"""Knowledge-base ingestion + SQLite persistence + query/feedback logs."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.config import (
    DB_PATH,
    KB_COLUMNS,
    KB_CSV_PATH,
    KB_XLSX_PATH,
)
from utils.logger import get_logger

log = get_logger(__name__)
_db_lock = threading.Lock()


@dataclass
class KBEntry:
    row_id: int
    issue_id: str
    title: str
    category: str
    description: str
    resolution_steps: str
    paraphrases: list[str] = field(default_factory=list)

    def search_text(self) -> str:
        # Indexing the resolution + LLM-generated paraphrases boosts recall
        # on verbose / multilingual queries.
        parts = [self.title, self.category, self.description, self.resolution_steps]
        parts.extend(self.paraphrases)
        return " | ".join(p for p in parts if p)


SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_entries (
    row_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id            TEXT,
    title               TEXT NOT NULL,
    category            TEXT,
    description         TEXT,
    resolution_steps    TEXT NOT NULL,
    paraphrases         TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

CREATE TABLE IF NOT EXISTS query_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    query           TEXT NOT NULL,
    matched_row_id  INTEGER,
    confidence      REAL,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    query_log_id    INTEGER,
    helpful         INTEGER NOT NULL,
    FOREIGN KEY (query_log_id) REFERENCES query_logs(id)
);
"""


@contextmanager
def _conn():
    with _db_lock:
        c = sqlite3.connect(DB_PATH, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()


def init_db() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)
        # Migration for older DBs created before paraphrases existed.
        try:
            c.execute("ALTER TABLE kb_entries ADD COLUMN paraphrases TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists


# --- KB ingest -------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=rename)

    # Tolerate common alternates from the spec.
    aliases = {
        "title": "issue_title",
        "category": "issue_category",
        "description": "issue_description",
    }
    for alt, canon in aliases.items():
        if alt in df.columns and canon not in df.columns:
            df = df.rename(columns={alt: canon})

    missing = [c for c in ("issue_title", "resolution_steps") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Knowledge base is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    for col in KB_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[KB_COLUMNS].fillna("").astype(str)
    df = df[df["issue_title"].str.strip() != ""]
    return df.reset_index(drop=True)


def _read_kb_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported KB file type: {suffix}")
    return _normalize_columns(df)


def resolve_kb_path() -> Optional[Path]:
    if KB_XLSX_PATH.exists():
        return KB_XLSX_PATH
    if KB_CSV_PATH.exists():
        return KB_CSV_PATH
    return None


def load_kb_into_db(path: Optional[Path] = None, replace: bool = True) -> int:
    """Load XLSX/CSV into SQLite. Returns number of rows ingested."""
    init_db()
    src = path or resolve_kb_path()
    if src is None:
        log.warning("No knowledge base file found at %s or %s", KB_XLSX_PATH, KB_CSV_PATH)
        return 0

    df = _read_kb_file(src)
    log.info("Loading %d KB rows from %s", len(df), src.name)

    with _conn() as c:
        if replace:
            c.execute("DELETE FROM kb_entries")
        c.executemany(
            "INSERT INTO kb_entries (issue_id, title, category, description, resolution_steps) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    r["issue_id"],
                    r["issue_title"],
                    r["issue_category"],
                    r["issue_description"],
                    r["resolution_steps"],
                )
                for _, r in df.iterrows()
            ],
        )
        c.execute(
            "INSERT INTO meta (key, value) VALUES ('last_updated', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (datetime.now(timezone.utc).isoformat(),),
        )
    return len(df)


def save_uploaded_kb(file_bytes: bytes, filename: str) -> Path:
    """Persist uploaded KB to disk, choosing extension from filename."""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".xlsx", ".csv"):
        raise ValueError("Knowledge base must be a .xlsx or .csv file")
    target = KB_XLSX_PATH if suffix == ".xlsx" else KB_CSV_PATH
    # Remove the other format so resolve_kb_path() points to the new one.
    other = KB_CSV_PATH if suffix == ".xlsx" else KB_XLSX_PATH
    if other.exists():
        other.unlink()
    target.write_bytes(file_bytes)
    return target


# --- KB read ---------------------------------------------------------------

def _row_to_entry(r) -> KBEntry:
    raw = r["paraphrases"] if "paraphrases" in r.keys() else None
    paraphrases: list[str] = []
    if raw:
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                paraphrases = [str(p) for p in loaded]
        except (json.JSONDecodeError, TypeError):
            paraphrases = []
    return KBEntry(
        row_id=r["row_id"],
        issue_id=r["issue_id"] or "",
        title=r["title"] or "",
        category=r["category"] or "",
        description=r["description"] or "",
        resolution_steps=r["resolution_steps"] or "",
        paraphrases=paraphrases,
    )


def get_all_entries() -> list[KBEntry]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT row_id, issue_id, title, category, description, "
            "resolution_steps, paraphrases "
            "FROM kb_entries ORDER BY row_id"
        ).fetchall()
    return [_row_to_entry(r) for r in rows]


def get_entry(row_id: int) -> Optional[KBEntry]:
    with _conn() as c:
        r = c.execute(
            "SELECT row_id, issue_id, title, category, description, "
            "resolution_steps, paraphrases "
            "FROM kb_entries WHERE row_id = ?",
            (row_id,),
        ).fetchone()
    if r is None:
        return None
    return _row_to_entry(r)


def entries_missing_paraphrases() -> list[KBEntry]:
    """Return entries that don't have paraphrases yet (empty or NULL)."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT row_id, issue_id, title, category, description, "
            "resolution_steps, paraphrases "
            "FROM kb_entries "
            "WHERE paraphrases IS NULL OR paraphrases = '' OR paraphrases = '[]' "
            "ORDER BY row_id"
        ).fetchall()
    return [_row_to_entry(r) for r in rows]


def set_paraphrases(row_id: int, paraphrases: list[str]) -> None:
    """Save the LLM-generated paraphrases for one KB row."""
    init_db()
    with _conn() as c:
        c.execute(
            "UPDATE kb_entries SET paraphrases = ? WHERE row_id = ?",
            (json.dumps(paraphrases, ensure_ascii=False), row_id),
        )


# --- KB export -------------------------------------------------------------

def export_kb_dataframe() -> pd.DataFrame:
    """Return the KB as a DataFrame with CSV-compatible column names.

    The downloaded file can be re-uploaded as-is via the admin panel; the
    extra `paraphrases` column is preserved for round-tripping but is
    ignored by the upload path.
    """
    entries = get_all_entries()
    rows = [
        {
            "issue_id": e.issue_id,
            "issue_title": e.title,
            "issue_category": e.category,
            "issue_description": e.description,
            "resolution_steps": e.resolution_steps,
            "paraphrases": (
                json.dumps(e.paraphrases, ensure_ascii=False) if e.paraphrases else ""
            ),
        }
        for e in entries
    ]
    return pd.DataFrame(rows, columns=[
        "issue_id", "issue_title", "issue_category",
        "issue_description", "resolution_steps", "paraphrases",
    ])


def export_kb_csv() -> bytes:
    """Return the KB as UTF-8 encoded CSV bytes."""
    return export_kb_dataframe().to_csv(index=False).encode("utf-8-sig")


def export_kb_xlsx() -> bytes:
    """Return the KB as XLSX bytes (single sheet 'knowledge_base')."""
    import io

    df = export_kb_dataframe()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="knowledge_base")
    return buf.getvalue()


def get_last_updated() -> Optional[str]:
    with _conn() as c:
        r = c.execute("SELECT value FROM meta WHERE key='last_updated'").fetchone()
    return r["value"] if r else None


def kb_count() -> int:
    init_db()
    with _conn() as c:
        return c.execute("SELECT COUNT(*) AS n FROM kb_entries").fetchone()["n"]


# --- Logging ---------------------------------------------------------------

def log_query(
    query: str,
    matched_row_id: Optional[int],
    confidence: Optional[float],
    source: str,
) -> int:
    init_db()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO query_logs (ts, query, matched_row_id, confidence, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                query,
                matched_row_id,
                confidence,
                source,
            ),
        )
        return cur.lastrowid


def log_feedback(query_log_id: int, helpful: bool) -> None:
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO feedback (ts, query_log_id, helpful) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), query_log_id, 1 if helpful else 0),
        )


def feedback_summary() -> dict:
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT "
            "  SUM(CASE WHEN helpful=1 THEN 1 ELSE 0 END) AS yes_count, "
            "  SUM(CASE WHEN helpful=0 THEN 1 ELSE 0 END) AS no_count, "
            "  COUNT(*) AS total "
            "FROM feedback"
        ).fetchone()
    return {
        "yes": row["yes_count"] or 0,
        "no": row["no_count"] or 0,
        "total": row["total"] or 0,
    }


def recent_queries(limit: int = 20) -> list[sqlite3.Row]:
    init_db()
    with _conn() as c:
        return list(
            c.execute(
                "SELECT ts, query, confidence, source FROM query_logs "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        )
