"""Centralized configuration loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "database.db"

# Knowledge base may arrive as .xlsx (preferred) or .csv (accepted).
KB_XLSX_PATH = DATA_DIR / "knowledge_base.xlsx"
KB_CSV_PATH = DATA_DIR / "knowledge_base.csv"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "kb.index"
FAISS_META_PATH = VECTOR_STORE_DIR / "kb_meta.pkl"

for d in (DATA_DIR, VECTOR_STORE_DIR, DB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Models ---
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Secrets ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

# --- Retrieval tuning ---
SEMANTIC_WEIGHT = _get_float("SEMANTIC_WEIGHT", 0.7)
FUZZY_WEIGHT = _get_float("FUZZY_WEIGHT", 0.3)
CONFIDENCE_THRESHOLD = _get_float("CONFIDENCE_THRESHOLD", 0.45)
TOP_K = 3

# --- Reranker (second-stage cross-encoder) ---
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-v2-m3",
)
# How many top hybrid candidates to send to the reranker.
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "15"))

# Expected KB column names (case-insensitive on load).
KB_COLUMNS = [
    "issue_id",
    "issue_title",
    "issue_category",
    "issue_description",
    "resolution_steps",
]
