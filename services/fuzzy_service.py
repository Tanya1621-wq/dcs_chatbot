"""Fuzzy string matching to complement semantic search.

Helps with literal-keyword overlap (e.g. error codes, command names like
"503", "fallow") that embeddings can sometimes underweight.
"""
from __future__ import annotations

from rapidfuzz import fuzz


def fuzzy_score(query: str, title: str, category: str) -> float:
    """Return a 0..1 fuzzy score combining title + category matches."""
    if not query.strip():
        return 0.0
    title_score = fuzz.token_set_ratio(query, title or "") / 100.0
    cat_score = fuzz.token_set_ratio(query, category or "") / 100.0
    # Title is the dominant signal; category nudges ties.
    return 0.8 * title_score + 0.2 * cat_score


def score_all(query: str, entries) -> dict[int, float]:
    """Return {row_id: score} for every KB entry."""
    return {
        e.row_id: fuzzy_score(query, e.title, e.category) for e in entries
    }
