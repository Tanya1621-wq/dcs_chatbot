"""Hybrid retrieval: semantic (FAISS) + fuzzy (rapidfuzz) + cross-encoder rerank.

Two-stage pipeline:
  1. Hybrid bi-encoder + fuzzy → wide pool of candidates (fast).
  2. Cross-encoder reranker → precise ranking of the top-K pool (accurate).

The chatbot calls `HybridSearcher.search(query)`; if confidence is below
the configured threshold the caller should fall back to Groq.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from services import fuzzy_service, kb_service, rerank_service
from services.embedding_service import FaissIndex
from utils.config import (
    CONFIDENCE_THRESHOLD,
    FUZZY_WEIGHT,
    RERANK_TOP_K,
    SEMANTIC_WEIGHT,
    TOP_K,
)
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SearchResult:
    entry: kb_service.KBEntry
    semantic: float
    fuzzy: float
    score: float
    hybrid: float = 0.0   # raw hybrid score before rerank (for transparency)
    rerank: float = 0.0   # cross-encoder score; 0.0 if reranker not used

    @property
    def confidence(self) -> float:
        return self.score


class HybridSearcher:
    """Holds a FAISS index + cached KB rows. Rebuild after KB changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._index: Optional[FaissIndex] = None
        self._entries: list[kb_service.KBEntry] = []
        self._by_id: dict[int, kb_service.KBEntry] = {}

    # --- lifecycle ----------------------------------------------------------

    def is_ready(self) -> bool:
        return self._index is not None and bool(self._entries)

    def ensure_loaded(self) -> bool:
        """Try to load existing FAISS index; rebuild if missing or stale."""
        with self._lock:
            entries = kb_service.get_all_entries()
            if not entries:
                log.warning("KB is empty — no entries to index.")
                self._index = None
                self._entries = []
                self._by_id = {}
                return False

            cached = FaissIndex.load()
            entry_ids = [e.row_id for e in entries]
            if cached is not None and cached.row_ids == entry_ids:
                self._index = cached
            else:
                texts = [e.search_text() for e in entries]
                self._index = FaissIndex.build(entry_ids, texts)
                self._index.save()

            self._entries = entries
            self._by_id = {e.row_id: e for e in entries}
            return True

    def rebuild(self) -> bool:
        """Force a fresh build (use after KB upload)."""
        with self._lock:
            self._index = None
            self._entries = []
            self._by_id = {}
        return self.ensure_loaded()

    # --- search -------------------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K) -> list[SearchResult]:
        if not self.is_ready():
            self.ensure_loaded()
        if not self.is_ready() or not query.strip():
            return []

        query_norm = _preprocess(query)

        # --- Stage 1: hybrid retrieval (bi-encoder + fuzzy) -----------------
        # Pull a wide pool so the reranker has enough material to work with.
        sem_pool = max(RERANK_TOP_K, top_k * 4, 10)
        sem_pairs = self._index.search(query_norm, top_k=sem_pool)
        sem_scores: dict[int, float] = {row_id: s for row_id, s in sem_pairs}

        fuz_scores = fuzzy_service.score_all(query_norm, self._entries)

        candidate_ids = set(sem_scores) | {
            rid for rid, s in fuz_scores.items() if s >= 0.5
        }

        results: list[SearchResult] = []
        for rid in candidate_ids:
            entry = self._by_id.get(rid)
            if entry is None:
                continue
            sem = sem_scores.get(rid, 0.0)
            fuz = fuz_scores.get(rid, 0.0)
            hybrid = SEMANTIC_WEIGHT * sem + FUZZY_WEIGHT * fuz
            results.append(
                SearchResult(
                    entry=entry,
                    semantic=sem,
                    fuzzy=fuz,
                    hybrid=hybrid,
                    rerank=0.0,
                    score=hybrid,
                )
            )

        results.sort(key=lambda r: r.hybrid, reverse=True)

        # --- Stage 2: cross-encoder rerank on the top-K pool ----------------
        if rerank_service.is_available() and results:
            pool = results[:RERANK_TOP_K]
            pairs = [(r.entry.row_id, r.entry.search_text()) for r in pool]
            rerank_scores = rerank_service.rerank(query_norm, pairs)
            if rerank_scores:
                for r in pool:
                    r.rerank = rerank_scores.get(r.entry.row_id, 0.0)
                    # Cross-encoder is more accurate — promote it to the
                    # authoritative ranking signal.
                    r.score = r.rerank
                pool.sort(key=lambda r: r.score, reverse=True)
                return pool[:top_k]

        return results[:top_k]

    @staticmethod
    def is_low_confidence(results: list[SearchResult]) -> bool:
        if not results:
            return True
        return results[0].score < CONFIDENCE_THRESHOLD


def _preprocess(text: str) -> str:
    """Light preprocessing — lowercase + collapse whitespace.

    No stemming or stop-word removal: the multilingual model handles morphology
    and we want to preserve Hindi/Hinglish tokens verbatim.
    """
    return " ".join(text.lower().strip().split())


# Module-level singleton — Streamlit caches this via @st.cache_resource.
_singleton: Optional[HybridSearcher] = None


def get_searcher() -> HybridSearcher:
    global _singleton
    if _singleton is None:
        _singleton = HybridSearcher()
    return _singleton
