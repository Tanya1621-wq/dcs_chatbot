"""Cross-encoder reranker — second stage of two-stage retrieval.

The bi-encoder (FAISS) is fast but encodes query and passage independently,
so it loses some accuracy. A cross-encoder takes (query, passage) together
and produces a much more precise relevance score. We run it on the top-K
hybrid candidates to refine the final ranking.

If the model fails to load (e.g. offline first run, no disk space), the
service degrades gracefully — search falls back to hybrid-only ranking.
"""
from __future__ import annotations

import math
import threading

from utils.config import RERANKER_ENABLED, RERANKER_MODEL_NAME
from utils.logger import get_logger

log = get_logger(__name__)

_model = None
_model_lock = threading.Lock()
_load_failed = False


def is_available() -> bool:
    """True if reranker is enabled in config and hasn't permanently failed."""
    return RERANKER_ENABLED and not _load_failed


def _load_model():
    """Lazy-load the CrossEncoder on first use; cache for the process."""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed or not RERANKER_ENABLED:
        return None
    with _model_lock:
        if _model is None and not _load_failed:
            try:
                from sentence_transformers import CrossEncoder

                log.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
                _model = CrossEncoder(RERANKER_MODEL_NAME)
            except Exception as exc:
                log.error(
                    "Reranker failed to load (%s); disabling for this session.",
                    exc,
                )
                _load_failed = True
                return None
    return _model


def _sigmoid(x: float) -> float:
    """Map a raw logit to [0, 1]. BGE rerankers output logits, not probabilities."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def rerank(query: str, candidates: list[tuple[int, str]]) -> dict[int, float]:
    """Score (row_id, text) pairs against the query.

    Returns {row_id: score in [0, 1]} for each candidate. Returns an empty
    dict if the reranker is unavailable or the call fails — callers should
    treat that as "no rerank info" and fall back to their existing ranking.
    """
    if not candidates or not query.strip():
        return {}
    model = _load_model()
    if model is None:
        return {}
    try:
        pairs = [[query, text] for _, text in candidates]
        raw_scores = model.predict(pairs, show_progress_bar=False)
        return {
            row_id: _sigmoid(float(score))
            for (row_id, _), score in zip(candidates, raw_scores)
        }
    except Exception as exc:
        log.error("Rerank failed: %s", exc)
        return {}
