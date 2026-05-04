"""Sentence-transformer embeddings + FAISS vector index.

Uses paraphrase-multilingual-MiniLM-L12-v2 so Hindi, Hinglish, and English
queries map into the same vector space as the KB entries.
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from utils.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
)
from utils.logger import get_logger

log = get_logger(__name__)

_model = None
_model_lock = threading.Lock()


def _load_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Imported lazily so import-time cost only hits when used.
                from sentence_transformers import SentenceTransformer

                log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    model = _load_model()
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32")


class FaissIndex:
    """Thin wrapper around a flat inner-product FAISS index.

    Embeddings are L2-normalized, so inner product == cosine similarity.
    `row_ids` aligns positions in the index back to KB row_id values.
    """

    def __init__(self, index, row_ids: list[int], dim: int):
        self.index = index
        self.row_ids = row_ids
        self.dim = dim

    @classmethod
    def build(cls, row_ids: list[int], texts: list[str]) -> "FaissIndex":
        import faiss

        if not texts:
            raise ValueError("Cannot build FAISS index with zero KB entries")
        vecs = embed(texts)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        log.info("Built FAISS index with %d vectors (dim=%d)", len(texts), dim)
        return cls(index=index, row_ids=list(row_ids), dim=dim)

    def save(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_META_PATH,
    ) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"row_ids": self.row_ids, "dim": self.dim}, f)
        log.info("Saved FAISS index to %s", index_path)

    @classmethod
    def load(
        cls,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_META_PATH,
    ) -> Optional["FaissIndex"]:
        import faiss

        if not index_path.exists() or not meta_path.exists():
            return None
        try:
            index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            return cls(index=index, row_ids=meta["row_ids"], dim=meta["dim"])
        except Exception as exc:  # pragma: no cover - corruption recovery
            log.warning("Failed to load FAISS index (%s); will rebuild.", exc)
            return None

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return [(row_id, similarity_score in [0,1])] for the top_k matches."""
        if not query.strip():
            return []
        vec = embed([query])
        k = min(top_k, len(self.row_ids))
        scores, idxs = self.index.search(vec, k)
        out: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.row_ids):
                continue
            # IP on normalized vectors is in [-1, 1]; clamp to [0, 1] for ranking.
            similarity = max(0.0, min(1.0, float(score)))
            out.append((self.row_ids[idx], similarity))
        return out
