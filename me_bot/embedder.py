"""Sentence-transformers wrapper. Singleton model on CPU."""

from __future__ import annotations

import functools

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


@functools.lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
