"""Cosine top-k retrieval over the JSON index."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from .embedder import embed_query

INDEX_PATH = Path(__file__).parent / "index.json"


@functools.lru_cache(maxsize=1)
def load_index() -> tuple[list[dict], np.ndarray]:
    if not INDEX_PATH.exists():
        return [], np.zeros((0, 384), dtype=np.float32)
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    return chunks, matrix


def retrieve(query: str, k: int = 5, min_score: float = 0.25) -> list[dict]:
    chunks, matrix = load_index()
    if not chunks:
        return []
    q = embed_query(query)
    scores = matrix @ q
    top_idx = np.argsort(-scores)[:k]
    out: list[dict] = []
    for i in top_idx:
        s = float(scores[i])
        if s < min_score:
            break
        c = chunks[int(i)]
        out.append({
            "chunk_id": c["chunk_id"],
            "title": c["title"],
            "section": c["section"],
            "text": c["text"],
            "score": s,
            "last_reviewed": c["last_reviewed"],
        })
    return out


def reset_cache() -> None:
    load_index.cache_clear()
