"""Reindex pipeline. Walks me_bot/corpus/, chunks each file, embeds, writes
me_bot/index.json. Called from the admin tab; also runnable directly via
`python -m me_bot.reindex`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .chunker import chunk_markdown
from .embedder import DIM, MODEL_NAME, embed_texts

CORPUS_DIR = Path(__file__).parent / "corpus"
INDEX_PATH = Path(__file__).parent / "index.json"


def reindex() -> dict:
    all_chunks = []
    for md in sorted(CORPUS_DIR.rglob("*.md")):
        content = md.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown(md, content, CORPUS_DIR))

    if not all_chunks:
        return {"status": "no chunks", "count": 0, "size_mb": 0.0}

    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts)

    index = {
        "version": "1.0",
        "model": MODEL_NAME,
        "dimension": DIM,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [
            {**c.to_dict(), "embedding": embeddings[i].tolist()}
            for i, c in enumerate(all_chunks)
        ],
    }

    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "count": len(all_chunks),
        "size_mb": INDEX_PATH.stat().st_size / 1024 / 1024,
    }


if __name__ == "__main__":
    result = reindex()
    print(json.dumps(result, indent=2))
