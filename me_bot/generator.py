"""Groq client + prompt builder. Llama 3.3 70B at temperature 0.2."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Iterable

from lib.secrets import get_secret

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")

NO_CONTEXT_RESPONSE = (
    "I don't have public information about that. I only cover Raven's "
    "professional and research work — for anything else, the contact "
    "form at the site's /contact page is the right route."
)


@functools.lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq
    return Groq(api_key=get_secret("groq", "api_key"))


def build_context_block(chunks: list[dict]) -> str:
    if not chunks:
        return "[no relevant context found]"
    blocks = []
    for c in chunks:
        blocks.append(
            f"--- chunk: [{c['chunk_id']}] (last reviewed {c['last_reviewed']}) ---\n"
            f"{c['text']}\n"
        )
    return "\n".join(blocks)


def build_messages(query: str, chunks: list[dict], history: list[dict]) -> list[dict]:
    context_block = build_context_block(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"# Retrieved context\n\n{context_block}"},
        *history,
        {"role": "user", "content": query},
    ]


def generate_stream(query: str, chunks: list[dict], history: list[dict]) -> Iterable[str]:
    client = get_groq_client()
    messages = build_messages(query, chunks, history)
    model = get_secret("groq", "model", "llama-3.3-70b-versatile")
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=400,
        top_p=0.9,
        stream=True,
    )
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            yield delta


def generate_full(query: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    """Non-streaming variant — used by the eval harness."""
    return "".join(generate_stream(query, chunks, history or []))
