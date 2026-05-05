"""Demo 06 — the me-bot chat surface.

A retrieval-augmented chat about Raven's published professional work.
Tightly scoped, with refusal posture enforced via the system prompt
and a regex moderator backstop.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from me_bot.generator import NO_CONTEXT_RESPONSE, generate_stream
from me_bot.moderator import check_output
from me_bot.retriever import retrieve

DATA_DIR = Path(__file__).parent.parent / "data"
QUERY_LOG = DATA_DIR / "me_bot_query_log.jsonl"
BLOCKED_LOG = DATA_DIR / "me_bot_blocked.jsonl"
ERROR_LOG = DATA_DIR / "me_bot_errors.jsonl"
KILLSWITCH = Path(__file__).parent.parent / "me_bot" / ".disabled"

MAX_HISTORY = 6
RATE_LIMIT_PER_SESSION = 20

REFUSAL_BLOCKED = (
    "I'd rather not answer that one — it's outside what I can confirm "
    "from public sources."
)


def render(embed: bool = False) -> None:
    if KILLSWITCH.exists():
        st.info("The lab assistant is temporarily offline. Use the contact form at /contact.")
        return

    if not embed:
        st.caption(
            "Retrieval-augmented assistant for Raven's professional and research work. "
            "Tightly scoped — won't speculate or invent. ~30 questions/min across all visitors."
        )

    if "me_bot_history" not in st.session_state:
        st.session_state.me_bot_history = []

    # render history
    for msg in st.session_state.me_bot_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # rate limit
    user_turns = sum(1 for m in st.session_state.me_bot_history if m["role"] == "user")
    if user_turns >= RATE_LIMIT_PER_SESSION:
        st.warning(
            "You've hit the per-session question limit. Refresh tomorrow, "
            "or use the contact form for anything specific."
        )
        return

    query = st.chat_input("Ask about Raven's work…")
    if not query:
        return

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching the corpus…"):
        chunks = retrieve(query, k=5, min_score=0.25)

    with st.chat_message("assistant"):
        if not chunks:
            st.markdown(NO_CONTEXT_RESPONSE)
            response_text = NO_CONTEXT_RESPONSE
            sources_to_render: list[dict] = []
        else:
            placeholder = st.empty()
            response_text = ""
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.me_bot_history[-MAX_HISTORY:]
                ]
                for token in generate_stream(query, chunks, history):
                    response_text += token
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            except Exception as e:
                placeholder.markdown(
                    "The lab is briefly unreachable. Try again in a moment, or use the contact form."
                )
                _log(ERROR_LOG, {"query": query, "error": str(e)})
                return

            ok, reason = check_output(response_text)
            if not ok:
                placeholder.markdown(REFUSAL_BLOCKED)
                _log(BLOCKED_LOG, {
                    "query": query,
                    "blocked_response": response_text,
                    "reason": reason,
                    "chunk_ids": [c["chunk_id"] for c in chunks],
                })
                response_text = REFUSAL_BLOCKED
                sources_to_render = []
            else:
                sources_to_render = chunks
                _render_sources(chunks)

    st.session_state.me_bot_history.append({"role": "user", "content": query})
    st.session_state.me_bot_history.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources_to_render,
    })

    _log(QUERY_LOG, {
        "query": query,
        "response": response_text,
        "chunk_ids": [c["chunk_id"] for c in (sources_to_render or chunks or [])],
    })


def _render_sources(chunks: list[dict]) -> None:
    if not chunks:
        return
    with st.expander(f"Sources · {len(chunks)} chunks"):
        for c in chunks:
            st.caption(
                f"**[{c['chunk_id']}]** · score {c['score']:.2f} · reviewed {c['last_reviewed']}"
            )
            preview = c["text"][:300] + ("…" if len(c["text"]) > 300 else "")
            st.markdown(f"> {preview}")


def _log(path: Path, record: dict) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
