"""Tab 7 — me-bot admin: index status, reindex, query log, blocked review,
eval harness, killswitch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.cache import clear_resource_caches
from me_bot.eval_runner import run_evals
from me_bot.reindex import INDEX_PATH, reindex
from me_bot.retriever import reset_cache as reset_retriever_cache

DATA_DIR = Path(__file__).parent.parent / "data"
QUERY_LOG = DATA_DIR / "me_bot_query_log.jsonl"
BLOCKED_LOG = DATA_DIR / "me_bot_blocked.jsonl"
KILLSWITCH = Path(__file__).parent.parent / "me_bot" / ".disabled"


def render() -> None:
    st.subheader("Me-bot admin")
    sub = st.radio(
        "Section",
        ["Index", "Query log", "Blocked outputs", "Evals", "Killswitch"],
        horizontal=True,
    )
    if sub == "Index":
        _render_index()
    elif sub == "Query log":
        _render_log(QUERY_LOG, "queries")
    elif sub == "Blocked outputs":
        _render_log(BLOCKED_LOG, "blocked")
    elif sub == "Evals":
        _render_evals()
    elif sub == "Killswitch":
        _render_killswitch()


def _render_index() -> None:
    if INDEX_PATH.exists():
        idx = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks", len(idx.get("chunks", [])))
        c2.metric("Built at", str(idx.get("built_at", ""))[:19])
        c3.metric("Size (MB)", f"{INDEX_PATH.stat().st_size / 1024 / 1024:.2f}")
    else:
        st.warning("Index not yet built. Click **Rebuild** below.")

    if st.button("Rebuild index now", type="primary"):
        with st.spinner("Embedding corpus…"):
            try:
                result = reindex()
            except Exception as e:
                st.error(f"Reindex failed: {e}")
                return
        st.success(f"Rebuilt: {result['count']} chunks, {result['size_mb']:.2f} MB")
        reset_retriever_cache()
        clear_resource_caches()


def _render_log(path: Path, kind: str) -> None:
    if not path.exists():
        st.info(f"No {kind} logged yet.")
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        st.info(f"No valid {kind} entries.")
        return
    df = pd.DataFrame(rows)
    today = datetime.now(timezone.utc).date().isoformat()
    today_count = sum(1 for r in rows if r.get("ts", "").startswith(today))
    c1, c2 = st.columns(2)
    c1.metric(f"Total {kind}", len(df))
    c2.metric("Today", today_count)
    st.dataframe(df.tail(200), use_container_width=True, hide_index=True)
    st.download_button(
        f"Download {kind} CSV",
        df.to_csv(index=False).encode(),
        file_name=f"{kind}-{today}.csv",
        mime="text/csv",
    )


def _render_evals() -> None:
    st.caption(
        "Regression evals from `me_bot/evals.yaml`. Run before any corpus change ships. "
        "100% pass rate is the bar."
    )
    if st.button("Run evals", type="primary"):
        with st.spinner("Running…"):
            try:
                results = run_evals()
            except Exception as e:
                st.error(f"Eval run failed: {e}")
                return
        df = pd.DataFrame(results)
        passes = int(df["passed"].sum())
        st.metric("Pass rate", f"{passes}/{len(df)} ({100 * passes / max(1, len(df)):.0f}%)")
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_killswitch() -> None:
    st.caption(
        "Take the bot offline without touching code. The chat surface will show "
        "an offline message until you flip this back."
    )
    currently_disabled = KILLSWITCH.exists()
    new = st.checkbox("Take bot offline", value=currently_disabled, key="bot_offline")
    if new and not currently_disabled:
        KILLSWITCH.touch()
        st.warning("Bot is now offline.")
    elif not new and currently_disabled:
        KILLSWITCH.unlink(missing_ok=True)
        st.success("Bot is back online.")
