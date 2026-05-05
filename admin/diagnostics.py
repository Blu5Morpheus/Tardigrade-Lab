"""Tab 8 — process memory diagnostics. The 512 MB Render free-tier
ceiling is real; this tab is how we know we're not flirting with OOM."""

from __future__ import annotations

import sys

import psutil
import streamlit as st


def render() -> None:
    st.subheader("Memory diagnostics")
    proc = psutil.Process()
    mi = proc.memory_info()
    vm = psutil.virtual_memory()

    c1, c2, c3 = st.columns(3)
    c1.metric("Process RSS", f"{mi.rss / 1024 / 1024:.1f} MB")
    c2.metric("Process VMS", f"{mi.vms / 1024 / 1024:.1f} MB")
    c3.metric("Available", f"{vm.available / 1024 / 1024:.1f} MB")

    st.markdown("**Headline budget:** 512 MB ceiling on Render free tier; target ≤ 400 MB working set.")
    st.markdown(f"**Python:** `{sys.version.splitlines()[0]}`")
    st.markdown(f"**CPU pct:** {proc.cpu_percent(interval=0.5):.1f}%")

    st.markdown("---")
    st.markdown("**Loaded ML modules**")
    loaded = [m for m in sys.modules if m.startswith(("torch", "sentence_transformers", "pennylane", "scipy", "numpy"))]
    st.code("\n".join(sorted(set(loaded))[:40]) or "(none yet)")
