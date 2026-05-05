"""Cache-clearing helpers — used after admin writes that should invalidate
demo / status reads."""

from __future__ import annotations

import streamlit as st


def clear_all_data_caches() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        pass


def clear_resource_caches() -> None:
    try:
        st.cache_resource.clear()
    except Exception:
        pass
