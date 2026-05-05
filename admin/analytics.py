"""Tab 5 — Plausible analytics top-line + embedded dashboard."""

from __future__ import annotations

import requests
import streamlit as st
from streamlit.components.v1 import iframe as components_iframe

from lib.secrets import get_secret, has_secret


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_aggregate(site_id: str, api_key: str) -> dict:
    url = (
        f"https://plausible.io/api/v1/stats/aggregate"
        f"?site_id={site_id}&period=7d&metrics=visitors,pageviews,bounce_rate,visit_duration"
    )
    r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
    r.raise_for_status()
    return r.json().get("results", {})


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_top_pages(site_id: str, api_key: str) -> list[dict]:
    url = f"https://plausible.io/api/v1/stats/breakdown?site_id={site_id}&property=event:page&period=7d&limit=5"
    r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
    r.raise_for_status()
    return r.json().get("results", [])


def render() -> None:
    st.subheader("Site analytics")
    if not (has_secret("plausible", "site_id") and has_secret("plausible", "api_key")):
        st.info(
            "Plausible is optional. To enable: sign up at plausible.io, generate a read-only API key "
            "and shared-link, then set `PLAUSIBLE_SITE_ID` and `PLAUSIBLE_API_KEY` in Render env vars."
        )
        return

    site_id = get_secret("plausible", "site_id")
    api_key = get_secret("plausible", "api_key")

    try:
        m = _fetch_aggregate(site_id, api_key)
        c1, c2, c3 = st.columns(3)
        c1.metric("Visitors (7d)", m.get("visitors", {}).get("value", "—"))
        c2.metric("Pageviews (7d)", m.get("pageviews", {}).get("value", "—"))
        c3.metric("Bounce rate", f"{m.get('bounce_rate', {}).get('value', 0)}%")
    except Exception as e:
        st.warning(f"Top-line metrics unavailable: {e}")

    try:
        pages = _fetch_top_pages(site_id, api_key)
        if pages:
            st.markdown("**Top pages**")
            st.dataframe(pages, hide_index=True, use_container_width=True)
    except Exception:
        pass

    st.markdown("---")
    if has_secret("plausible", "share_auth"):
        share_auth = get_secret("plausible", "share_auth")
        url = f"https://plausible.io/share/{site_id}?auth={share_auth}&embed=true&theme=dark"
        components_iframe(url, height=900, scrolling=True)
    else:
        st.caption("Set `PLAUSIBLE_SHARE_AUTH` to embed the full dashboard inline.")
