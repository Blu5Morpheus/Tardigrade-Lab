"""Tab 1 — read shop-notify submissions from Formspree, export CSV."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from lib.secrets import get_secret, has_secret


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_submissions(api_key: str, form_id: str) -> list[dict]:
    url = f"https://formspree.io/api/0/forms/{form_id}/submissions"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    payload = r.json()
    return payload.get("submissions", [])


def render() -> None:
    st.subheader("Shop notify-me list")
    st.caption("Live submissions from the Formspree form on the cosmic-dust shop gate.")

    if not (has_secret("formspree", "api_key") and has_secret("formspree", "form_id")):
        st.info(
            "Formspree not yet configured. Add `FORMSPREE_API_KEY` and `FORMSPREE_FORM_ID` "
            "to Render's environment variables (or `secrets.toml` locally)."
        )
        return

    try:
        subs = _fetch_submissions(
            get_secret("formspree", "api_key"),
            get_secret("formspree", "form_id"),
        )
    except Exception as e:
        st.error(f"Could not reach Formspree: {e}")
        return

    rows = [{
        "email": s.get("data", {}).get("email"),
        "submitted": s.get("date"),
        "ip": s.get("ip"),
    } for s in subs]
    df = pd.DataFrame(rows)
    st.metric("Total signups", len(df))

    if df.empty:
        st.info("No submissions yet.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Export CSV",
        df.to_csv(index=False).encode(),
        file_name=f"shop-notify-{pd.Timestamp.utcnow():%Y-%m-%d}.csv",
        mime="text/csv",
    )

    if st.button("Refresh"):
        _fetch_submissions.clear()
        st.rerun()
