"""Tab 3 — toggle demo enabled/disabled, edit defaults, store in Supabase."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from lib.supabase_client import get_service_client, supabase_configured

DEMO_SLUGS = ["vqe", "clifford", "amplituhedron", "lattice", "page-curve", "me-bot"]


def render() -> None:
    st.subheader("Demo controls")
    st.caption("Backed by Supabase `demo_settings`. Toggling `enabled = false` makes that demo show an offline panel.")

    if not supabase_configured():
        st.info(
            "Supabase not yet configured. Add `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and "
            "`SUPABASE_SERVICE_ROLE_KEY` to Render's environment variables. SQL schema in "
            "`scripts/supabase_schema.sql`."
        )
        return

    try:
        client = get_service_client()
        rows = client.table("demo_settings").select("*").order("display_order").execute().data
    except Exception as e:
        st.error(f"Could not read demo_settings: {e}")
        return

    if not rows:
        st.warning("No rows in `demo_settings`. Run the seed insert in `scripts/supabase_schema.sql`.")
        return

    df = pd.DataFrame(rows).set_index("slug")
    st.markdown("**Toggle visibility**")
    edited = st.data_editor(
        df[["display_order", "enabled", "notes"]],
        use_container_width=True,
        column_config={
            "display_order": st.column_config.NumberColumn(min_value=0, max_value=99),
            "enabled": st.column_config.CheckboxColumn(),
            "notes": st.column_config.TextColumn(),
        },
        num_rows="fixed",
    )

    if st.button("Save", type="primary"):
        for slug, row in edited.iterrows():
            try:
                client.table("demo_settings").update({
                    "display_order": int(row["display_order"]),
                    "enabled": bool(row["enabled"]),
                    "notes": row["notes"] or None,
                }).eq("slug", slug).execute()
            except Exception as e:
                st.error(f"Could not save `{slug}`: {e}")
                return
        st.success("Saved.")

    st.markdown("---")
    st.markdown("**Default parameters** (per demo, JSON)")
    selected = st.selectbox("Demo", df.index.tolist())
    current = df.loc[selected, "default_params"] or {}
    text = st.text_area(
        "JSON",
        value=json.dumps(current, indent=2) if isinstance(current, dict) else json.dumps({}, indent=2),
        height=200,
    )
    if st.button("Save params"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return
        try:
            client.table("demo_settings").update({"default_params": payload}).eq("slug", selected).execute()
            st.success(f"Saved defaults for `{selected}`.")
        except Exception as e:
            st.error(f"Could not save: {e}")
