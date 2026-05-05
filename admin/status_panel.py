"""Tab 6 — edit the hero `// LAB STATUS` rows; trigger Astro rebuild."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from lib.secrets import get_secret, has_secret
from lib.supabase_client import get_service_client, supabase_configured


def render() -> None:
    st.subheader("Lab status panel")
    st.caption(
        "Backed by Supabase `lab_status`. The Astro site reads these at build time. "
        "Use **Trigger rebuild** after editing to push the change live."
    )

    if not supabase_configured():
        st.info("Supabase not yet configured — see `scripts/supabase_schema.sql`.")
        return

    client = get_service_client()
    try:
        rows = client.table("lab_status").select("*").order("display_order").execute().data
    except Exception as e:
        st.error(f"Could not read lab_status: {e}")
        return

    df = pd.DataFrame(rows or [])
    if df.empty:
        st.warning("No lab_status rows. Run the seed insert in `scripts/supabase_schema.sql`.")
        return

    edit_cols = ["display_order", "label", "value", "kind", "active"]
    edited = st.data_editor(
        df[["id", *edit_cols]].set_index("id"),
        use_container_width=True,
        column_config={
            "display_order": st.column_config.NumberColumn(min_value=0, max_value=99),
            "label": st.column_config.TextColumn(),
            "value": st.column_config.TextColumn(),
            "kind": st.column_config.SelectboxColumn(options=["ok", "warn", "default"]),
            "active": st.column_config.CheckboxColumn(),
        },
        num_rows="dynamic",
    )

    if st.button("Save rows", type="primary"):
        for row_id, row in edited.iterrows():
            try:
                payload = {
                    "display_order": int(row["display_order"]),
                    "label": row["label"],
                    "value": row["value"],
                    "kind": row["kind"],
                    "active": bool(row["active"]),
                }
                client.table("lab_status").update(payload).eq("id", row_id).execute()
            except Exception as e:
                st.error(f"Save failed for {row_id}: {e}")
                return
        st.success("Saved.")

    st.markdown("---")
    if has_secret("render", "deploy_hook_url"):
        if st.button("Trigger Astro rebuild"):
            try:
                r = requests.post(get_secret("render", "deploy_hook_url"), timeout=10)
                r.raise_for_status()
                st.success("Rebuild kicked off.")
            except Exception as e:
                st.error(f"Could not trigger deploy: {e}")
    else:
        st.caption(
            "Set `RENDER_DEPLOY_HOOK_URL` to enable the **Trigger rebuild** button. "
            "Until then, push a commit to the Astro repo or click Manual Deploy in Render."
        )
