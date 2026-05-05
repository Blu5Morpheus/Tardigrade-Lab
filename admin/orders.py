"""Tab 4 — orders dashboard. Stub until the shop launches."""

from __future__ import annotations

import streamlit as st

from lib.secrets import has_secret


def render() -> None:
    st.subheader("Orders")
    if not has_secret("stripe", "secret_key"):
        st.info(
            "🚧 Orders dashboard activates when the shop goes live. "
            "Add `STRIPE_SECRET_KEY` to Render env vars and uncomment the Stripe code path here."
        )
        st.caption("Schema for the `orders` Supabase table is already created — see `scripts/supabase_schema.sql`.")
        return

    st.warning(
        "Stripe integration not yet wired in this build. The hooks are documented in "
        "LAB_AND_ADMIN_SPEC.md §4.5 — implement `stripe.PaymentIntent.list()`, mirror "
        "into Supabase `orders`, render here."
    )
