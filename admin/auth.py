"""Admin auth wrapper. Single bcrypt-hashed account, signed-cookie session."""

from __future__ import annotations

import streamlit as st

from lib.secrets import get_secret


def _build_authenticator():
    import streamlit_authenticator as stauth
    config = {
        "credentials": {
            "usernames": {
                get_secret("admin", "username", "raven"): {
                    "name": "Raven",
                    "password": get_secret("admin", "password_hash"),
                    "email": "raven@tardigrade.dev",
                    "failed_login_attempts": 0,
                    "logged_in": False,
                }
            }
        },
        "cookie": {
            "name": "tardigrade_admin_session",
            "key": get_secret("admin", "session_secret"),
            "expiry_days": 7,
        },
    }
    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )


def render_admin_or_login() -> None:
    st.markdown("# Tardigrade Admin")

    try:
        authenticator = _build_authenticator()
    except RuntimeError as e:
        st.error(f"Admin not configured: {e}")
        st.caption(
            "Set `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH`, and `ADMIN_SESSION_SECRET` "
            "in Render's environment variables (or `.streamlit/secrets.toml` locally). "
            "See `LAUNCH_CHECKLIST.md`."
        )
        return

    try:
        authenticator.login(location="main")
    except Exception:
        st.error("Login subsystem unavailable.")
        return

    auth_status = st.session_state.get("authentication_status")
    if auth_status is False:
        st.error("Invalid credentials.")
        return
    if auth_status is None:
        st.info("Enter admin credentials above.")
        return

    authenticator.logout("Logout", "sidebar")
    _render_dashboard()


def _render_dashboard() -> None:
    from admin import (
        analytics,
        content_editor,
        demo_controls,
        diagnostics,
        me_bot_admin,
        notify_list,
        orders,
        status_panel,
    )

    tabs = st.tabs([
        "📋 Notify list",
        "📝 Content",
        "🔬 Demos",
        "📦 Orders",
        "📊 Analytics",
        "⚡ Lab status",
        "🤖 Me-bot",
        "🧪 Diagnostics",
    ])

    with tabs[0]: notify_list.render()
    with tabs[1]: content_editor.render()
    with tabs[2]: demo_controls.render()
    with tabs[3]: orders.render()
    with tabs[4]: analytics.render()
    with tabs[5]: status_panel.render()
    with tabs[6]: me_bot_admin.render()
    with tabs[7]: diagnostics.render()
