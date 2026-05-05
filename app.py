"""Tardigrade Lab — Streamlit demo router.

One app, two responsibilities:
  1. Serve the public interactive demos (iframed by the Astro site).
  2. Serve the password-gated admin panel at ?demo=admin.

Routing is driven entirely by the ?demo=<slug> query param. ?embed=true
strips Streamlit chrome so the iframe looks native to the Astro site.
"""

from __future__ import annotations

import importlib

import streamlit as st

from lib.theme import inject_css

DEMO_REGISTRY: dict[str, tuple[str, str]] = {
    "vqe":           ("VQE × LIGO signal classifier",     "demos.vqe_ligo"),
    "clifford":      ("Clifford geometric algebra agent", "demos.clifford_agent"),
    "amplituhedron": ("Amplituhedron explorer",           "demos.amplituhedron"),
    "lattice":       ("Lattice gauge sandbox",            "demos.lattice_gauge"),
    "page-curve":    ("Page curve simulator",             "demos.page_curve"),
    "me-bot":        ("Ask the lab",                      "demos.me_bot"),
}


def _embed_param(value: object) -> bool:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).lower() in {"true", "1", "yes"}


def _set_page_config(slug: str, embed: bool) -> None:
    title = DEMO_REGISTRY.get(slug, ("Tardigrade Lab", ""))[0]
    st.set_page_config(
        page_title=f"{title} — Tardigrade Lab" if title else "Tardigrade Lab",
        page_icon="🟢",
        layout="wide",
        initial_sidebar_state="collapsed" if embed else "auto",
        menu_items={
            "Get help": "https://portfolio-lab-v05x.onrender.com/contact",
            "About": "Tardigrade Innovation Lab — interactive demos",
        },
    )


def main() -> None:
    params = st.query_params
    raw_slug = params.get("demo", "vqe")
    slug = raw_slug if isinstance(raw_slug, str) else (raw_slug[0] if raw_slug else "vqe")
    embed = _embed_param(params.get("embed", "false"))

    _set_page_config(slug, embed)
    inject_css(embed=embed)

    if slug == "admin":
        # admin should never be iframed — force embed off.
        from admin import auth as admin_auth
        admin_auth.render_admin_or_login()
        return

    if slug not in DEMO_REGISTRY:
        st.error(f"Unknown demo: `{slug}`")
        st.caption(
            "Try one of: " + ", ".join(f"`{s}`" for s in DEMO_REGISTRY if s != "admin")
        )
        return

    title, module_path = DEMO_REGISTRY[slug]
    if not embed:
        st.markdown(f"# {title}")

    module = importlib.import_module(module_path)
    if not hasattr(module, "render"):
        st.error(f"Demo `{slug}` has no `render()` function.")
        return

    module.render(embed=embed)


if __name__ == "__main__":
    main()
