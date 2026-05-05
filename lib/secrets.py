"""Unified secret access — env vars in production, secrets.toml in dev.

Streamlit's `st.secrets` reads from `.streamlit/secrets.toml`. In production
on Render we want to use environment variables instead, because Render's
secret-file workflow is more friction than its env-var workflow.

This wrapper checks env first (uppercase `SECTION_KEY`), falls back to
`st.secrets[section][key]`, and raises a clean RuntimeError if both are
missing — easier to surface in the admin diagnostics tab than a KeyError.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st


def get_secret(section: str, key: str, default: Optional[str] = None) -> str:
    env_key = f"{section.upper()}_{key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]
    try:
        return st.secrets[section][key]
    except (KeyError, FileNotFoundError):
        if default is not None:
            return default
        raise RuntimeError(
            f"Missing secret: {section}.{key}. "
            f"Set env var {env_key} or add it to .streamlit/secrets.toml."
        )


def has_secret(section: str, key: str) -> bool:
    env_key = f"{section.upper()}_{key.upper()}"
    if env_key in os.environ:
        return True
    try:
        _ = st.secrets[section][key]
        return True
    except (KeyError, FileNotFoundError):
        return False
