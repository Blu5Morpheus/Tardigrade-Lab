"""Singleton Supabase clients — anon key for read paths, service role for writes."""

from __future__ import annotations

import functools
from typing import Optional

from .secrets import get_secret, has_secret


@functools.lru_cache(maxsize=1)
def get_anon_client():
    from supabase import create_client
    return create_client(
        get_secret("supabase", "url"),
        get_secret("supabase", "anon_key"),
    )


@functools.lru_cache(maxsize=1)
def get_service_client():
    from supabase import create_client
    return create_client(
        get_secret("supabase", "url"),
        get_secret("supabase", "service_role_key"),
    )


def supabase_configured() -> bool:
    return has_secret("supabase", "url") and has_secret("supabase", "anon_key")
