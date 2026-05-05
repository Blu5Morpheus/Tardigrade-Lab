"""GitHub repo wrapper used by the admin content editor.

The PAT is fine-grained, scoped to the tardigrade-site repo with
contents:read+write only. See LAUNCH_CHECKLIST.md.
"""

from __future__ import annotations

import functools

from .secrets import get_secret, has_secret


@functools.lru_cache(maxsize=1)
def get_repo():
    from github import Auth, Github
    auth = Auth.Token(get_secret("github", "token"))
    gh = Github(auth=auth)
    return gh.get_repo(get_secret("github", "repo"))


def github_configured() -> bool:
    return has_secret("github", "token") and has_secret("github", "repo")


def committer_dict() -> dict:
    return {
        "name": get_secret("github", "committer_name", "Tardigrade Admin"),
        "email": get_secret("github", "committer_email", "raven@tardigrade.dev"),
    }


def default_branch() -> str:
    return get_secret("github", "default_branch", "main")
