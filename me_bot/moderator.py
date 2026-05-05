"""Output moderator — backstop pattern check.

The primary defense is the system prompt + low temperature + tight corpus.
This is a paranoia layer for fabrication patterns the model occasionally
produces despite the system prompt.
"""

from __future__ import annotations

import re

# Hedging that should not appear in factual responses about Raven.
HEDGE_PATTERNS = [
    r"\bI think\b",
    r"\bI believe\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bmight be\b",
    r"\bI assume\b",
    r"\bperhaps\b",
]

# Impersonation patterns.
IMPERSONATION_PATTERNS = [
    r"\bI am Raven\b",
    r"\bAs Raven,\b",
    r"\bAs the founder of Tardigrade\b",
]

# Credential fabrication: "graduated from X" only allowed when X is in
# the known-true set; otherwise it's a hallucinated credential.
KNOWN_INSTITUTIONS = ("LSU", "Roane State", "Harvard Extension", "St. Petersburg College", "Louisiana State")
CREDENTIAL_PATTERNS = [
    (r"\bgraduated from\b", KNOWN_INSTITUTIONS),
    (r"\benrolled at\b", KNOWN_INSTITUTIONS),
]

AWARDS_PATTERNS = [
    r"\bwon (?:the|a)\b",
    r"\bnominated for\b",
    r"\bawarded the\b(?! .*Academic Scholars)",
]


def check_output(text: str) -> tuple[bool, str | None]:
    """Return (passes, reason_if_blocked).

    A `False` return means the moderator caught a fabrication pattern; the
    caller should swap the response for a refusal and log the incident.
    """
    for pat in HEDGE_PATTERNS + IMPERSONATION_PATTERNS + AWARDS_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False, f"matched pattern: {pat}"
    for pat, allowed in CREDENTIAL_PATTERNS:
        # If the pattern matches AND none of the allowed institutions appear
        # in the same response, treat it as fabricated credential.
        if re.search(pat, text, re.IGNORECASE) and not any(inst in text for inst in allowed):
            return False, f"unsanctioned credential claim near pattern: {pat}"
    return True, None
