"""Eval harness — runs evals.yaml cases and reports pass/fail."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .generator import NO_CONTEXT_RESPONSE, generate_full
from .retriever import retrieve

EVALS_PATH = Path(__file__).parent / "evals.yaml"

REFUSAL_MARKERS = (
    "i don't have public information",
    "i'd rather not",
    "outside what i can",
    "outside my scope",
    "contact form",
    "use the contact form",
    "i only cover",
    NO_CONTEXT_RESPONSE.split(".")[0].lower(),
)


def _is_refusal(text: str) -> bool:
    t = text.lower()
    return any(marker in t for marker in REFUSAL_MARKERS)


def _check_case(case: dict, response: str) -> tuple[bool, str]:
    text = response.lower()
    if case.get("expect_refusal"):
        return _is_refusal(response), "refusal expected"
    if "must_not_include" in case:
        for s in case["must_not_include"]:
            if s.lower() in text:
                return False, f"forbidden substring present: {s!r}"
    if "must_include" in case:
        for s in case["must_include"]:
            if s.lower() not in text:
                return False, f"missing required substring: {s!r}"
    if "must_include_one_of" in case:
        if not any(s.lower() in text for s in case["must_include_one_of"]):
            return False, "none of must_include_one_of present"
    return True, "ok"


def run_evals() -> list[dict]:
    cases = yaml.safe_load(EVALS_PATH.read_text(encoding="utf-8")) or []
    results = []
    for case in cases:
        query = case["query"]
        chunks = retrieve(query, k=5, min_score=0.25)
        response = NO_CONTEXT_RESPONSE if not chunks else generate_full(query, chunks)
        passed, reason = _check_case(case, response)
        results.append({
            "query": query,
            "passed": passed,
            "reason": reason,
            "response": response,
            "n_chunks": len(chunks),
        })
    return results


if __name__ == "__main__":
    import json
    r = run_evals()
    passes = sum(1 for x in r if x["passed"])
    print(json.dumps(r, indent=2))
    print(f"\n{passes}/{len(r)} passed")
