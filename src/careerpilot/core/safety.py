"""Guardrails & Safety — PII redaction and a light content filter.

Part of the Shared Foundation. ``redact_pii`` keeps personal data out of logs and
metrics; ``scan_pii`` flags what a staged outreach draft contains so the human reviewer
sees it before confirming; ``content_check`` is a minimal safety gate for clearly
out-of-bounds requests (the app is a career copilot, not a general assistant).
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
# Phone: 10-15 digits possibly with separators / country code.
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)")
# Credit-card-like: 13-16 digit runs (with optional spaces/dashes).
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")

_PII_PATTERNS = [("card", _CARD), ("email", _EMAIL), ("phone", _PHONE)]


def redact_pii(text: str) -> str:
    """Replace emails / phones / card-like numbers with typed placeholders."""
    if not text:
        return text
    out = text
    for label, pattern in _PII_PATTERNS:
        out = pattern.sub(f"[{label} redacted]", out)
    return out


def scan_pii(text: str) -> list[str]:
    """Return the distinct kinds of PII present (for flagging a draft to the reviewer)."""
    found: list[str] = []
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text or ""):
            found.append(label)
    return found


# Clearly out-of-scope / unsafe intents for a career assistant. Kept intentionally
# narrow so normal job-hunt questions are never blocked.
_BLOCK_PATTERNS = [
    re.compile(r"\b(kill myself|suicide|self[- ]?harm|end my life)\b", re.I),
    re.compile(r"\b(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)\b", re.I),
    re.compile(r"\b(fake|forge|falsif\w*)\s+(my\s+)?(degree|diploma|experience|references?)\b", re.I),
]

_SAFE_REDIRECTS = {
    0: ("It sounds like you may be going through something really hard. You're not alone — "
        "please consider reaching out to a crisis line in your country (e.g. in India, "
        "iCall at 9152987821). I'm here for the job-search side whenever you're ready."),
    1: "I can't help with that. I'm a job-hunt assistant — happy to help with your search instead.",
    2: ("I can't help fabricate credentials or experience — it tends to backfire in background "
        "checks. I can help you genuinely strengthen your resume for the roles you want, though."),
}


def content_check(text: str) -> str | None:
    """Return a safe redirect message if the request is out of bounds, else None."""
    for i, pattern in enumerate(_BLOCK_PATTERNS):
        if pattern.search(text or ""):
            return _SAFE_REDIRECTS[i]
    return None
