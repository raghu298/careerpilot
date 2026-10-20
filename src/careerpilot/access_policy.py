"""Persona → typed capability resolution.

Access is decided here, before any agent or data adapter is constructed — not left to
prompt text. Two canonical demo personas, no credentials required:

- Riya Sharma  — an active job seeker (software engineer): her own application tracker
  (basic SQL role), general career-advice docs + her resume, live job search, and can
  stage outreach drafts.
- Arjun Mehta  — a senior career switcher: full read-only analytics across the tracker,
  plus premium documents (salary-negotiation and leadership-resume playbooks).

Missing capabilities fail closed.
"""

from __future__ import annotations

from careerpilot.core.agent import Capabilities

_PERSONAS: dict[str, Capabilities] = {
    "riya": Capabilities(
        persona_id="riya",
        display_name="Riya Sharma",
        knowledge_labels=("general",),
        sql_role="basic",
        can_use_jobs=True,
        can_stage_notifications=True,
    ),
    "arjun": Capabilities(
        persona_id="arjun",
        display_name="Arjun Mehta",
        knowledge_labels=("general", "premium"),
        sql_role="full",
        can_use_jobs=True,
        can_stage_notifications=True,
    ),
}

DEFAULT_PERSONA = "riya"


def resolve_capabilities(persona_id: str) -> Capabilities:
    if persona_id not in _PERSONAS:
        raise KeyError(f"Unknown persona: {persona_id!r}")
    return _PERSONAS[persona_id]


def list_personas() -> list[Capabilities]:
    return list(_PERSONAS.values())
