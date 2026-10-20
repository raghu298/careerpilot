"""The only shapes a routing decision is allowed to take.

Forcing the LLM's routing choice through ``RoutingDecision`` is the hard schema
guardrail: the model may reason about intent however it likes, but its answer must
resolve to one or more of these four fixed agent names before anything acts on it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AgentName = Literal["knowledge", "jobs", "analytics", "notify"]


class RoutingDecision(BaseModel):
    agents: list[AgentName] = Field(
        min_length=1,
        description=(
            "Every agent needed to fully answer the request. More than one means the "
            "request should fan out to several agents running in parallel."
        ),
    )
    reasoning: str = Field(
        description="Short justification, kept for tracing/eval. Not shown to the user."
    )
