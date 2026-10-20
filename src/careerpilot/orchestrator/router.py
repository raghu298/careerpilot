"""Hybrid router: an LLM reasons about the request, but its decision must validate
against ``RoutingDecision`` before this class returns it. Invalid output is retried,
never passed downstream. Each agent's description is kept mutually exclusive so the
router does not misroute on keyword overlap between two agents.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from careerpilot.core.llm_client import LLMClient
from careerpilot.orchestrator.schemas import RoutingDecision

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the routing layer of a job-hunt / career assistant.

Decide which of the following agents are needed to answer the user's request.
Choose only the agents actually required — a request may need one agent or several.

- knowledge: answers from the user's own career documents and career-advice guides —
  their resume, a specific job description they're considering, and general guidance on
  resumes, interviews, and salary negotiation. Use it for "does my resume fit this
  role", "what are my gaps for this JD", or "how should I answer a behavioural
  question". It does NOT browse live job listings and does NOT see the application
  tracker.
- jobs: searches LIVE external job postings and company information through job-board
  tools (find openings for a role/location, look up a company). Use it for "find me
  backend roles in Bangalore" or "tell me about company X". It does NOT know the user's
  resume or which jobs they've already applied to.
- analytics: answers questions about the user's OWN application tracker via read-only
  SQL — which jobs they applied to, current status, response/interview rates, and which
  applications have gone stale. Anything about "my applications", "how many responses",
  "what's pending" belongs here, never jobs.
- notify: drafts an outreach message for the user (a tailored cover letter or a
  follow-up email to a recruiter) and stages it for confirmation. Nothing is ever sent
  without the user's explicit approval.
"""


class RoutingError(Exception):
    """Raised when no schema-valid routing decision could be obtained after retries."""


class Router:
    def __init__(self, llm_client: LLMClient, *, max_retries: int = 2) -> None:
        self._llm_client = llm_client
        self._max_retries = max_retries

    async def route(self, user_request: str, history_text: str = "") -> RoutingDecision:
        user_prompt = user_request
        if history_text:
            user_prompt = f"Conversation so far:\n{history_text}\n\nCurrent request: {user_request}"

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._llm_client.get_structured_output(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema=RoutingDecision,
                )
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Routing attempt %d/%d produced an invalid decision: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
        raise RoutingError(
            f"Could not obtain a valid routing decision after {self._max_retries + 1} attempts"
        ) from last_error
