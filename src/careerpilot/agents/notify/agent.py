"""Notify Agent — the only agent with a real side effect, so it is human-in-the-loop.

``handle()`` never sends anything: it drafts a budget alert or payment reminder with
the LLM (schema-guarded) and stages it as a ``PendingAction``. The destination is
chosen by configuration/policy, never by the model. Execution happens only on an
explicit ``confirm_pending`` (or ``cancel_pending`` / ``revise_pending``), which the
LangGraph HITL interrupt drives. In demo mode the confirmed message is captured in the
``DemoOutbox`` instead of contacting Slack or email.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from careerpilot.core.agent import (
    AgentResult,
    Capabilities,
    OnEvent,
    PendingAction,
    emit,
)
from careerpilot.core.llm_client import LLMClient
from careerpilot.core.tool_cache import ToolCache
from careerpilot.integrations.notifications.notifiers import DemoOutbox

_SYSTEM_PROMPT = """You are the notify sub-agent of a job-hunt / career assistant.
Draft a short, professional outreach message based on the user's request — a tailored
cover letter or a follow-up email to a recruiter. Produce only the message content —
you do NOT choose the recipient and you do NOT send it. Keep it concise, specific, and
warm, and reference the role/company when the request names one.
"""


class Draft(BaseModel):
    channel: str = Field(description='"slack" or "email"')
    subject: str = Field(description="Short subject line (may be empty for slack).")
    body: str = Field(description="The notification body text.")


class NotifyAgent:
    name = "notify"

    def __init__(
        self,
        llm_client: LLMClient,
        capabilities: Capabilities,
        *,
        demo_outbox: DemoOutbox,
        default_destination: str = "recruiter@careerpilot.demo",
    ) -> None:
        self._llm = llm_client
        self._caps = capabilities
        self._outbox = demo_outbox
        self._destination = default_destination  # resolved by policy, never by the LLM
        self._pending: PendingAction | None = None

    async def handle(
        self,
        user_request: str,
        history: list[dict] | None = None,
        on_event: OnEvent | None = None,
        tool_cache: ToolCache | None = None,
    ) -> AgentResult:
        if not self._caps.can_stage_notifications:
            return AgentResult(self.name, "Staging notifications isn't available for this profile.")

        draft = await self._llm.get_structured_output(
            system_prompt=_SYSTEM_PROMPT, user_prompt=user_request, schema=Draft
        )
        channel = draft.channel if draft.channel in ("slack", "email") else "email"
        preview = f"[{channel}] to {self._destination}\nSubject: {draft.subject}\n\n{draft.body}"
        self._pending = PendingAction(
            kind="message",
            destination=self._destination,
            subject=draft.subject,
            body=draft.body,
            preview=preview,
        )
        emit(on_event, {"type": "staged", "agent": self.name})
        return AgentResult(
            self.name,
            f"I've drafted this notification. Review and confirm before it's sent:\n\n{preview}",
            metadata={"staged": True},
        )

    # ---- ConfirmableAgent ---------------------------------------------------

    def has_pending(self) -> bool:
        return self._pending is not None

    def pending(self) -> PendingAction | None:
        return self._pending

    async def confirm_pending(self) -> AgentResult:
        if self._pending is None:
            return AgentResult(self.name, "There's nothing staged to send.")
        p = self._pending
        await self._outbox.capture(channel="email", destination=p.destination, subject=p.subject, body=p.body)
        self._pending = None
        return AgentResult(self.name, "Sent — captured in your Demo Outbox.", metadata={"sent": True})

    def cancel_pending(self) -> AgentResult:
        self._pending = None
        return AgentResult(self.name, "Cancelled — nothing was sent.", metadata={"cancelled": True})

    async def revise_pending(self, edit_instructions: str) -> AgentResult:
        if self._pending is None:
            return AgentResult(self.name, "There's nothing staged to revise.")
        revised = await self._llm.get_structured_output(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Revise this draft. Instructions: {edit_instructions}\n\nCurrent body:\n{self._pending.body}",
            schema=Draft,
        )
        p = self._pending
        p.subject = revised.subject
        p.body = revised.body
        p.preview = f"[revised] to {p.destination}\nSubject: {p.subject}\n\n{p.body}"
        return AgentResult(self.name, f"Updated draft:\n\n{p.preview}", metadata={"staged": True})
