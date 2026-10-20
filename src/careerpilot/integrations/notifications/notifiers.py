"""Notification delivery for the Notify Agent.

The one place with a real side effect, so two guardrails the read-only agents do not
need: (1) the LLM never controls the destination — it is fixed by configuration or
resolved from a whitelist by code; (2) nothing sends until an explicit human confirm,
which is enforced by the agent, not here.

In the public demo, ``DemoOutbox`` captures confirmed messages instead of contacting
Slack or email, so nothing ever leaves the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Delivery:
    channel: str  # "slack" | "email"
    destination: str
    subject: str
    body: str


class Notifier(Protocol):
    channel: str

    async def send(self, *, destination: str, subject: str, body: str) -> dict: ...


class DemoOutbox:
    """Session-scoped capture used instead of a real Notifier in demo mode."""

    def __init__(self) -> None:
        self._items: list[Delivery] = []

    async def capture(self, *, channel: str, destination: str, subject: str, body: str) -> dict:
        self._items.append(Delivery(channel=channel, destination=destination, subject=subject, body=body))
        return {"captured": True, "channel": channel, "destination": destination}

    def items(self) -> list[Delivery]:
        return list(self._items)


class SlackNotifier:
    """Real adapter — posts to one fixed channel via Slack's web API over httpx."""

    channel = "slack"

    def __init__(self, bot_token: str, channel_id: str) -> None:
        self._token = bot_token
        self._channel_id = channel_id  # destination is fixed, never LLM-controlled

    async def send(self, *, destination: str, subject: str, body: str) -> dict:
        import httpx

        text = f"*{subject}*\n{body}" if subject else body
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {self._token}"},
                json={"channel": self._channel_id, "text": text},
            )
        return resp.json()


class EmailNotifier:
    """Real adapter — sends via Resend's HTTP API (PaaS hosts often drop SMTP)."""

    channel = "email"

    def __init__(self, api_key: str, from_address: str, recipient: str) -> None:
        self._api_key = api_key
        self._from = from_address
        self._recipient = recipient  # fixed recipient, never LLM-controlled

    async def send(self, *, destination: str, subject: str, body: str) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"from": self._from, "to": [self._recipient], "subject": subject, "text": body},
            )
        return resp.json()
