"""LLM gateway seam.

Nothing in ``orchestrator/`` or ``agents/`` imports a vendor SDK. They depend on the
``LLMClient`` protocol defined here; ``OpenAILLMClient`` is the single real adapter.
This is what lets every agent be unit-tested with a ``FakeLLMClient`` — no API key,
no network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResponse:
    """One turn of a tool-calling loop: a final answer (``content`` set, no tool
    calls) or a request to run tools (``tool_calls`` set). The caller owns the
    running message history, keeping this adapter thin and stateless per call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    async def get_structured_output(
        self, *, system_prompt: str, user_prompt: str, schema: type[SchemaT]
    ) -> SchemaT:
        """Return an instance of ``schema``. The vendor validates the model output
        against the schema before returning, so malformed output never passes
        silently — the hard schema guardrail used by the router and the RAG agent."""
        ...

    async def get_tool_response(self, *, messages: list[dict], tools: list[dict]) -> ToolResponse:
        """Multi-turn function-calling for agents that need several lookups before
        answering. ``messages``/``tools`` use the OpenAI chat shape directly; there
        is one adapter, so there is no abstraction cost to that today."""
        ...


class OpenAILLMClient:
    """Concrete ``LLMClient`` backed by OpenAI Structured Outputs + function calling.

    OpenAI is the only adapter implemented here; ``LLMClient`` stays a Protocol so a
    different provider could be dropped in without touching any calling code.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4.1-mini") -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    async def get_structured_output(
        self, *, system_prompt: str, user_prompt: str, schema: type[SchemaT]
    ) -> SchemaT:
        completion = await self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=schema,
        )
        message = completion.choices[0].message
        if message.refusal:
            raise ValueError(f"Model refused structured output: {message.refusal}")
        if message.parsed is None:
            raise ValueError("Model response did not parse against the required schema")
        return message.parsed

    async def get_tool_response(self, *, messages: list[dict], tools: list[dict]) -> ToolResponse:
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
        message = completion.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in (message.tool_calls or [])
        ]
        return ToolResponse(content=message.content, tool_calls=tool_calls)


def default_llm_client() -> LLMClient:
    """The LLM client used for real (non-test) runs."""
    return OpenAILLMClient()
