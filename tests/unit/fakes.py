"""Hand-written fakes for the protocol seams, so agents/orchestrator are tested with
no network and no API key. These are the payoff of the ports-and-adapters design.
"""

from __future__ import annotations

from careerpilot.core.llm_client import ToolCall, ToolResponse


class ScriptedLLMClient:
    """Returns pre-programmed structured outputs and tool responses in order.

    - ``structured`` maps a schema class name to a value (or a list consumed in order).
    - ``tool_responses`` is a list of ``ToolResponse`` returned in order per call.
    """

    def __init__(self, *, structured: dict | None = None, tool_responses: list | None = None) -> None:
        self._structured = structured or {}
        self._tool_responses = list(tool_responses or [])
        self.structured_calls: list[tuple[str, str]] = []
        self.tool_calls_made = 0

    async def get_structured_output(self, *, system_prompt, user_prompt, schema):
        self.structured_calls.append((schema.__name__, user_prompt))
        value = self._structured[schema.__name__]
        if isinstance(value, list):
            return value.pop(0)
        return value

    async def get_tool_response(self, *, messages, tools) -> ToolResponse:
        self.tool_calls_made += 1
        if self._tool_responses:
            return self._tool_responses.pop(0)
        return ToolResponse(content="done")


class FakeEmbeddingClient:
    """Deterministic 3-d embeddings keyed on a few career terms, for retrieval tests."""

    _TERMS = ["resume", "salary", "interview"]

    async def embed(self, texts):
        return [self._vec(t) for t in texts]

    def _vec(self, text: str):
        t = text.lower()
        return [1.0 if term in t else 0.0 for term in self._TERMS]


def tool_call(name: str, args: dict, _id: str = "c1") -> ToolCall:
    return ToolCall(id=_id, name=name, arguments=args)
