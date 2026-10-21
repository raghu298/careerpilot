"""LLM Gateway — the Shared-Foundation seam in front of every model call.

Wraps any ``LLMClient`` (real or offline) and adds, transparently to the agents:
  - **model routing**: a per-call-type model choice (cheap model for routing, stronger
    for generation) recorded for observability — real adapters can read it;
  - **observability**: per-call counters + latency into the metrics registry;
  - a single choke point where provider rate limiting / retries would live.

It implements the same ``LLMClient`` protocol, so nothing downstream changes.
"""

from __future__ import annotations

from careerpilot.core.llm_client import LLMClient, ToolResponse
from careerpilot.core.observability import get_logger, metrics

logger = get_logger("llm_gateway")

# Model routing policy: which model each call type should use when a real provider is
# active. Offline the inner client ignores it; the choice is still recorded.
_MODEL_ROUTING = {
    "RoutingDecision": "gpt-4.1-mini",   # cheap/fast classification
    "KnowledgeAnswer": "gpt-4.1",        # grounded generation
    "Draft": "gpt-4.1-mini",
    "_tools": "gpt-4.1-mini",
    "_default": "gpt-4.1-mini",
}


class LLMGateway:
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    @staticmethod
    def route_model(call_type: str) -> str:
        return _MODEL_ROUTING.get(call_type, _MODEL_ROUTING["_default"])

    async def get_structured_output(self, *, system_prompt, user_prompt, schema):
        call_type = schema.__name__
        model = self.route_model(call_type)
        metrics.incr("llm.calls")
        metrics.incr(f"llm.structured.{call_type}")
        logger.info("llm_call type=structured schema=%s model=%s", call_type, model)
        with metrics.timer("llm.structured.latency_ms"):
            return await self._inner.get_structured_output(
                system_prompt=system_prompt, user_prompt=user_prompt, schema=schema
            )

    async def get_tool_response(self, *, messages, tools) -> ToolResponse:
        model = self.route_model("_tools")
        metrics.incr("llm.calls")
        metrics.incr("llm.tool_calls")
        logger.info("llm_call type=tools tools=%d model=%s", len(tools or []), model)
        with metrics.timer("llm.tools.latency_ms"):
            return await self._inner.get_tool_response(messages=messages, tools=tools)
