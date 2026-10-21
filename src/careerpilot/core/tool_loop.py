"""A small multi-turn tool-calling loop shared by the Market and Analytics agents.

The agent owns its message history; this driver asks the LLM for the next step, runs
any requested tools (through a caller-supplied async ``execute``), appends the results,
and repeats until the model returns a final text answer or the step budget is hit.
Tool results are memoized in the session ``ToolCache`` and executed concurrently.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from careerpilot.core.agent import OnEvent, emit
from careerpilot.core.llm_client import LLMClient, ToolCall
from careerpilot.core.tool_cache import ToolCache

ExecuteTool = Callable[[str, dict], Awaitable[object]]


@dataclass
class ToolLoopResult:
    content: str
    citations: list[str] = field(default_factory=list)
    # Raw (name, parsed-result) for each tool call, so an agent can surface structured
    # data (e.g. the job list) to the UI instead of only the composed text answer.
    tool_results: list[tuple[str, object]] = field(default_factory=list)


async def run_tool_loop(
    *,
    llm: LLMClient,
    messages: list[dict],
    tools: list[dict],
    execute: ExecuteTool,
    agent_name: str,
    on_event: OnEvent | None = None,
    tool_cache: ToolCache | None = None,
    max_steps: int = 5,
) -> ToolLoopResult:
    citations: list[str] = []
    tool_results: list[tuple[str, object]] = []

    for _ in range(max_steps):
        response = await llm.get_tool_response(messages=messages, tools=tools)
        if not response.tool_calls:
            return ToolLoopResult(content=response.content or "", citations=citations, tool_results=tool_results)

        # Record the assistant's tool-call turn before appending results.
        messages.append(_assistant_tool_calls_message(response.tool_calls))

        async def one(call: ToolCall) -> tuple[ToolCall, object]:
            emit(on_event, {"type": "tool_called", "agent": agent_name, "tool": call.name, "args": call.arguments})
            cached = tool_cache.get(call.name, call.arguments) if tool_cache else None
            result = cached if cached is not None else await execute(call.name, call.arguments)
            if tool_cache is not None and cached is None:
                tool_cache.set(call.name, call.arguments, result)
            emit(on_event, {"type": "tool_result", "agent": agent_name, "tool": call.name})
            return call, result

        for call, result in await asyncio.gather(*(one(c) for c in response.tool_calls)):
            citations.append(f"{call.name}({json.dumps(call.arguments, sort_keys=True)})")
            tool_results.append((call.name, result))
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)}
            )

    # Out of steps — ask once more for a best-effort final answer, no tools.
    final = await llm.get_tool_response(messages=messages, tools=[])
    return ToolLoopResult(content=final.content or "", citations=citations, tool_results=tool_results)


def _assistant_tool_calls_message(tool_calls: list[ToolCall]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in tool_calls
        ],
    }
