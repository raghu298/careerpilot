"""Jobs Agent — finds live openings and company info through a jobs MCP server.

A genuine multi-turn tool-calling loop: the LLM decides which MCP tools to call
(``search_jobs``, ``get_company``), the agent executes them over MCP, and the model
composes the final answer. Citations are collected programmatically from the tool
calls actually made, not self-reported by the model.
"""

from __future__ import annotations

from careerpilot.core.agent import AgentResult, Capabilities, OnEvent
from careerpilot.core.llm_client import LLMClient
from careerpilot.core.tool_cache import ToolCache
from careerpilot.core.tool_loop import run_tool_loop
from careerpilot.integrations.jobs_data.mcp_client import JobsSource

_SYSTEM_PROMPT = """You are the jobs sub-agent of a career assistant.
Use the job tools to find live openings and look up company information before
answering. List roles only from tool results — never invent a posting. If nothing
matches, say so plainly. You do not have access to the user's resume or their
application history.
"""


class JobsAgent:
    name = "jobs"

    def __init__(self, llm_client: LLMClient, source: JobsSource, capabilities: Capabilities) -> None:
        self._llm = llm_client
        self._source = source
        self._caps = capabilities

    async def handle(
        self,
        user_request: str,
        history: list[dict] | None = None,
        on_event: OnEvent | None = None,
        tool_cache: ToolCache | None = None,
    ) -> AgentResult:
        if not self._caps.can_use_jobs:
            return AgentResult(self.name, "Job search isn't available for this profile.")

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages += history or []
        messages.append({"role": "user", "content": user_request})

        result = await run_tool_loop(
            llm=self._llm,
            messages=messages,
            tools=self._source.tool_specs(),
            execute=self._source.call_tool,
            agent_name=self.name,
            on_event=on_event,
            tool_cache=tool_cache,
        )
        return AgentResult(self.name, result.content, metadata={"citations": result.citations})
