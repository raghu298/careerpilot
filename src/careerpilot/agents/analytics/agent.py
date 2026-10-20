"""Analytics Agent — natural-language questions over the user's own financial data.

Same tool-calling loop as the Market Agent, but a single tool: ``run_sql_query``.
SQL's flexibility means one tool is enough. The read-only guardrail lives in the SQL
client (regex pre-filter + a read-only DB role in production); the live schema is
injected into the prompt so the model never guesses table or column names.
"""

from __future__ import annotations

from careerpilot.core.agent import AgentResult, Capabilities, OnEvent
from careerpilot.core.llm_client import LLMClient
from careerpilot.core.tool_cache import ToolCache
from careerpilot.core.tool_loop import run_tool_loop
from careerpilot.integrations.sql_db.postgres_client import ReadOnlyViolation, SQLClient

_SYSTEM_PROMPT = """You are the analytics sub-agent of a job-hunt / career assistant.
Answer questions about the user's own job-application tracker (which companies/roles they
applied to, current status, response and interview rates, and which applications have
gone stale) by writing a single read-only SQL SELECT and calling run_sql_query. Only
ever read data; never attempt to modify it. Base every number on query results. Here is
the live database schema:

{schema}
"""

_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "run_sql_query",
        "description": "Run one read-only SQL SELECT against the user's financial database.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["query"],
        },
    },
}


class AnalyticsAgent:
    name = "analytics"

    def __init__(self, llm_client: LLMClient, sql_client: SQLClient, capabilities: Capabilities) -> None:
        self._llm = llm_client
        self._sql = sql_client
        self._caps = capabilities

    async def handle(
        self,
        user_request: str,
        history: list[dict] | None = None,
        on_event: OnEvent | None = None,
        tool_cache: ToolCache | None = None,
    ) -> AgentResult:
        if self._caps.sql_role == "none":
            return AgentResult(self.name, "You don't have access to the analytics database.")

        schema = await self._sql.schema_description()
        messages = [{"role": "system", "content": _SYSTEM_PROMPT.format(schema=schema)}]
        messages += history or []
        messages.append({"role": "user", "content": user_request})

        async def execute(name: str, arguments: dict):
            if name != "run_sql_query":
                return {"error": f"unknown tool {name}"}
            try:
                rows = await self._sql.run_query(arguments.get("query", ""))
            except ReadOnlyViolation as exc:
                return {"error": f"blocked: {exc}"}
            return {"rows": rows, "row_count": len(rows)}

        result = await run_tool_loop(
            llm=self._llm,
            messages=messages,
            tools=[_TOOL_SPEC],
            execute=execute,
            agent_name=self.name,
            on_event=on_event,
            tool_cache=tool_cache,
        )
        return AgentResult(self.name, result.content, metadata={"citations": result.citations})
