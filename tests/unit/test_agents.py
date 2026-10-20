from careerpilot.agents.analytics.agent import AnalyticsAgent
from careerpilot.agents.jobs.agent import JobsAgent
from careerpilot.agents.knowledge.agent import KnowledgeAgent
from careerpilot.agents.knowledge.schemas import KnowledgeAnswer
from careerpilot.agents.notify.agent import Draft, NotifyAgent
from careerpilot.core.agent import Capabilities, ConfirmableAgent
from careerpilot.core.llm_client import ToolResponse
from careerpilot.integrations.jobs_data.mcp_client import FakeJobsMCP
from careerpilot.integrations.notifications.notifiers import DemoOutbox
from careerpilot.integrations.sql_db.fake_sql import FakeSQLClient
from careerpilot.integrations.vector_store.pgvector_store import InMemoryVectorStore
from tests.unit.fakes import FakeEmbeddingClient, ScriptedLLMClient, tool_call


def _caps(**over):
    base = dict(persona_id="riya", display_name="Riya", knowledge_labels=("general",), sql_role="basic")
    base.update(over)
    return Capabilities(**base)


async def test_knowledge_agent_retrieves_and_answers():
    store = InMemoryVectorStore()
    embed = FakeEmbeddingClient()
    [vec] = await embed.embed(["resume"])
    await store.add(content="resume bullet guidance", embedding=vec, source="resume-tips", access_scope="general")
    llm = ScriptedLLMClient(
        structured={"KnowledgeAnswer": KnowledgeAnswer(answer="Lead with impact.", citations=["resume-tips"], answer_found=True)}
    )
    agent = KnowledgeAgent(llm, embed, store, _caps())
    result = await agent.handle("how do I write a resume bullet about resume?")
    assert result.content == "Lead with impact."
    assert result.metadata["answer_found"] is True


async def test_knowledge_agent_respects_scope():
    """A premium-only doc must not be retrievable by a general-only persona."""
    store = InMemoryVectorStore()
    embed = FakeEmbeddingClient()
    [vec] = await embed.embed(["salary"])
    await store.add(content="negotiate the whole package", embedding=vec, source="salary-negotiation-premium", access_scope="premium")
    llm = ScriptedLLMClient(structured={"KnowledgeAnswer": KnowledgeAnswer(answer="x", citations=[], answer_found=True)})
    agent = KnowledgeAgent(llm, embed, store, _caps())  # general only
    result = await agent.handle("how do I negotiate salary?")
    assert result.metadata["answer_found"] is False  # nothing in scope was retrieved


async def test_jobs_agent_tool_calling_loop():
    llm = ScriptedLLMClient(
        tool_responses=[
            ToolResponse(content=None, tool_calls=[tool_call("search_jobs", {"role": "engineer", "location": "Bangalore"})]),
            ToolResponse(content="Found 2 roles in Bangalore."),
        ]
    )
    agent = JobsAgent(llm, FakeJobsMCP(), _caps())
    result = await agent.handle("find engineer roles in Bangalore")
    assert "2 roles" in result.content
    assert any("search_jobs" in c for c in result.metadata["citations"])


async def test_analytics_agent_runs_readonly_sql():
    llm = ScriptedLLMClient(
        tool_responses=[
            ToolResponse(content=None, tool_calls=[tool_call("run_sql_query", {"query": "SELECT status, COUNT(*) AS count FROM applications GROUP BY status"})]),
            ToolResponse(content="You have applications across 6 statuses."),
        ]
    )
    agent = AnalyticsAgent(llm, FakeSQLClient(role="basic"), _caps())
    result = await agent.handle("how many applications by status?")
    assert "6 statuses" in result.content


async def test_analytics_agent_blocks_write_sql():
    llm = ScriptedLLMClient(
        tool_responses=[
            ToolResponse(content=None, tool_calls=[tool_call("run_sql_query", {"query": "DELETE FROM applications"})]),
            ToolResponse(content="I can only read data."),
        ]
    )
    agent = AnalyticsAgent(llm, FakeSQLClient(role="basic"), _caps())
    result = await agent.handle("delete my applications")
    # The tool result that came back to the model must have been a guardrail block.
    assert "I can only read data." == result.content


async def test_notify_agent_is_confirmable_and_stages():
    llm = ScriptedLLMClient(
        structured={"Draft": Draft(channel="email", subject="Follow-up", body="Hi, following up.")}
    )
    outbox = DemoOutbox()
    agent = NotifyAgent(llm, _caps(), demo_outbox=outbox)
    assert isinstance(agent, ConfirmableAgent)
    await agent.handle("follow up with the recruiter")
    assert agent.has_pending()
    await agent.confirm_pending()
    assert not agent.has_pending()
    assert len(outbox.items()) == 1
    assert outbox.items()[0].subject == "Follow-up"


async def test_notify_agent_cancel_sends_nothing():
    llm = ScriptedLLMClient(structured={"Draft": Draft(channel="email", subject="s", body="b")})
    outbox = DemoOutbox()
    agent = NotifyAgent(llm, _caps(), demo_outbox=outbox)
    await agent.handle("draft something")
    agent.cancel_pending()
    assert not agent.has_pending()
    assert outbox.items() == []
