from careerpilot.agents.analytics.agent import AnalyticsAgent
from careerpilot.agents.jobs.agent import JobsAgent
from careerpilot.agents.interview.agent import InterviewPrepAgent
from careerpilot.agents.interview.schemas import InterviewPlan, InterviewQuestion
from careerpilot.agents.knowledge.agent import KnowledgeAgent
from careerpilot.agents.knowledge.schemas import KnowledgeAnswer
from careerpilot.agents.notify.agent import Draft, NotifyAgent
from careerpilot.core.agent import Capabilities, ConfirmableAgent
from careerpilot.core.llm_client import ToolResponse
from careerpilot.integrations.jobs_data.mcp_client import FakeJobsMCP
from careerpilot.integrations.notifications.notifiers import DemoOutbox
from careerpilot.integrations.sql_db.sqlite_client import SqliteSQLClient
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


async def test_interview_agent_builds_grounded_plan():
    store = InMemoryVectorStore()
    embed = FakeEmbeddingClient()
    [vec] = await embed.embed(["interview"])
    await store.add(content="interview prep: lead with impact", embedding=vec,
                    source="interview-guide", access_scope="general")
    plan = InterviewPlan(
        focus="Lead with your LLM projects.",
        questions=[
            InterviewQuestion(category="technical", question="Explain RAG.", talking_point="Cite your project."),
            InterviewQuestion(category="behavioral", question="A conflict?", talking_point="Use STAR."),
        ],
        grounded=True,
    )
    llm = ScriptedLLMClient(structured={"InterviewPlan": plan})
    agent = InterviewPrepAgent(llm, embed, store, _caps())
    result = await agent.handle("prep me for the interview")
    assert "Explain RAG." in result.content
    assert "**Technical**" in result.content and "**Behavioral**" in result.content
    assert result.metadata["grounded"] is True
    assert len(result.metadata["questions"]) == 2


async def test_offline_interview_plan_is_grounded_in_resume_and_jd():
    from careerpilot.integrations.offline.offline_llm import OfflineLLMClient

    ctx = (
        "[my-resume] Python, FastAPI and PostgreSQL engineer. Gaps: kubernetes.\n\n"
        "[jd-target] Senior GenAI Engineer. Required: Python, kubernetes, LLM."
    )
    plan = await OfflineLLMClient().get_structured_output(
        system_prompt="", user_prompt=f"Context:\n{ctx}\n\nRequest: prep me", schema=InterviewPlan
    )
    assert plan.grounded is True
    assert plan.questions
    cats = {q.category for q in plan.questions}
    assert {"technical", "behavioral", "role-specific"} <= cats
    # A JD skill the resume lacks must surface an honest ramp-up talking point.
    assert any("kubernetes" in q.question.lower() or "kubernetes" in q.talking_point.lower()
               for q in plan.questions)
    assert "Senior Genai Engineer".lower() in " ".join(q.question for q in plan.questions).lower()


async def test_interview_agent_requires_knowledge_access():
    agent = InterviewPrepAgent(
        ScriptedLLMClient(structured={}), FakeEmbeddingClient(), InMemoryVectorStore(),
        _caps(knowledge_labels=()),
    )
    result = await agent.handle("prep me")
    assert "access" in result.content.lower()


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


async def test_fake_mcp_salary_insights_aggregates():
    out = await FakeJobsMCP().call_tool("salary_insights", {"role": "genai engineer"})
    assert out["sample_size"] >= 1
    assert "LPA" in out["typical_range"]
    assert out["low"] and out["high"]


async def test_jobs_agent_reports_salary_via_offline_brain():
    """A pay question drives the salary_insights tool through the real offline brain."""
    from careerpilot.integrations.offline.offline_llm import OfflineLLMClient

    agent = JobsAgent(OfflineLLMClient(), FakeJobsMCP(), _caps())
    result = await agent.handle("what do senior genai engineer roles pay?")
    assert "Typical pay" in result.content
    assert any("salary_insights" in c for c in result.metadata["citations"])


async def test_analytics_agent_runs_readonly_sql():
    llm = ScriptedLLMClient(
        tool_responses=[
            ToolResponse(content=None, tool_calls=[tool_call("run_sql_query", {"query": "SELECT status, COUNT(*) AS count FROM applications GROUP BY status"})]),
            ToolResponse(content="You have applications across 6 statuses."),
        ]
    )
    agent = AnalyticsAgent(llm, SqliteSQLClient(role="basic"), _caps())
    result = await agent.handle("how many applications by status?")
    assert "6 statuses" in result.content


async def test_analytics_agent_blocks_write_sql():
    llm = ScriptedLLMClient(
        tool_responses=[
            ToolResponse(content=None, tool_calls=[tool_call("run_sql_query", {"query": "DELETE FROM applications"})]),
            ToolResponse(content="I can only read data."),
        ]
    )
    agent = AnalyticsAgent(llm, SqliteSQLClient(role="basic"), _caps())
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


async def test_notify_offline_draft_is_tailored_not_echo():
    """Offline Draft must produce a real, sendable message — parsing the company and
    role — not echo the user's request verbatim."""
    from careerpilot.integrations.offline.offline_llm import OfflineLLMClient

    outbox = DemoOutbox()
    agent = NotifyAgent(OfflineLLMClient(), _caps(), demo_outbox=outbox)
    await agent.handle("draft a follow-up email to the recruiter at Askable")
    assert agent.has_pending()
    body = agent.pending().body
    assert "draft a follow-up email" not in body.lower()  # not an echo
    assert "Askable" in body
    assert "Best regards" in body


async def test_notify_offline_cover_letter():
    from careerpilot.integrations.offline.offline_llm import OfflineLLMClient

    agent = NotifyAgent(OfflineLLMClient(), _caps(), demo_outbox=DemoOutbox())
    await agent.handle("write a cover letter for the Senior GenAI Engineer role at Acme")
    p = agent.pending()
    assert "Dear Hiring Team" in p.body
    assert "GenAI" in p.subject and "Acme" in p.subject


async def test_jobs_agent_multi_tool_search_then_company():
    """'roles at <Company>' must drive a multi-tool loop: search_jobs THEN get_company."""
    from careerpilot.integrations.offline.offline_llm import OfflineLLMClient

    events: list[dict] = []
    agent = JobsAgent(OfflineLLMClient(), FakeJobsMCP(), _caps())
    await agent.handle("find GenAI roles at Nimbus Labs", on_event=lambda e: events.append(e))
    tools = [e["tool"] for e in events if e["type"] == "tool_called"]
    assert tools == ["search_jobs", "get_company"]


async def test_jobs_agent_company_only_lookup():
    from careerpilot.integrations.offline.offline_llm import OfflineLLMClient

    events: list[dict] = []
    agent = JobsAgent(OfflineLLMClient(), FakeJobsMCP(), _caps())
    await agent.handle("tell me about Nimbus Labs", on_event=lambda e: events.append(e))
    tools = [e["tool"] for e in events if e["type"] == "tool_called"]
    assert tools == ["get_company"]  # no job search for a pure company question
