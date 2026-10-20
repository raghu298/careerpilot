from careerpilot.core.agent import AgentResult, Capabilities
from careerpilot.orchestrator.orchestrator import Orchestrator
from careerpilot.orchestrator.router import Router
from careerpilot.orchestrator.schemas import RoutingDecision
from careerpilot.agents.notify.agent import Draft, NotifyAgent
from careerpilot.integrations.notifications.notifiers import DemoOutbox
from tests.unit.fakes import ScriptedLLMClient


class RecordingAgent:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    async def handle(self, user_request, history=None, on_event=None, tool_cache=None):
        self.calls += 1
        return AgentResult(self.name, f"{self.name} answered")


def _caps():
    return Capabilities(persona_id="riya", display_name="Riya", knowledge_labels=("general",), sql_role="basic")


async def test_orchestrator_fans_out_to_selected_agents():
    router_llm = ScriptedLLMClient(
        structured={"RoutingDecision": RoutingDecision(agents=["jobs", "analytics"], reasoning="both")}
    )
    agents = {"jobs": RecordingAgent("jobs"), "analytics": RecordingAgent("analytics"), "knowledge": RecordingAgent("knowledge")}
    orch = Orchestrator(Router(router_llm), agents)
    events = []
    out = await orch.handle("find roles and show my pipeline", on_event=lambda e: events.append(e["type"]))
    assert set(out.results.keys()) == {"jobs", "analytics"}
    assert agents["knowledge"].calls == 0
    assert "routing_decided" in events and "done" in events


async def test_orchestrator_surfaces_pending_then_resumes():
    router_llm = ScriptedLLMClient(
        structured={"RoutingDecision": RoutingDecision(agents=["notify"], reasoning="draft")}
    )
    notify_llm = ScriptedLLMClient(structured={"Draft": Draft(channel="email", subject="Hi", body="Body")})
    outbox = DemoOutbox()
    agents = {"notify": NotifyAgent(notify_llm, _caps(), demo_outbox=outbox)}
    orch = Orchestrator(Router(router_llm), agents)

    out = await orch.handle("draft a follow-up")
    assert out.pending is not None
    assert out.pending_agent == "notify"
    assert out.pending.kind == "message"

    resumed = await orch.resume_pending({"action": "confirm"})
    assert len(outbox.items()) == 1
    assert "notify" in resumed.results


async def test_orchestrator_memory_accumulates():
    router_llm = ScriptedLLMClient(
        structured={"RoutingDecision": [
            RoutingDecision(agents=["jobs"], reasoning="1"),
            RoutingDecision(agents=["jobs"], reasoning="2"),
        ]}
    )
    orch = Orchestrator(Router(router_llm), {"jobs": RecordingAgent("jobs")})
    await orch.handle("first")
    await orch.handle("second")
    text = orch.memory.as_text()
    assert "first" in text and "second" in text
