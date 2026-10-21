from pydantic import ValidationError

from careerpilot.integrations.offline.offline_llm import OfflineLLMClient
from careerpilot.orchestrator.router import Router, RoutingError
from careerpilot.orchestrator.schemas import RoutingDecision
from tests.unit.fakes import ScriptedLLMClient


async def test_route_single_agent():
    llm = ScriptedLLMClient(
        structured={"RoutingDecision": RoutingDecision(agents=["jobs"], reasoning="live listings")}
    )
    decision = await Router(llm).route("find backend roles in Bangalore")
    assert decision.agents == ["jobs"]


async def test_route_multi_agent_fan_out():
    llm = ScriptedLLMClient(
        structured={
            "RoutingDecision": RoutingDecision(agents=["jobs", "notify"], reasoning="find + draft")
        }
    )
    decision = await Router(llm).route("find roles and draft a cover letter")
    assert set(decision.agents) == {"jobs", "notify"}


async def test_router_retries_then_raises_on_invalid():
    class AlwaysInvalid:
        def __init__(self):
            self.calls = 0

        async def get_structured_output(self, **_):
            self.calls += 1
            raise ValidationError.from_exception_data("RoutingDecision", [])

        async def get_tool_response(self, **_):  # pragma: no cover
            raise AssertionError

    llm = AlwaysInvalid()
    try:
        await Router(llm, max_retries=2).route("anything")
        assert False, "expected RoutingError"
    except RoutingError:
        pass
    assert llm.calls == 3  # initial + 2 retries


async def test_offline_router_ignores_history_cross_talk():
    """Offline brain must route on the CURRENT request only — prior 'find jobs' /
    'draft email' turns in the history must not keep re-triggering those agents."""
    history = (
        "user: find remote python jobs\n"
        "assistant: I found 2 roles...\n"
        "user: draft a follow-up email to the recruiter\n"
        "assistant: [email] staged for confirmation"
    )
    decision = await Router(OfflineLLMClient()).route(
        "how many applications have I sent by status?", history
    )
    assert decision.agents == ["analytics"]


async def test_offline_router_current_request_only_jobs():
    history = "user: how many applications are pending?\nassistant: Applied (1)..."
    decision = await Router(OfflineLLMClient()).route(
        "find remote backend engineer roles", history
    )
    assert decision.agents == ["jobs"]


async def test_offline_router_routes_interview():
    """Mock-interview requests go to the interview agent, not knowledge."""
    decision = await Router(OfflineLLMClient()).route(
        "prep me for the interview — what questions will they ask?"
    )
    assert decision.agents == ["interview"]
