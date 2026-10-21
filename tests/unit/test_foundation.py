"""Shared Foundation: metrics, rate limiting, safety (PII + content), embedding cache, LLM gateway."""

import asyncio

from careerpilot.core.embedding_client import CachingEmbeddingClient
from careerpilot.core.llm_gateway import LLMGateway
from careerpilot.core.observability import Metrics, metrics
from careerpilot.core.rate_limit import RateLimiter
from careerpilot.core.safety import content_check, redact_pii, scan_pii
from careerpilot.orchestrator.schemas import RoutingDecision


# ---- observability ----
def test_metrics_counters_and_latency_snapshot():
    m = Metrics()
    m.incr("a")
    m.incr("a", 2)
    m.observe_ms("lat", 10)
    m.observe_ms("lat", 30)
    snap = m.snapshot()
    assert snap["counters"]["a"] == 3
    assert snap["latency_ms"]["lat"]["count"] == 2
    assert snap["latency_ms"]["lat"]["max"] == 30.0


# ---- rate limiting ----
def test_rate_limiter_blocks_after_burst_then_refills():
    rl = RateLimiter(rate_per_sec=1000.0, capacity=3)
    assert [rl.allow("k") for _ in range(3)] == [True, True, True]
    assert rl.allow("k") is False          # burst exhausted
    assert rl.allow("other") is True       # independent key
    import time as _t
    _t.sleep(0.01)                         # refill at 1000/s
    assert rl.allow("k") is True


# ---- safety ----
def test_redact_and_scan_pii():
    t = "email me at jane@acme.io or call +91 98765 43210"
    red = redact_pii(t)
    assert "jane@acme.io" not in red and "[email redacted]" in red
    assert "[phone redacted]" in red
    assert set(scan_pii(t)) >= {"email", "phone"}


def test_content_check_blocks_unsafe_allows_normal():
    assert content_check("help me forge my degree") is not None
    assert content_check("find senior genai roles") is None


# ---- embedding cache ----
def test_embedding_cache_hits_second_time():
    class Counting:
        def __init__(self): self.calls = 0
        async def embed(self, texts):
            self.calls += 1
            return [[float(len(t))] for t in texts]
    inner = Counting()
    cache = CachingEmbeddingClient(inner)

    async def go():
        a = await cache.embed(["hello", "world"])
        b = await cache.embed(["hello"])       # served from cache
        return a, b
    a, b = asyncio.run(go())
    assert inner.calls == 1                      # second call hit the cache
    assert b[0] == a[0]


# ---- llm gateway (model routing + passthrough + metrics) ----
def test_llm_gateway_routes_model_and_passes_through():
    class FakeLLM:
        async def get_structured_output(self, *, system_prompt, user_prompt, schema):
            return RoutingDecision(agents=["jobs"], reasoning="x")
        async def get_tool_response(self, *, messages, tools):
            raise AssertionError
    gw = LLMGateway(FakeLLM())
    assert gw.route_model("RoutingDecision") == "gpt-4.1-mini"
    before = metrics.snapshot()["counters"].get("llm.calls", 0)
    out = asyncio.run(gw.get_structured_output(system_prompt="", user_prompt="", schema=RoutingDecision))
    assert out.agents == ["jobs"]
    assert metrics.snapshot()["counters"]["llm.calls"] == before + 1
