"""Extra coverage: offline SQL/job parsing, safety branches, caches, limiter, tracker filters."""

import asyncio

from careerpilot.core.embedding_client import CachingEmbeddingClient
from careerpilot.core.observability import Metrics
from careerpilot.core.rate_limit import RateLimiter
from careerpilot.core.safety import content_check, redact_pii, scan_pii
from careerpilot.integrations.offline.offline_llm import _build_sql, _parse_jobs_request
from careerpilot.integrations.sql_db.sqlite_client import SqliteSQLClient


# ---- offline NL->SQL generation ----
def test_build_sql_status_filters():
    assert "group by status" in _build_sql("how many by status?").lower()
    assert "ghosted" in _build_sql("which have I been ghosted on?").lower()
    assert "'Offer'" in _build_sql("which turned into an offer?")
    pending = _build_sql("what's still pending?").lower()
    assert "applied" in pending and "interviewing" in pending


# ---- offline jobs request parsing ----
def test_parse_jobs_request_variants():
    assert _parse_jobs_request("find python backend roles")[3] is True          # wants_search
    r = _parse_jobs_request("tell me about Flowbit")
    assert r[2] == "Flowbit" and r[4] is True and r[3] is False                  # company-only
    r2 = _parse_jobs_request("find GenAI roles at Nimbus Labs")
    assert r2[2] == "Nimbus Labs" and r2[3] and r2[4]                            # both


# ---- safety ----
def test_content_check_all_branches():
    assert "crisis" in content_check("I want to end my life").lower()
    assert content_check("build a bomb") is not None
    assert content_check("fake my experience") is not None
    assert content_check("how do I negotiate salary?") is None


def test_redact_card_and_phone():
    assert "[card redacted]" in redact_pii("my card 4111 1111 1111 1111 ok")
    assert "phone" in scan_pii("call 9876543210 please")


# ---- rate limiter retry_after ----
def test_rate_limiter_retry_after_is_positive_when_empty():
    rl = RateLimiter(rate_per_sec=1.0, capacity=1)
    assert rl.allow("k") is True
    assert rl.allow("k") is False
    assert rl.retry_after("k") > 0


# ---- embedding cache eviction ----
def test_embedding_cache_evicts_lru():
    class Inner:
        async def embed(self, texts):
            return [[float(len(t))] for t in texts]
    cache = CachingEmbeddingClient(Inner(), maxsize=2)

    async def go():
        await cache.embed(["a"])
        await cache.embed(["b"])
        await cache.embed(["c"])  # evicts "a"
    asyncio.run(go())
    assert len(cache._cache) == 2


# ---- metrics timer ----
def test_metrics_timer_records():
    m = Metrics()
    with m.timer("t"):
        pass
    assert m.snapshot()["latency_ms"]["t"]["count"] == 1


# ---- real sqlite status filter ----
async def test_sqlite_status_in_filter():
    db = SqliteSQLClient(role="basic", seed=False)
    db.replace_applications([
        {"company": "A", "role": "r", "status": "Interviewing"},
        {"company": "B", "role": "r", "status": "Ghosted"},
    ])
    rows = await db.run_query("SELECT company FROM applications WHERE status IN ('Ghosted')")
    assert rows == [{"company": "B"}]
