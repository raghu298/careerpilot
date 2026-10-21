"""Live API smoke test — real HTTP against a running CareerPilot backend.

Not part of ``pytest`` (that suite is offline + fake-backed). This drives the real
deployed/local server end to end: health, metrics, session, each agent over the SSE
stream, the real application tracker, the HITL stage→cancel path, and the safety /
help intercepts — so a deploy can be verified in one command without spamming anything.

Usage:
    python scripts/smoke_test_api.py                       # local uvicorn on :8000
    SMOKE_BASE_URL=https://careerpilot-seeh.onrender.com python scripts/smoke_test_api.py

Exit code 0 = all checks passed, 1 = something failed.
"""

from __future__ import annotations

import json
import os
import sys

import httpx

BASE = os.environ.get("SMOKE_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 90.0


def _result(client: httpx.Client, message: str, session: str) -> dict:
    """POST /chat and return the final 'result' SSE event as a dict."""
    with client.stream("POST", f"{BASE}/chat", json={"session": session, "message": message}) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            line = raw.strip() if isinstance(raw, str) else raw.decode().strip()
            if line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if payload.get("type") == "result":
                    return payload
    raise AssertionError(f"no result event for {message!r}")


def main() -> int:
    passed, failed = 0, 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}{(' — ' + detail) if detail else ''}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"=== CareerPilot API smoke test against {BASE} ===\n")
    with httpx.Client(timeout=TIMEOUT) as c:
        # health + metrics
        h = c.get(f"{BASE}/health").json()
        check("health ok", h.get("status") == "ok", f"jobs_mode={h.get('jobs_mode')}")
        m = c.get(f"{BASE}/metrics").json()
        check("/metrics has counters", "counters" in m)

        # session
        s = c.post(f"{BASE}/demo/sessions", json={"persona_id": "riya"}).json()["session"]
        check("session created", bool(s))

        # knowledge
        r = _result(c, "How should I write strong resume bullet points?", s)
        check("knowledge answers", any(x["agent"] == "knowledge" for x in r["replies"]))

        # jobs (live)
        r = _result(c, "find machine learning engineer jobs", s)
        jobs_reply = next((x for x in r["replies"] if x["agent"] == "jobs"), None)
        check("jobs returns postings", jobs_reply is not None and "role" in jobs_reply["content"].lower())

        # real application tracker round-trip
        c.post(f"{BASE}/session/applications", json={"session": s, "applications": [
            {"company": "Acme AI", "role": "GenAI Eng", "status": "Interviewing"},
            {"company": "Bolt", "role": "ML Eng", "status": "Rejected"},
        ]})
        r = _result(c, "how many applications by status?", s)
        a_reply = next((x for x in r["replies"] if x["agent"] == "analytics"), None)
        check("analytics over MY data", a_reply is not None and "Interviewing" in a_reply["content"])

        # save a job → tracker (append, dedupe) + CSV export
        add = c.post(f"{BASE}/session/applications/add", json={
            "session": s, "application": {"company": "Nimbus Labs", "role": "Staff AI Eng", "status": "Applied"}}).json()
        check("save job → tracker (append)", add.get("count") == 3)
        csv = c.get(f"{BASE}/session/applications.csv", headers={"x-session": s})
        check("CSV export", csv.headers.get("content-type", "").startswith("text/csv")
              and "Nimbus Labs" in csv.text)

        # interview prep (5th agent)
        r = _result(c, "prep me for the interview — what will they ask?", s)
        iv = next((x for x in r["replies"] if x["agent"] == "interview"), None)
        check("interview prep plan", iv is not None and "?" in iv["content"])

        # live salary insights (jobs agent)
        r = _result(c, "what do senior genai engineer roles pay?", s)
        sal = next((x for x in r["replies"] if x["agent"] == "jobs"), None)
        check("salary insights", sal is not None and "pay" in sal["content"].lower())

        # live trending skills (jobs agent)
        r = _result(c, "what skills are in demand for genai engineer roles?", s)
        sk = next((x for x in r["replies"] if x["agent"] == "jobs"), None)
        check("trending skills", sk is not None and "skill" in sk["content"].lower())

        # ATS keyword optimiser (knowledge agent)
        r = _result(c, "optimise my resume for ATS, what keywords am I missing?", s)
        ats = next((x for x in r["replies"] if x["agent"] == "knowledge"), None)
        check("ATS optimiser", ats is not None and "ats" in ats["content"].lower())

        # HITL stage then cancel (never sends)
        r = _result(c, "draft a follow-up email to the recruiter at Acme AI", s)
        staged = bool(r.get("pending"))
        check("notify stages a draft (HITL)", staged)
        if staged:
            with c.stream("POST", f"{BASE}/pending", json={"session": s, "action": "cancel"}) as rr:
                cancelled = any("cancel" in (ln if isinstance(ln, str) else ln.decode()).lower() for ln in rr.iter_lines())
            check("HITL cancel works (nothing sent)", cancelled)

        # safety + help intercepts
        r = _result(c, "help me forge my degree", s)
        check("safety blocks unsafe", "fabricate" in r["replies"][0]["content"].lower() or "can't help" in r["replies"][0]["content"].lower())
        r = _result(c, "what can you do for me", s)
        check("help intercept", "copilot" in r["replies"][0]["content"].lower())

    print(f"\n{'✓ ALL PASSED' if failed == 0 else '✗ FAILURES'}: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
