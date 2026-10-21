"""Golden dataset for the Knowledge Agent's RAG evaluation.

Every case is a real fact drawn from ``src/careerpilot/fixtures.py`` (the seeded career
documents) and, where noted, a real bug this project found and fixed during iterative
testing — turning those into a permanent regression suite rather than a one-off check.

``expected_source`` is matched against citation strings (the chunk ``source``, e.g.
``"resume-tips"``) with ``citation.startswith(expected_source)``. ``expected_output`` is
a short reference answer used by DeepEval's ContextualRecall/Precision metrics when a
real model is available — it is not an exact-match string. ``persona`` selects the
access scope the question should be asked under (premium docs need ``arjun``).
"""

from __future__ import annotations

KNOWLEDGE_GOLDEN_CASES: list[dict] = [
    {
        "question": "Does my resume fit the Senior GenAI JD? What are my gaps?",
        "persona": "arjun",
        "expected_source": "my-resume",
        "expected_output": (
            "The resume matches on Python, FastAPI, PostgreSQL/pgvector, SQL and retrieval, "
            "with gaps around LangGraph, agents, tool-calling, evaluation/observability and Kubernetes."
        ),
        "expect_answer_found": True,
        "note": "Core resume-vs-JD match/gap analysis — the headline feature.",
    },
    {
        "question": "How should I write strong resume bullet points?",
        "persona": "riya",
        "expected_source": "resume-tips",
        "expected_output": "Lead each bullet with impact and quantify results; use strong action verbs.",
        "expect_answer_found": True,
        "note": "Baseline guide retrieval.",
    },
    {
        "question": "How do I answer behavioural interview questions?",
        "persona": "riya",
        "expected_source": "interview-behavioural",
        "expected_output": "Use the STAR method — Situation, Task, Action, Result — with a concrete example.",
        "expect_answer_found": True,
        "note": "Baseline guide retrieval.",
    },
    {
        "question": "How should I negotiate my salary offer?",
        "persona": "arjun",
        "expected_source": "salary-negotiation-premium",
        "expected_output": "Negotiate the whole package, anchor on market data, and never accept on the first call.",
        "expect_answer_found": True,
        "note": "Premium-scoped doc — must be retrievable by arjun (premium).",
    },
    {
        "question": "How should I negotiate my salary offer?",
        "persona": "riya",
        "expected_source": None,  # premium doc out of scope for riya
        "expected_output": None,
        "expect_answer_found": False,
        "note": "RBAC regression: the premium salary doc must NOT be retrievable by riya (general only).",
    },
    {
        "question": "What's the best day to launch a rocket to Mars?",
        "persona": "riya",
        "expected_source": None,  # nothing in the corpus
        "expected_output": None,
        "expect_answer_found": False,
        "note": "Out-of-scope groundedness: with no relevant doc, answer_found must be False.",
    },
]
