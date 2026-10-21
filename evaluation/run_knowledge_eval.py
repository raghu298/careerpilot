"""RAG evaluation for the Knowledge Agent against ``evaluation/datasets/knowledge_golden.py``.

Runs in two tiers:

1. **Offline (always, no API key):**
   - deterministic regression checks — does the expected source get cited, and does
     ``answer_found`` match the expectation (groundedness + RBAC scope enforcement)?
   - model-agnostic retrieval ranking metrics computed directly off embedding-similarity
     ordering: Recall@1/@3/@5 and MRR. No LLM judge, pure math over ``vector_store.search()``.

2. **DeepEval RAG metrics (only when ``OPENAI_API_KEY`` is set and ``deepeval`` is
   installed):** ContextualPrecision/Recall, Faithfulness, AnswerRelevancy — real
   LLM-judge metrics. Skipped cleanly otherwise so the harness stays runnable with zero
   config and zero cost.

Usage:  python -m evaluation.run_knowledge_eval
"""

from __future__ import annotations

import asyncio
import os
import sys

from careerpilot.access_policy import resolve_capabilities
from careerpilot.agents.knowledge.agent import KnowledgeAgent
from careerpilot.bootstrap import build_shared_resources
from evaluation.datasets.knowledge_golden import KNOWLEDGE_GOLDEN_CASES

RETRIEVAL_K = 5


def _knowledge_agent(resources, persona: str) -> KnowledgeAgent:
    caps = resolve_capabilities(persona)
    floor = 0.05 if resources.offline else 0.2
    return KnowledgeAgent(
        resources.llm, resources.embeddings, resources.vector_store, caps, retrieval_floor=floor
    )


async def main() -> int:
    resources = await build_shared_resources()
    mode = "OFFLINE (deterministic brain)" if resources.offline else "REAL (OpenAI)"
    print(f"=== Knowledge Agent eval — {mode} ===\n")

    failures: list[str] = []
    ranks: list[int | None] = []
    deepeval_cases = []

    for case in KNOWLEDGE_GOLDEN_CASES:
        q, persona = case["question"], case.get("persona", "riya")
        caps = resolve_capabilities(persona)
        agent = _knowledge_agent(resources, persona)
        result = await agent.handle(q)
        citations = result.metadata.get("citations", [])
        answer_found = result.metadata.get("answer_found")

        print(f"Q [{persona}]: {q}")
        print(f"  answer_found: {answer_found}   citations: {citations}")

        # (1) deterministic regression checks
        exp_src = case["expected_source"]
        if exp_src is not None:
            if not any(str(c).startswith(exp_src) for c in citations):
                failures.append(f"[cite] {q!r}: expected source {exp_src!r} not cited (got {citations})")
                print(f"  ✗ expected source {exp_src!r} NOT cited")
            else:
                print(f"  ✓ expected source {exp_src!r} cited")

        exp_found = case["expect_answer_found"]
        if answer_found != exp_found:
            failures.append(f"[found] {q!r}: expected answer_found={exp_found}, got {answer_found}")
            print(f"  ✗ expected answer_found={exp_found}, got {answer_found}")
        else:
            print(f"  ✓ answer_found={answer_found} as expected")

        # (2) ranking metrics — only for cases with a known expected doc
        if exp_src is not None:
            [qvec] = await resources.embeddings.embed([q])
            chunks = await resources.vector_store.search(
                embedding=qvec, scopes=caps.knowledge_labels, k=RETRIEVAL_K
            )
            rank = next((i for i, c in enumerate(chunks, 1) if c.source.startswith(exp_src)), None)
            ranks.append(rank)
            print(f"  expected-doc rank in top {RETRIEVAL_K}: {rank or f'> {RETRIEVAL_K}'}")
            deepeval_cases.append(
                {"input": q, "actual": result.content, "expected": case["expected_output"],
                 "context": [c.content for c in chunks[:3]]}
            )
        print()

    # ---- ranking summary ----
    if ranks:
        print("=== Retrieval ranking (embedding similarity only, no LLM judge) ===")
        n = len(ranks)
        for k in (1, 3, 5):
            hits = sum(1 for r in ranks if r is not None and r <= k)
            print(f"  Recall@{k}: {hits}/{n} ({hits / n:.0%})")
        mrr = sum((1 / r) if r else 0.0 for r in ranks) / n
        print(f"  MRR:       {mrr:.3f}\n")

    # ---- optional DeepEval tier ----
    _maybe_run_deepeval(deepeval_cases)

    # ---- verdict ----
    if failures:
        print(f"✗ {len(failures)} regression check(s) FAILED:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("✓ All deterministic regression checks passed.")
    return 0


def _maybe_run_deepeval(cases: list[dict]) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("=== DeepEval RAG metrics: SKIPPED (no OPENAI_API_KEY — offline tier only) ===\n")
        return
    try:
        from deepeval import evaluate
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        print("=== DeepEval RAG metrics: SKIPPED (deepeval not installed — `pip install -e '.[eval]'`) ===\n")
        return

    print("=== DeepEval RAG metrics (real LLM judge) ===\n")
    test_cases = [
        LLMTestCase(input=c["input"], actual_output=c["actual"],
                    expected_output=c["expected"], retrieval_context=c["context"])
        for c in cases
    ]
    evaluate(
        test_cases=test_cases,
        metrics=[
            ContextualPrecisionMetric(),
            ContextualRecallMetric(),
            FaithfulnessMetric(),
            AnswerRelevancyMetric(),
        ],
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
