"""Schema-guarded output for the Knowledge Agent's generation step.

The same hard-schema mechanism the router uses is reused here: the model must return
this exact shape, so ``answer_found`` is a real, structured groundedness signal rather
than something parsed out of free text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeAnswer(BaseModel):
    answer: str = Field(description="The answer, grounded only in the provided context.")
    citations: list[str] = Field(
        default_factory=list, description="Sources actually used, from the context."
    )
    answer_found: bool = Field(
        description="True only if the context genuinely supports the answer; False if not."
    )
