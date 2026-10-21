"""Schema-guarded output for the Interview Prep agent.

Same hard-schema mechanism as the router and the Knowledge agent: the model must
return this exact shape. ``grounded`` is a structured signal for whether the plan was
built from the user's own resume/JD (vs. generic practice), and every ``question``
carries a ``talking_point`` so the output is actionable, not just a question list.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionCategory = Literal["technical", "behavioral", "system-design", "role-specific"]


class InterviewQuestion(BaseModel):
    category: QuestionCategory = Field(description="Which kind of interview question this is.")
    question: str = Field(description="A question the interviewer is likely to ask.")
    talking_point: str = Field(
        description="A concrete prep note grounded in the user's resume/JD — what to emphasise."
    )


class InterviewPlan(BaseModel):
    focus: str = Field(description="One line on what this candidate should emphasise overall.")
    questions: list[InterviewQuestion] = Field(
        default_factory=list, description="Likely questions with grounded talking points."
    )
    grounded: bool = Field(
        description="True if the plan was built from the user's actual resume/JD; False if generic."
    )
