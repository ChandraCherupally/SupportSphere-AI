from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class EvaluationResult(BaseModel):
    """
    Structured evaluation returned by the judge model.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Evaluation score between 0 and 1.",
    )

    reasoning: str = Field(
        description="Short explanation supporting the score.",
    )