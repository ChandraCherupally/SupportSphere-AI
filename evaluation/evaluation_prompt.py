"""
Generic evaluation prompt for SupportSphere AI.

This prompt is used by the evaluation framework to score
different quality metrics using an LLM-as-a-Judge.

Supported Metrics
-----------------
- Answer Correctness
- Faithfulness
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate


class EvaluationPromptBuilder:
    """
    Builds prompts for LLM-based evaluation.

    A single prompt template is reused for all evaluation metrics.
    The metric name determines what the judge should evaluate.
    """

    def __init__(self) -> None:

        self.template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert evaluator for AI-powered "
                        "customer support systems.\n\n"
                        "Evaluate ONLY the requested metric.\n"
                        "Be objective and strict.\n"
                        "Return ONLY valid JSON."
                    ),
                ),
                ("human", "{evaluation_prompt}"),
            ]
        )

    def build(
        self,
        *,
        metric: str,
        question: str,
        prediction: str,
        reference: str = "",
        retrieved_context: str = "",
    ) -> list[BaseMessage]:
        """
        Build evaluation messages.

        Parameters
        ----------
        metric
            Metric to evaluate.

        question
            Original user question.

        prediction
            Model-generated answer.

        reference
            Ground-truth answer.

        retrieved_context
            Retrieved documentation.
        """

        prompt = f"""
Metric
------
{metric}

Question
--------
{question}

Ground Truth
------------
{reference}

Retrieved Context
-----------------
{retrieved_context}

Generated Answer
----------------
{prediction}

Instructions
------------
Evaluate ONLY the requested metric.

Score between 0.0 and 1.0.

Scoring Guidelines

1.0
Perfect.

0.8
Minor issues.

0.6
Some important information missing.

0.4
Several factual problems.

0.2
Mostly incorrect.

0.0
Completely incorrect.

Return ONLY JSON.

{{
    "score": 0.0,
    "reasoning": "short explanation"
}}
"""

        return self.template.format_messages(
            evaluation_prompt=prompt,
        )