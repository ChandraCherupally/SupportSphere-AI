"""
SupportSphere Evaluation Experiment.

Runs the SupportSphere agent against the evaluation dataset,
computes business metrics, evaluates the generated answer
using an LLM-as-a-Judge, and saves the results.

This module is intentionally independent of Streamlit.
"""

from __future__ import annotations

import pandas as pd

from evaluation.dataset import EvaluationDataset

from src.graph.support_agent import SupportAgent
from src.ai.models import SupportTicket

from src.ai.client import LLMClient
from evaluation.evaluation_models import EvaluationResult
from evaluation.evaluation_prompt import EvaluationPromptBuilder


class EvaluationExperiment:

    def __init__(self) -> None:
        self.dataset = EvaluationDataset()
        self.agent = SupportAgent()
        self.llm = LLMClient()
        self.prompt_builder = EvaluationPromptBuilder()

    # ---------------------------------------------------------
    # Exact Match
    # ---------------------------------------------------------

    @staticmethod
    def exact_match(
        prediction: str,
        expected: str,
    ) -> int:

        return int(
            prediction.strip().lower()
            ==
            expected.strip().lower()
        )

    # ---------------------------------------------------------
    # Generic Judge
    # ---------------------------------------------------------

    def judge(
        self,
        *,
        metric: str,
        question: str,
        prediction: str,
        reference: str,
        retrieved_context: list[str],
    ) -> EvaluationResult:

        messages = self.prompt_builder.build(
            metric=metric,
            question=question,
            prediction=prediction,
            reference=reference,
            retrieved_context="\n\n".join(
                retrieved_context
            ),
        )

        return self.llm.generate(
            messages=messages,
            response_schema=EvaluationResult,
        )

    # ---------------------------------------------------------
    # Evaluate One Example
    # ---------------------------------------------------------

    def evaluate_example(
        self,
        example: dict,
    ) -> dict:

        ticket = SupportTicket(

            issue=example["issue"],

            subject=example["subject"],

            company=example["company"],
        )

        import time
        start_time = time.time()
        prediction = self.agent.invoke(ticket)
        latency = time.time() - start_time

        correctness = self.judge(

            metric="Answer Correctness",

            question=example["issue"],

            prediction=prediction.response.response,

            reference=example["expected_response"],

            retrieved_context=prediction.retrieved_context,
        )

        faithfulness = self.judge(

            metric="Faithfulness",

            question=example["issue"],

            prediction=prediction.response.response,

            reference=example["expected_response"],

            retrieved_context=prediction.retrieved_context,
        )

        return {

            # --------------------------------------------------
            # Inputs
            # --------------------------------------------------

            "company": example["company"],

            "subject": example["subject"],

            "issue": example["issue"],

            # --------------------------------------------------
            # Business Metrics
            # --------------------------------------------------

            "request_type_accuracy": self.exact_match(
                prediction.response.request_type,
                example["expected_request_type"],
            ),

            "status_accuracy": self.exact_match(
                prediction.response.status,
                example["expected_status"],
            ),

            "product_area_accuracy": self.exact_match(
                prediction.response.product_area,
                example["expected_product_area"],
            ),

            # --------------------------------------------------
            # Judge Metrics
            # --------------------------------------------------

            "answer_correctness":
                correctness.score,

            "faithfulness":
                faithfulness.score,

            "judge_reasoning":
                faithfulness.reasoning,

            # --------------------------------------------------
            # Runtime
            # --------------------------------------------------

            "latency":
                latency,

            "num_chunks":
                prediction.num_chunks,

            "token_estimate":
                prediction.token_estimate,
        }

    # ---------------------------------------------------------
    # Run Evaluation
    # ---------------------------------------------------------

    def run(self) -> pd.DataFrame:

        examples = self.dataset.load()

        rows = []

        for example in examples:

            rows.append(
                self.evaluate_example(
                    example
                )
            )

        df = pd.DataFrame(rows)

        return df

