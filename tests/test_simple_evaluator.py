"""
Unit tests for the new simple evaluator.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.ai.models import AgentResponse, SupportResponse
from src.evaluation.evaluator import SimpleEvaluator


class TestSimpleEvaluator:
    """Test suite for the lightweight SimpleEvaluator."""

    def test_run_evaluation_success(self):
        # Sample data
        data = {
            "issue": ["Need refund", "Password reset"],
            "subject": ["Billing question", "Account settings"],
            "company": ["Visa", "HackerRank"],
            "expected_request_type": ["invalid", "product_issue"],
            "expected_product_area": ["billing_invoicing", "account_settings"],
            "expected_status": ["replied", "replied"],
        }
        df = pd.DataFrame(data)

        # Mock AgentResponse
        mock_response_1 = AgentResponse(
            response=SupportResponse(
                request_type="invalid",
                product_area="billing_invoicing",
                status="Replied",
                response="Here is your refund.",
                justification="Simple billing request.",
            ),
            retrieved_context=[],
            retrieved_chunks=[],
            num_chunks=0,
            retrieval_required=False,
            routing_reason="Processed successfully",
            decision_input_tokens=100,
            decision_output_tokens=50,
            generation_input_tokens=200,
            generation_output_tokens=100,
        )

        mock_response_2 = AgentResponse(
            response=SupportResponse(
                request_type="product_issue",
                product_area="account_settings",
                status="Replied",
                response="Password reset link sent.",
                justification="Simple account settings request.",
            ),
            retrieved_context=[],
            retrieved_chunks=[],
            num_chunks=0,
            retrieval_required=False,
            routing_reason="Processed successfully",
            decision_input_tokens=100,
            decision_output_tokens=50,
            generation_input_tokens=200,
            generation_output_tokens=100,
        )

        # Create evaluator instance
        config = {
            "reranker": "none",
            "decision_provider": "openai",
            "decision_model": "gpt-5-mini",
            "generation_provider": "openai",
            "generation_model": "gpt-5.4-mini",
            "eval_provider": "openai",
            "eval_model": "gpt-5-mini",
        }
        evaluator = SimpleEvaluator(config)

        # Mock self.agent.invoke & ragas evaluation
        evaluator.agent.invoke = MagicMock(
            side_effect=[mock_response_1, mock_response_2]
        )
        mock_ragas = {
            "faithfulness": 0.88,
            "answer_relevancy": 0.85,
            "context_precision": 0.90,
            "context_recall": 0.87,
            "answer_correctness": 0.86,
            "avg_ragas_score": 0.872,
        }
        evaluator.ragas_evaluator.evaluate_generation = MagicMock(return_value=mock_ragas)

        # Run evaluation
        out = evaluator.run_evaluation(df)

        metrics = out["metrics"]
        predictions = out["predictions"]

        # Assert correct stats
        assert metrics["total_tickets"] == 2
        assert metrics["succeeded_predictions"] == 2
        assert metrics["failed_predictions"] == 0
        assert metrics["request_type_accuracy"] == 1.0
        assert metrics["product_area_accuracy"] == 1.0
        assert metrics["status_accuracy"] == 1.0
        assert metrics["avg_tokens"] == 450.0
        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics
        assert "context_precision" in metrics
        assert "context_recall" in metrics
        assert "answer_correctness" in metrics
        assert "avg_ragas_score" in metrics

        # Assert predictions list
        assert len(predictions) == 2
        assert predictions[0]["predicted_request_type"] == "invalid"
        assert predictions[1]["predicted_product_area"] == "account_settings"

    def test_run_evaluation_with_failures(self):
        # Sample data
        data = {
            "issue": ["Refund please", "Cannot login"],
            "subject": ["Refund", "Login error"],
            "company": ["Visa", "HackerRank"],
            "expected_request_type": ["invalid", "product_issue"],
            "expected_product_area": ["billing_invoicing", "account_settings"],
            "expected_status": ["replied", "replied"],
        }
        df = pd.DataFrame(data)

        # Create evaluator instance
        config = {
            "reranker": "none",
            "decision_provider": "openai",
            "decision_model": "gpt-5-mini",
            "generation_provider": "openai",
            "generation_model": "gpt-5.4-mini",
            "eval_provider": "openai",
            "eval_model": "gpt-5-mini",
        }
        evaluator = SimpleEvaluator(config)

        # Mock self.agent.invoke & ragas evaluation
        mock_response = AgentResponse(
            response=SupportResponse(
                request_type="product_issue",
                product_area="account_settings",
                status="Replied",
                response="Password reset link sent.",
                justification="Simple account settings request.",
            ),
            retrieved_context=[],
            retrieved_chunks=[],
            num_chunks=0,
            retrieval_required=False,
            routing_reason="Processed successfully",
            decision_input_tokens=100,
            decision_output_tokens=50,
            generation_input_tokens=200,
            generation_output_tokens=100,
        )
        evaluator.agent.invoke = MagicMock(
            side_effect=[Exception("Invoke timeout error"), mock_response]
        )
        mock_ragas = {
            "faithfulness": 0.88,
            "answer_relevancy": 0.85,
            "context_precision": 0.90,
            "context_recall": 0.87,
            "answer_correctness": 0.86,
            "avg_ragas_score": 0.872,
        }
        evaluator.ragas_evaluator.evaluate_generation = MagicMock(return_value=mock_ragas)

        # Run evaluation
        out = evaluator.run_evaluation(df)

        metrics = out["metrics"]
        predictions = out["predictions"]

        # Assertions
        assert metrics["total_tickets"] == 2
        assert metrics["succeeded_predictions"] == 1
        assert metrics["failed_predictions"] == 1
        assert (
            metrics["request_type_accuracy"] == 0.5
        )  # 1 match out of 2 total evaluated tickets
        assert predictions[0]["status"] == "FAILED"
        assert predictions[1]["status"] == "SUCCESS"
