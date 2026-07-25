import unittest
from unittest.mock import MagicMock, patch

from src.ai.models import DecisionResult
from src.graph.nodes import decision_engine_node, retrieve_node


class TestCompanyDetection(unittest.TestCase):
    def test_correct_company_selected(self):
        """Case 1: selected_company == detected_company."""
        state = {
            "issue": "Please pause our subscription on HackerRank.",
            "subject": "Subscription pause",
            "company": "HackerRank",
        }

        mock_decision = DecisionResult(
            safe=True,
            intent="PRODUCT_SUPPORT",
            company="HackerRank",
            routing_strategy="SUPPORT_RESPONSE",
            retrieval_required=True,
            reason="HackerRank subscription references.",
            confidence=0.95,
            company_reference_found=True,
            routing_status="COMPANY_VERIFIED",
            normalized_issue="pause subscription",
            normalized_subject="subscription pause",
        )

        with patch("src.graph.nodes.decision_engine.llm.generate") as mock_gen:
            mock_result = MagicMock()
            mock_result.response = mock_decision
            mock_result.usage.input_tokens = 100
            mock_result.usage.output_tokens = 50
            mock_gen.return_value = mock_result

            res = decision_engine_node(state)

            self.assertTrue(res["company_match"])
            self.assertEqual(res["verified_company"], "HackerRank")
            self.assertEqual(res["detected_company"], "HackerRank")
            self.assertEqual(res["routing_status"], "COMPANY_VERIFIED")
            self.assertTrue(res["company_reference_found"])

    def test_wrong_company_selected_and_automatically_corrected(self):
        """Case 2: selected_company != detected_company."""
        state = {
            "issue": "Claude Sonnet is throwing timeouts when calling the Anthropic API.",
            "subject": "Claude API timeout",
            "company": "Visa",
        }

        mock_decision = DecisionResult(
            safe=True,
            intent="PRODUCT_SUPPORT",
            company="Claude",
            routing_strategy="SUPPORT_RESPONSE",
            retrieval_required=True,
            reason="References Claude Sonnet and Anthropic API.",
            confidence=0.98,
            company_reference_found=True,
            routing_status="COMPANY_MISMATCH",
            normalized_issue="claude sonnet timeout",
            normalized_subject="api timeout",
        )

        with patch("src.graph.nodes.decision_engine.llm.generate") as mock_gen:
            mock_result = MagicMock()
            mock_result.response = mock_decision
            mock_result.usage.input_tokens = 100
            mock_result.usage.output_tokens = 50
            mock_gen.return_value = mock_result

            res = decision_engine_node(state)

            self.assertFalse(res["company_match"])
            self.assertEqual(res["verified_company"], "Claude")
            self.assertEqual(res["detected_company"], "Claude")
            self.assertEqual(res["routing_status"], "COMPANY_MISMATCH")
            self.assertTrue(res["company_reference_found"])

    def test_unknown_company_low_confidence(self):
        """Case 3: detected_company is None (no reference)."""
        state = {
            "issue": "General random issue inquiry about something else.",
            "subject": "Random inquiry",
            "company": "Visa",
        }

        mock_decision = DecisionResult(
            safe=True,
            intent="PRODUCT_SUPPORT",
            company="",
            routing_strategy="SUPPORT_RESPONSE",
            retrieval_required=True,
            reason="No explicit company reference found. Using user-selected company.",
            confidence=0.90,
            company_reference_found=False,
            routing_status="NO_COMPANY_REFERENCE",
            normalized_issue="random inquiry",
            normalized_subject="random inquiry",
        )

        with patch("src.graph.nodes.decision_engine.llm.generate") as mock_gen:
            mock_result = MagicMock()
            mock_result.response = mock_decision
            mock_result.usage.input_tokens = 100
            mock_result.usage.output_tokens = 50
            mock_gen.return_value = mock_result

            res = decision_engine_node(state)

            self.assertTrue(res["company_match"])
            self.assertEqual(res["verified_company"], "Visa")
            self.assertIsNone(res["detected_company"])
            self.assertFalse(res["company_reference_found"])
            self.assertEqual(res["routing_status"], "NO_COMPANY_REFERENCE")

    @patch("src.graph.nodes.retrieval.get_retriever")
    def test_retriever_always_uses_verified_company(self, mock_get_retriever):
        """Verify that retrieve_node passes verified_company to Retriever.retrieve."""
        state = {
            "issue": "Claude Sonnet is throwing timeouts when calling the Anthropic API.",
            "subject": "Claude API timeout",
            "company": "Visa",
            "verified_company": "Claude",
        }

        mock_retriever_inst = MagicMock()
        mock_retriever_inst.retrieve.return_value = {
            "context": [],
            "retrieval_trace": {},
        }
        mock_get_retriever.return_value = mock_retriever_inst

        retrieve_node(state)

        mock_retriever_inst.retrieve.assert_called_once_with(
            issue="Claude Sonnet is throwing timeouts when calling the Anthropic API.",
            subject="Claude API timeout",
            company="Claude",  # Must be verified_company!
        )
