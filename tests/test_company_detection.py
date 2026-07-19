import unittest
from unittest.mock import patch, MagicMock
from src.graph.nodes import decision_gate_node, retrieve_node, RoutingDecision

class TestCompanyDetection(unittest.TestCase):
    def test_correct_company_selected(self):
        """Case 1: selected_company == detected_company, confidence >= 0.90."""
        state = {
            "issue": "Please pause our subscription on HackerRank.",
            "subject": "Subscription pause",
            "company": "HackerRank",
        }
        
        mock_decision = RoutingDecision(
            selected_company="HackerRank",
            detected_company="HackerRank",
            verified_company="HackerRank",
            company_match=True,
            company_confidence=0.95,
            retrieval_required=True,
            normalized_issue="pause subscription",
            normalized_subject="subscription pause",
            classification_confidence=0.95,
            routing_reason="HackerRank subscription references."
        )

        with patch("src.graph.nodes.llm.generate") as mock_gen:
            # Mock generate return
            mock_result = MagicMock()
            mock_result.response = mock_decision
            mock_result.usage.input_tokens = 100
            mock_result.usage.output_tokens = 50
            mock_gen.return_value = mock_result
            
            res = decision_gate_node(state)
            
            self.assertTrue(res["company_match"])
            self.assertEqual(res["verified_company"], "HackerRank")
            self.assertEqual(res["detected_company"], "HackerRank")
            self.assertEqual(res["company_confidence"], 0.95)

    def test_wrong_company_selected_and_automatically_corrected(self):
        """Case 2: selected_company != detected_company, confidence >= 0.90 (auto-corrected)."""
        state = {
            "issue": "Claude Sonnet is throwing 500 timeouts when calling the Anthropic API.",
            "subject": "Claude API timeout",
            "company": "Visa",
        }
        
        mock_decision = RoutingDecision(
            selected_company="Visa",
            detected_company="Claude",
            verified_company="Claude",
            company_match=False,
            company_confidence=0.98,
            retrieval_required=True,
            normalized_issue="claude sonnet timeout",
            normalized_subject="api timeout",
            classification_confidence=0.98,
            routing_reason="References Claude Sonnet and Anthropic API."
        )

        with patch("src.graph.nodes.llm.generate") as mock_gen:
            mock_result = MagicMock()
            mock_result.response = mock_decision
            mock_result.usage.input_tokens = 100
            mock_result.usage.output_tokens = 50
            mock_gen.return_value = mock_result
            
            res = decision_gate_node(state)
            
            self.assertFalse(res["company_match"])
            self.assertEqual(res["verified_company"], "Claude")
            self.assertEqual(res["detected_company"], "Claude")
            self.assertEqual(res["company_confidence"], 0.98)

    def test_unknown_company_low_confidence(self):
        """Case 3: detected_company is Unknown, confidence < 0.60 (keep selected)."""
        state = {
            "issue": "General random issue inquiry about something else.",
            "subject": "Random inquiry",
            "company": "Visa",
        }
        
        mock_decision = RoutingDecision(
            selected_company="Visa",
            detected_company="Unknown",
            verified_company="Visa",
            company_match=False,
            company_confidence=0.45,
            retrieval_required=True,
            normalized_issue="random inquiry",
            normalized_subject="random inquiry",
            classification_confidence=0.90,
            routing_reason="Unidentified company terminology."
        )

        with patch("src.graph.nodes.llm.generate") as mock_gen:
            mock_result = MagicMock()
            mock_result.response = mock_decision
            mock_result.usage.input_tokens = 100
            mock_result.usage.output_tokens = 50
            mock_gen.return_value = mock_result
            
            res = decision_gate_node(state)
            
            self.assertFalse(res["company_match"])
            self.assertEqual(res["verified_company"], "Visa")
            self.assertEqual(res["detected_company"], "Unknown")
            self.assertEqual(res["company_confidence"], 0.45)

    @patch("src.graph.nodes.Retriever")
    def test_retriever_always_uses_verified_company(self, mock_retriever_cls):
        """Verify that retrieve_node passes verified_company to Retriever.retrieve."""
        state = {
            "issue": "Claude Sonnet is throwing 500 timeouts when calling the Anthropic API.",
            "subject": "Claude API timeout",
            "company": "Visa",
            "verified_company": "Claude",
        }
        
        mock_retriever_inst = mock_retriever_cls.return_value
        mock_retriever_inst.retrieve.return_value = {
            "context": [],
            "retrieval_trace": {}
        }
        mock_retriever_inst.hybrid.reranker = None
        
        retrieve_node(state)
        
        mock_retriever_inst.retrieve.assert_called_once_with(
            issue="Claude Sonnet is throwing 500 timeouts when calling the Anthropic API.",
            subject="Claude API timeout",
            company="Claude"  # Must be verified_company!
        )
