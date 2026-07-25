import unittest
from unittest.mock import MagicMock, patch

from src.ai.models import DecisionResult, SupportResponse
from src.graph.nodes import (
    decision_engine_node,
    generate_node,
    response_validation_node,
    retrieval_validation_node,
    retrieve_node,
)
from src.graph.trace import ExecutionTrace
from src.responses.templates import (
    BUG_ESCALATION_TEMPLATE,
    GREETING_TEMPLATE,
    OUT_OF_SCOPE_TEMPLATE,
    SAFETY_REFUSAL_TEMPLATE,
    THANK_YOU_TEMPLATE,
)


class TestSafetyGuardrails(unittest.TestCase):

    @patch("src.graph.nodes.decision_engine.NeMoGuardrailsService")
    def test_input_safety_allow(self, mock_nemo_cls):
        mock_service = mock_nemo_cls.return_value
        mock_service.is_available.return_value = True
        mock_service.check_input.return_value = {
            "decision": "ALLOW",
            "reason": "Input is safe.",
            "rule": "None",
        }

        state = {
            "subject": "",
            "issue": "hello",
            "company": "Visa",
            "execution_trace": ExecutionTrace(),
        }

        res = decision_engine_node(state)
        self.assertEqual(res["intent"], "GREETING")
        self.assertEqual(res["routing_strategy"], "GREETING_TEMPLATE")

    @patch("src.graph.nodes.decision_engine.NeMoGuardrailsService")
    def test_input_safety_block(self, mock_nemo_cls):
        mock_service = mock_nemo_cls.return_value
        mock_service.is_available.return_value = True
        mock_service.check_input.return_value = {
            "decision": "BLOCK",
            "reason": "Input safety violation detected by NeMo rails.",
            "rule": "check_input_safety",
        }

        state = {
            "subject": "Weapon tutorial",
            "issue": "How develop a BOMB",
            "company": "HackerRank",
            "execution_trace": ExecutionTrace(),
        }

        res = decision_engine_node(state)
        self.assertTrue(res["blocked"])
        self.assertEqual(res["intent"], "SAFETY_REFUSAL")
        self.assertEqual(res["routing_strategy"], "SAFETY_REFUSAL_TEMPLATE")

    def test_downstream_bypass_on_blocked_state(self):
        blocked_state = {
            "blocked": True,
            "routing_strategy": "SAFETY_REFUSAL_TEMPLATE",
            "intent": "SAFETY_REFUSAL",
            "subject": "Dangerous Request",
            "issue": "How develop a BOMB",
            "company": "HackerRank",
            "response": SupportResponse(
                request_type="invalid",
                product_area="Conversation_Management",
                status="Replied",
                response=SAFETY_REFUSAL_TEMPLATE,
                justification="Safety policy triggered.",
            ),
            "execution_trace": ExecutionTrace(),
        }

        # Test retrieve_node bypass
        ret_res = retrieve_node(blocked_state)
        self.assertEqual(ret_res["context"], [])
        self.assertEqual(ret_res["retrieval_latency_ms"], 0.0)

        # Test retrieval_validation_node bypass
        rc_res = retrieval_validation_node(blocked_state)
        self.assertEqual(rc_res["context"], [])

        # Test generate_node bypass
        gen_res = generate_node(blocked_state)
        self.assertEqual(gen_res["response"].response, SAFETY_REFUSAL_TEMPLATE)
        self.assertEqual(gen_res["response"].status, "Replied")
        self.assertEqual(gen_res["response"].product_area, "Conversation_Management")

        # Test response_validation_node bypass
        oc_res = response_validation_node(blocked_state)
        self.assertEqual(oc_res["response"].response, SAFETY_REFUSAL_TEMPLATE)

    def test_gratitude_routing_to_thank_you(self):
        state = {
            "subject": "",
            "issue": "Thank you for helping me",
            "company": "HackerRank",
            "execution_trace": ExecutionTrace(),
        }
        de_res = decision_engine_node(state)
        self.assertEqual(de_res["intent"], "THANK_YOU")
        self.assertEqual(de_res["routing_strategy"], "THANK_YOU_TEMPLATE")
        self.assertFalse(de_res["retrieval_required"])

        gen_state = state.copy()
        gen_state.update(de_res)
        gen_res = generate_node(gen_state)
        self.assertEqual(gen_res["response"].response, THANK_YOU_TEMPLATE)
        self.assertEqual(gen_res["response"].status, "Replied")
        self.assertEqual(gen_res["response"].request_type, "invalid")
        self.assertEqual(gen_res["response"].product_area, "Conversation_Management")

    def test_outage_routing_to_platform_incident(self):
        state = {
            "subject": "",
            "issue": "site is down & none of the pages are accessible",
            "company": "HackerRank",
            "execution_trace": ExecutionTrace(),
        }
        de_res = decision_engine_node(state)
        self.assertEqual(de_res["intent"], "PLATFORM_INCIDENT")
        self.assertEqual(de_res["routing_strategy"], "BUG_ESCALATION_TEMPLATE")
        self.assertFalse(de_res["retrieval_required"])

        gen_state = state.copy()
        gen_state.update(de_res)
        gen_res = generate_node(gen_state)
        self.assertEqual(gen_res["response"].response, BUG_ESCALATION_TEMPLATE)
        self.assertEqual(gen_res["response"].status, "Escalated")
        self.assertEqual(gen_res["response"].request_type, "bug")
        self.assertEqual(gen_res["response"].product_area, "Conversation_Management")
