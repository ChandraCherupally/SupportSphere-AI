import unittest
from unittest.mock import patch, MagicMock
import src.config as src_config
from src.ai.registry import LLMRegistry
from src.billing.calculator import BillingCalculator
from src.billing.models import TokenUsage

class TestModelValidation(unittest.TestCase):
    def setUp(self):
        # Save previous config
        self.old_config = getattr(src_config, "LLM_CONFIG", {}).copy()

    def tearDown(self):
        src_config.LLM_CONFIG = self.old_config

    def test_strict_model_consistency(self):
        """Verify that the selected models in config are exactly what get_stage_config returns, without fallbacks."""
        src_config.LLM_CONFIG = {
            "decision": {
                "provider": "google",
                "model": "gemini-3.1-flash-lite",
            },
            "generation": {
                "provider": "openai",
                "model": "gpt-5.4-mini",
            }
        }
        dec_cfg = LLMRegistry.get_stage_config("decision")
        gen_cfg = LLMRegistry.get_stage_config("generation")

        self.assertEqual(dec_cfg["model"], "gemini-3.1-flash-lite")
        self.assertEqual(dec_cfg["provider"], "google")
        self.assertEqual(gen_cfg["model"], "gpt-5.4-mini")
        self.assertEqual(gen_cfg["provider"], "openai")

    def test_unsupported_model_raises_error(self):
        """Verify that selecting an unsupported model produces a validation error."""
        src_config.LLM_CONFIG = {
            "decision": {
                "provider": "google",
                "model": "unsupported-model-xyz",
            },
            "generation": {
                "provider": "openai",
                "model": "gpt-5.4-mini",
            }
        }
        with self.assertRaises(ValueError) as context:
            LLMRegistry.get_stage_config("decision")
        self.assertIn("is not available", str(context.exception))

    def test_billing_uses_executed_model(self):
        """Verify that BillingCalculator accurately references the configured stage models without falling back."""
        billing_calc = BillingCalculator(
            decision_provider="google",
            decision_model="gemini-3.1-flash-lite",
            generation_provider="openai",
            generation_model="gpt-5.4-mini",
            embedding_provider="google",
            embedding_model="gemini-embedding-001",
            reranker="none"
        )
        dec_usage = TokenUsage(input_tokens=1000, output_tokens=500)
        gen_usage = TokenUsage(input_tokens=2000, output_tokens=1000)

        cost = billing_calc.calculate_ticket_cost(
            decision_usage=dec_usage,
            generation_usage=gen_usage,
            embedding_tokens=0,
            retrieval_required=True
        )

        # Expected cost calculation:
        # gemini-3.1-flash-lite: $0.10/M input, $0.40/M output -> 1000 * 0.10/1M + 500 * 0.40/1M = 0.0001 + 0.0002 = 0.0003
        # gpt-5.4-mini: $0.75/M input, $4.50/M output -> 2000 * 0.75/1M + 1000 * 4.50/1M = 0.0015 + 0.0045 = 0.0060
        # Pinecone query cost: $0.00002
        # total decision = 0.0003, generation = 0.0060, total = 0.00632
        self.assertAlmostEqual(cost.decision_cost, 0.0003, places=6)
        self.assertAlmostEqual(cost.generation_cost, 0.0060, places=6)
        self.assertAlmostEqual(cost.total_cost, 0.00632, places=6)
