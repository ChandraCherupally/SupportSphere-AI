import unittest

import src.config as src_config
from src.billing.calculator import BillingCalculator
from src.billing.models import TokenUsage
from src.runtime_config.runtime import RuntimeConfig


class TestModelValidation(unittest.TestCase):
    def test_strict_model_consistency(self):
        """Verify that RuntimeConfig properly exposes stage providers and models."""
        cfg = RuntimeConfig(
            decision_provider="google",
            decision_model="gemini-3.1-flash-lite",
            generation_provider="openai",
            generation_model="gpt-5.4-mini",
        )

        self.assertEqual(cfg.get_stage_provider("decision"), "google")
        self.assertEqual(cfg.get_stage_model("decision"), "gemini-3.1-flash-lite")
        self.assertEqual(cfg.get_stage_provider("generation"), "openai")
        self.assertEqual(cfg.get_stage_model("generation"), "gpt-5.4-mini")

    def test_provider_validation(self):
        """Verify that supported providers in src_config validate correctly."""
        self.assertIn("google", src_config.SUPPORTED_LLM_PROVIDERS)
        self.assertIn("openai", src_config.SUPPORTED_LLM_PROVIDERS)
        self.assertIn("groq", src_config.SUPPORTED_LLM_PROVIDERS)

    def test_billing_uses_executed_model(self):
        """Verify that BillingCalculator accurately references the configured stage models without falling back."""
        billing_calc = BillingCalculator(
            decision_provider="google",
            decision_model="gemini-3.1-flash-lite",
            generation_provider="openai",
            generation_model="gpt-5.4-mini",
            embedding_provider="google",
            embedding_model="gemini-embedding-001",
            reranker="none",
        )
        dec_usage = TokenUsage(input_tokens=1000, output_tokens=500)
        gen_usage = TokenUsage(input_tokens=2000, output_tokens=1000)

        cost = billing_calc.calculate_ticket_cost(
            decision_usage=dec_usage,
            generation_usage=gen_usage,
            embedding_tokens=0,
            retrieval_required=True,
        )

        # Expected cost calculation:
        # gemini-3.1-flash-lite: $0.25/M input, $1.50/M output -> 1000 * 0.25/1M + 500 * 1.50/1M = 0.00025 + 0.00075 = 0.0010
        # gpt-5.4-mini: $1.50/M input, $9.00/M output -> 2000 * 1.50/1M + 1000 * 9.00/1M = 0.0030 + 0.0090 = 0.0120
        # Pinecone query cost: $0.00002
        # total decision = 0.0010, generation = 0.0120, total = 0.01302
        self.assertAlmostEqual(cost.decision_cost, 0.0010, places=6)
        self.assertAlmostEqual(cost.generation_cost, 0.0120, places=6)
        self.assertAlmostEqual(cost.total_cost, 0.01302, places=6)
