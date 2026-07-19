"""
Billing calculator for SupportSphere AI.
Calculates per-ticket inference cost from token usage data.
Aggregates cost metrics across a full evaluation experiment.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.billing.catalog import (
    PINECONE_QUERY_COST,
    get_embedding_price,
    get_model_price,
)
from src.billing.models import (
    ExperimentCostSummary,
    TicketCost,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_TOKENS_PER_MILLION: float = 1_000_000.0


class BillingCalculator:
    """Calculates inference cost for the SupportSphere AI pipeline.

    Supports stage-aware pricing for Decision and Generation models.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        embedding_provider: str = "google",
        embedding_model: str = "gemini-embedding-001",
        reranker: str = "none",
        decision_provider: Optional[str] = None,
        decision_model: Optional[str] = None,
        generation_provider: Optional[str] = None,
        generation_model: Optional[str] = None,
    ) -> None:
        self._embedding_provider = embedding_provider.lower()
        self._embedding_model = embedding_model.lower()
        self._reranker = reranker.lower()

        # Fallback resolves
        self._decision_provider = (decision_provider or provider or "google").lower()
        self._decision_model = (decision_model or model or "gemini-2.5-flash-lite").lower()
        self._generation_provider = (generation_provider or provider or "google").lower()
        self._generation_model = (generation_model or model or "gemini-2.5-flash").lower()

        # Load stage prices
        self._decision_price = get_model_price(self._decision_provider, self._decision_model)
        self._generation_price = get_model_price(self._generation_provider, self._generation_model)
        self._emb_price = get_embedding_price(self._embedding_provider, self._embedding_model)

        if self._decision_price is None:
            logger.warning(
                "No billing price found for decision stage: provider=%s model=%s. Costs will be 0.0.",
                self._decision_provider, self._decision_model
            )
        if self._generation_price is None:
            logger.warning(
                "No billing price found for generation stage: provider=%s model=%s. Costs will be 0.0.",
                self._generation_provider, self._generation_model
            )
        if self._emb_price is None:
            logger.warning(
                "No billing price found for embedding provider=%s model=%s. Costs will be 0.0.",
                self._embedding_provider, self._embedding_model
            )

    def _llm_cost(self, usage: Optional[TokenUsage], price: Optional[Any]) -> float:
        """Calculate LLM cost for a single stage from token usage and price details."""
        if usage is None or price is None:
            return 0.0
        input_cost = (usage.input_tokens / _TOKENS_PER_MILLION) * price.input_per_million
        output_cost = (usage.output_tokens / _TOKENS_PER_MILLION) * price.output_per_million
        return input_cost + output_cost

    def _embedding_cost(self, token_count: int) -> float:
        """Calculate embedding query cost from token count."""
        if self._emb_price is None or token_count <= 0:
            return 0.0
        return (token_count / _TOKENS_PER_MILLION) * self._emb_price.input_per_million

    def calculate_ticket_cost(
        self,
        decision_usage: Optional[TokenUsage],
        generation_usage: Optional[TokenUsage],
        embedding_tokens: int = 0,
        retrieval_required: bool = True,
    ) -> TicketCost:
        """Calculate the full runtime inference cost for one support ticket."""
        decision_cost = self._llm_cost(decision_usage, self._decision_price)

        embedding_cost = 0.0
        retriever_cost = 0.0
        reranker_cost = 0.0

        if retrieval_required:
            embedding_cost = self._embedding_cost(embedding_tokens)
            retriever_cost = PINECONE_QUERY_COST
            if self._reranker == "llm":
                reranker_cost = 0.0

        generation_cost = self._llm_cost(generation_usage, self._generation_price)

        return TicketCost(
            decision_cost=decision_cost,
            embedding_cost=embedding_cost,
            retriever_cost=retriever_cost,
            reranker_cost=reranker_cost,
            generation_cost=generation_cost,
            decision_input_tokens=decision_usage.input_tokens if decision_usage else 0,
            decision_output_tokens=decision_usage.output_tokens if decision_usage else 0,
            generation_input_tokens=generation_usage.input_tokens if generation_usage else 0,
            generation_output_tokens=generation_usage.output_tokens if generation_usage else 0,
            embedding_tokens=embedding_tokens if retrieval_required else 0,
        )

    def calculate_experiment_totals(
        self, ticket_costs: List[TicketCost]
    ) -> ExperimentCostSummary:
        """Aggregate billing metrics across all tickets in an experiment."""
        if not ticket_costs:
            return ExperimentCostSummary()

        totals = [tc.total_cost for tc in ticket_costs]
        total_cost = sum(totals)
        n = len(ticket_costs)

        avg_cost = total_cost / n
        avg_input = sum(tc.total_input_tokens for tc in ticket_costs) / n
        avg_output = sum(tc.total_output_tokens for tc in ticket_costs) / n
        avg_total = sum(tc.total_tokens for tc in ticket_costs) / n

        max_idx = totals.index(max(totals))
        min_idx = totals.index(min(totals))

        return ExperimentCostSummary(
            total_cost=round(total_cost, 8),
            avg_cost_per_ticket=round(avg_cost, 8),
            avg_input_tokens=round(avg_input, 2),
            avg_output_tokens=round(avg_output, 2),
            avg_total_tokens=round(avg_total, 2),
            most_expensive_ticket_idx=max_idx,
            cheapest_ticket_idx=min_idx,
            most_expensive_ticket_cost=round(totals[max_idx], 8),
            cheapest_ticket_cost=round(totals[min_idx], 8),
        )
