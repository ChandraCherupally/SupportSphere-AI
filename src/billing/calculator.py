"""
Billing calculator for SupportSphere AI.

Calculates per-ticket inference cost from token usage data and
aggregates cost metrics across a full evaluation experiment.
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

    Supports cost calculation for:
    - Decision Gate LLM call
    - Query embedding
    - Vector database (Pinecone) query
    - Reranker (FlashRank and CrossEncoder are free; LLM reranker billed)
    - Generation LLM call

    Args:
        provider: LLM provider name (e.g. "google", "openai", "groq").
        model: LLM model name.
        embedding_provider: Embedding provider name.
        embedding_model: Embedding model name.
        reranker: Active reranker type ("none", "flashrank", "cross_encoder", "llm").
    """

    def __init__(
        self,
        provider: str,
        model: str,
        embedding_provider: str = "google",
        embedding_model: str = "gemini-embedding-001",
        reranker: str = "none",
    ) -> None:
        self._provider = provider.lower()
        self._model = model.lower()
        self._embedding_provider = embedding_provider.lower()
        self._embedding_model = embedding_model.lower()
        self._reranker = reranker.lower()

        self._llm_price = get_model_price(self._provider, self._model)
        self._emb_price = get_embedding_price(
            self._embedding_provider, self._embedding_model
        )

        if self._llm_price is None:
            logger.warning(
                "No billing price found for provider=%s model=%s. "
                "LLM costs will be reported as 0.0.",
                self._provider,
                self._model,
            )
        if self._emb_price is None:
            logger.warning(
                "No billing price found for embedding provider=%s model=%s. "
                "Embedding costs will be reported as 0.0.",
                self._embedding_provider,
                self._embedding_model,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _llm_cost(self, usage: Optional[TokenUsage]) -> float:
        """Calculate LLM cost for a single call from token usage.

        Args:
            usage: Token usage from the LLM call, or None.

        Returns:
            USD cost for the call.
        """
        if usage is None or self._llm_price is None:
            return 0.0
        input_cost = (
            usage.input_tokens / _TOKENS_PER_MILLION
        ) * self._llm_price.input_per_million
        output_cost = (
            usage.output_tokens / _TOKENS_PER_MILLION
        ) * self._llm_price.output_per_million
        return input_cost + output_cost

    def _embedding_cost(self, token_count: int) -> float:
        """Calculate embedding query cost from token count.

        Args:
            token_count: Number of tokens in the embedding query.

        Returns:
            USD cost for the embedding call.
        """
        if self._emb_price is None or token_count <= 0:
            return 0.0
        return (token_count / _TOKENS_PER_MILLION) * self._emb_price.input_per_million

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_ticket_cost(
        self,
        decision_usage: Optional[TokenUsage],
        generation_usage: Optional[TokenUsage],
        embedding_tokens: int = 0,
        retrieval_required: bool = True,
    ) -> TicketCost:
        """Calculate the full runtime inference cost for one support ticket.

        Billing flow:
            Decision Gate LLM call → input + output tokens → decision_cost
            Query embedding (if retrieval_required) → embedding_cost
            Pinecone vector query (if retrieval_required) → retriever_cost
            Reranker (FlashRank/CrossEncoder = free; LLM = billed at LLM rate) → reranker_cost
            Generation LLM call → input + output tokens → generation_cost

        Args:
            decision_usage: Token usage from the Decision Gate LLM call.
            generation_usage: Token usage from the Generation LLM call.
            embedding_tokens: Token count used for the query embedding.
            retrieval_required: Whether retrieval was executed for this ticket.

        Returns:
            TicketCost with full cost breakdown and token counts.
        """
        decision_cost = self._llm_cost(decision_usage)

        embedding_cost = 0.0
        retriever_cost = 0.0
        reranker_cost = 0.0

        if retrieval_required:
            embedding_cost = self._embedding_cost(embedding_tokens)
            retriever_cost = PINECONE_QUERY_COST
            # FlashRank and CrossEncoder are local — no API cost
            if self._reranker == "llm":
                # LLM reranker billed at same LLM rate as generation
                # Approximate: same rate, usage tracked separately if available
                reranker_cost = 0.0  # token usage not separately tracked; kept at 0

        generation_cost = self._llm_cost(generation_usage)

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
        """Aggregate billing metrics across all tickets in an experiment.

        Args:
            ticket_costs: List of TicketCost objects, one per evaluated ticket.

        Returns:
            ExperimentCostSummary with totals and averages.
        """
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
