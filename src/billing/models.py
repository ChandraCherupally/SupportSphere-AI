"""
Billing data models for SupportSphere AI.

Defines immutable price structs and mutable cost accumulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token price for a chat/completion model.

    Attributes:
        input_per_million: USD cost per 1 million input tokens.
        output_per_million: USD cost per 1 million output tokens.
    """

    input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class EmbeddingPrice:
    """Per-million-token price for an embedding model.

    Attributes:
        input_per_million: USD cost per 1 million input tokens.
    """

    input_per_million: float


@dataclass
class TokenUsage:
    """Token counts returned by a single LLM call.

    Attributes:
        input_tokens: Number of prompt/input tokens consumed.
        output_tokens: Number of completion/output tokens generated.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class TicketCost:
    """Inference cost breakdown for a single support ticket.

    Attributes:
        decision_cost: Cost of the Decision Gate LLM call (USD).
        embedding_cost: Cost of query embedding (USD).
        retriever_cost: Cost of vector database query (USD).
        reranker_cost: Cost of reranker inference (USD).
        generation_cost: Cost of the RAG generation LLM call (USD).
        decision_input_tokens: Input tokens used in Decision Gate call.
        decision_output_tokens: Output tokens used in Decision Gate call.
        generation_input_tokens: Input tokens used in Generation call.
        generation_output_tokens: Output tokens used in Generation call.
        embedding_tokens: Tokens used for query embedding.
    """

    decision_cost: float = 0.0
    embedding_cost: float = 0.0
    retriever_cost: float = 0.0
    reranker_cost: float = 0.0
    generation_cost: float = 0.0

    decision_input_tokens: int = 0
    decision_output_tokens: int = 0
    generation_input_tokens: int = 0
    generation_output_tokens: int = 0
    embedding_tokens: int = 0

    @property
    def total_cost(self) -> float:
        """Total ticket runtime inference cost (USD)."""
        return (
            self.decision_cost
            + self.embedding_cost
            + self.retriever_cost
            + self.reranker_cost
            + self.generation_cost
        )

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all LLM calls."""
        return self.decision_input_tokens + self.generation_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all LLM calls."""
        return self.decision_output_tokens + self.generation_output_tokens

    @property
    def total_tokens(self) -> int:
        """Grand total tokens consumed for this ticket."""
        return self.total_input_tokens + self.total_output_tokens


@dataclass
class ExperimentCostSummary:
    """Aggregated billing metrics for a complete evaluation experiment.

    Attributes:
        total_cost: Sum of all ticket costs (USD).
        avg_cost_per_ticket: Average cost per evaluated ticket (USD).
        avg_input_tokens: Average input tokens per ticket.
        avg_output_tokens: Average output tokens per ticket.
        avg_total_tokens: Average total tokens per ticket.
        most_expensive_ticket_idx: Index (0-based) of the most expensive ticket.
        cheapest_ticket_idx: Index (0-based) of the cheapest ticket.
        most_expensive_ticket_cost: Cost of the most expensive ticket (USD).
        cheapest_ticket_cost: Cost of the cheapest ticket (USD).
    """

    total_cost: float = 0.0
    avg_cost_per_ticket: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    most_expensive_ticket_idx: int = 0
    cheapest_ticket_idx: int = 0
    most_expensive_ticket_cost: float = 0.0
    cheapest_ticket_cost: float = 0.0
