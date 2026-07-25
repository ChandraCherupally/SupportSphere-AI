"""
Data models for SupportSphere AI evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EvaluationSummary:
    """Summary metrics of an evaluation run across Decision Engine, Generation Engine, and Operations."""

    # Operations / Runtime
    total_evaluated: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost: float = 0.0
    avg_tokens: float = 0.0
    success_rate: float = 0.0

    # Decision Engine Accuracies
    request_type_accuracy: float = 0.0
    product_area_accuracy: float = 0.0
    status_accuracy: float = 0.0

    # Generation Engine Quality (RAGAS Metrics)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    avg_ragas_score: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result container containing summary and detailed prediction records."""

    summary: EvaluationSummary
    predictions: List[Dict[str, Any]] = field(default_factory=list)
