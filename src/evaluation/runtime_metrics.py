"""
Runtime metrics module for Operations performance evaluation.
Computes Average Latency, Average Cost, Average Tokens, and Success Rate.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_runtime_metrics(
    results: List[Dict[str, Any]],
    total_tickets: int,
    succeeded_count: int,
    failed_count: int,
) -> Dict[str, Any]:
    """
    Compute operational runtime metrics across executed predictions.
    """
    success_results = [r for r in results if r.get("execution_status") == "SUCCESS"]

    avg_latency = 0.0
    avg_cost = 0.0
    total_cost = 0.0
    avg_tokens = 0.0

    if success_results:
        avg_latency = sum(r.get("latency", 0.0) for r in success_results) / len(success_results)
        total_cost = sum(r.get("cost", 0.0) for r in success_results)
        avg_cost = total_cost / len(success_results)
        avg_tokens = sum(r.get("tokens", 0) for r in success_results) / len(success_results)

    success_rate = (succeeded_count / total_tickets * 100.0) if total_tickets > 0 else 0.0

    return {
        "total_evaluated": total_tickets,
        "total_tickets": total_tickets,
        "succeeded_count": succeeded_count,
        "succeeded_predictions": succeeded_count,
        "failed_count": failed_count,
        "failed_predictions": failed_count,
        "total_cost": round(total_cost, 6),
        "total_latency_ms": round(avg_latency * succeeded_count, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_cost": round(avg_cost, 6),
        "avg_tokens": round(avg_tokens, 1),
        "success_rate": round(success_rate, 1),
    }
