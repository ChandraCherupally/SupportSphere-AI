"""
Shared utilities for SupportSphere AI Streamlit UI.

Contains:
- AppConfig dataclass used across all modules
- patch_agent_configs() — syncs sidebar selections to src.config
- load_experiment_from_paths() — reconstructs summary/results from disk
- Formatting helpers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import src.config as src_config
from evaluation.models import (
    ClassificationMetrics,
    ClassificationPrediction,
    BillingMetrics,
    EvaluationResult,
    EvaluationSample,
    EvaluationSummary,
    RagasMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# App Configuration Dataclass
# =============================================================================

@dataclass
class AppConfig:
    """Sidebar-configured runtime settings."""
    decision_provider: str = "Google"
    decision_model: str = "gemini-2.5-flash-lite"
    generation_provider: str = "Google"
    generation_model: str = "gemini-2.5-flash"
    google_key: str = ""
    openai_key: str = ""
    groq_key: str = ""
    anthropic_key: str = ""
    top_k: int = 5
    search_mode: str = "Hybrid"
    reranker: str = "None"
    keys_provided: bool = True

    # Properties for backward compatibility
    @property
    def provider(self) -> str:
        return self.generation_provider

    @property
    def selected_model(self) -> str:
        return self.generation_model


# =============================================================================
# Agent Config Patcher
# =============================================================================

def patch_agent_configs(cfg: AppConfig) -> None:
    """Sync sidebar configuration to src.config module variables.

    Args:
        cfg: The active AppConfig object produced by render_sidebar().
    """
    src_config.DECISION_PROVIDER = cfg.decision_provider.lower()
    src_config.DECISION_MODEL = cfg.decision_model
    src_config.GENERATION_PROVIDER = cfg.generation_provider.lower()
    src_config.GENERATION_MODEL = cfg.generation_model
    
    src_config.LLM_PROVIDER = cfg.generation_provider.lower()
    src_config.LLM_MODEL = cfg.generation_model

    src_config.LLM_CONFIG = {
        "decision": {
            "provider": cfg.decision_provider.lower(),
            "model": cfg.decision_model,
        },
        "generation": {
            "provider": cfg.generation_provider.lower(),
            "model": cfg.generation_model,
        }
    }

    src_config.GOOGLE_API_KEY = cfg.google_key
    src_config.GOOGLE_API_KEY_EMBED = cfg.google_key
    src_config.OPENAI_API_KEY = cfg.openai_key
    src_config.ANTHROPIC_API_KEY = cfg.anthropic_key
    src_config.GROQ_API_KEY = cfg.groq_key
    src_config.FINAL_TOP_K = cfg.top_k
    src_config.MAX_CONTEXT_CHUNKS = cfg.top_k
    src_config.BM25_TOP_K = cfg.top_k * 2
    src_config.VECTOR_TOP_K = cfg.top_k * 2
    src_config.SEARCH_MODE = cfg.search_mode.lower()


# =============================================================================
# Experiment Loader
# =============================================================================

def load_experiment_from_paths(
    summary_json_path: Any, report_csv_path: Any
) -> Tuple[EvaluationSummary, List[EvaluationResult]]:
    """Reconstruct EvaluationSummary and results list from persisted files.

    Args:
        summary_json_path: Path to summary.json artifact.
        report_csv_path: Path to evaluation_report.csv artifact.

    Returns:
        Tuple of (EvaluationSummary, List[EvaluationResult]).
    """
    import json

    with open(summary_json_path, "r", encoding="utf-8") as f:
        exp_json = json.load(f)

    meta = exp_json.get("metadata", {})
    class_dict = exp_json.get("classification", {})
    routing_dict = exp_json.get("routing", {})
    ragas_dict = exp_json.get("ragas", {})
    billing_dict = exp_json.get("billing", {})

    def _make_metrics(d: Dict[str, Any]) -> ClassificationMetrics:
        return ClassificationMetrics(
            accuracy=d.get("accuracy", 0.0),
            precision_macro=d.get("precision_macro", 0.0),
            recall_macro=d.get("recall_macro", 0.0),
            f1_macro=d.get("f1_macro", 0.0),
            precision_weighted=d.get("precision_weighted", 0.0),
            recall_weighted=d.get("recall_weighted", 0.0),
            f1_weighted=d.get("f1_weighted", 0.0),
            confusion_matrix=d.get("confusion_matrix", [[]]),
            per_label_metrics=d.get("per_label_metrics", {}),
        )

    summary = EvaluationSummary(
        total_tickets=meta.get("total_tickets", 0),
        avg_latency=meta.get("avg_latency_seconds", 0.0),
        avg_chunks=meta.get("avg_chunks_retrieved", 0.0),
        elapsed_time_seconds=meta.get("elapsed_time_seconds", 0.0),
        request_type_metrics=_make_metrics(class_dict.get("request_type", {})),
        product_area_metrics=_make_metrics(class_dict.get("product_area", {})),
        status_metrics=_make_metrics(class_dict.get("status", {})),
        retrieval_decision_accuracy=routing_dict.get("retrieval_decision_accuracy", 1.0),
        escalation_accuracy=routing_dict.get("escalation_accuracy", 1.0),
        out_of_scope_accuracy=routing_dict.get("out_of_scope_accuracy", 1.0),
        greeting_accuracy=routing_dict.get("greeting_accuracy", 1.0),
        retrieval_skip_rate=meta.get("retrieval_skip_rate", 0.0),
        avg_context_precision=ragas_dict.get("avg_context_precision"),
        avg_context_recall=ragas_dict.get("avg_context_recall"),
        avg_faithfulness=ragas_dict.get("avg_faithfulness"),
        avg_answer_correctness=ragas_dict.get("avg_answer_correctness"),
        avg_answer_relevancy=ragas_dict.get("avg_answer_relevancy"),
        provider=meta.get("provider", "unknown"),
        model=meta.get("model", "unknown"),
        search_mode=meta.get("search_mode", "unknown"),
        reranker=meta.get("reranker", "unknown"),
        top_k=meta.get("top_k", 0),
        avg_cost_per_ticket=billing_dict.get("avg_cost_per_ticket", 0.0),
        total_experiment_cost=billing_dict.get("total_experiment_cost", 0.0),
        avg_input_tokens=billing_dict.get("avg_input_tokens", 0.0),
        avg_output_tokens=billing_dict.get("avg_output_tokens", 0.0),
        avg_total_tokens=billing_dict.get("avg_total_tokens", 0.0),
    )

    df_csv = pd.read_csv(report_csv_path)
    results: List[EvaluationResult] = []
    for _, row in df_csv.iterrows():
        sample = EvaluationSample(
            issue=str(row.get("issue")),
            subject=str(row.get("subject", "")),
            company=str(row.get("company", "None")),
            expected_response=str(row.get("expected_response")),
            expected_request_type=str(row.get("expected_request_type")),
            expected_product_area=str(row.get("expected_product_area")),
            expected_status=str(row.get("expected_status")),
        )
        pred = ClassificationPrediction(
            predicted_request_type=str(row.get("predicted_request_type")),
            predicted_product_area=str(row.get("predicted_product_area")),
            predicted_status=str(row.get("predicted_status")),
            request_type_accuracy=int(row.get("request_type_accuracy", 0)),
            product_area_accuracy=int(row.get("product_area_accuracy", 0)),
            status_accuracy=int(row.get("status_accuracy", 0)),
        )
        ragas = RagasMetrics(
            context_precision=float(row.get("ragas_context_precision")) if pd.notna(row.get("ragas_context_precision")) else None,
            context_recall=float(row.get("ragas_context_recall")) if pd.notna(row.get("ragas_context_recall")) else None,
            faithfulness=float(row.get("ragas_faithfulness")) if pd.notna(row.get("ragas_faithfulness")) else None,
            answer_correctness=float(row.get("ragas_answer_correctness")) if pd.notna(row.get("ragas_answer_correctness")) else None,
            answer_relevancy=float(row.get("ragas_answer_relevancy")) if pd.notna(row.get("ragas_answer_relevancy")) else None,
        )
        billing = BillingMetrics(
            total_cost=float(row.get("billing_total_cost", 0.0)) if pd.notna(row.get("billing_total_cost", 0.0)) else 0.0,
            decision_cost=float(row.get("billing_decision_cost", 0.0)) if pd.notna(row.get("billing_decision_cost", 0.0)) else 0.0,
            generation_cost=float(row.get("billing_generation_cost", 0.0)) if pd.notna(row.get("billing_generation_cost", 0.0)) else 0.0,
            total_tokens=int(row.get("billing_total_tokens", 0)) if pd.notna(row.get("billing_total_tokens", 0)) else 0,
        )
        results.append(
            EvaluationResult(
                sample=sample,
                generated_response=str(row.get("generated_response")),
                retrieved_contexts=[],
                prediction=pred,
                ragas_metrics=ragas,
                latency=float(row.get("latency", 0.0)),
                num_chunks=int(row.get("num_chunks", 0)),
                retrieval_required=bool(row.get("retrieval_required", True)) if "retrieval_required" in row else True,
                normalized_issue=str(row.get("normalized_issue", "")),
                normalized_subject=str(row.get("normalized_subject", "")),
                routing_reason=str(row.get("routing_reason", "")),
                confidence=float(row.get("routing_confidence", 1.0)),
                billing=billing,
            )
        )

    return summary, results


# =============================================================================
# Formatting Helpers
# =============================================================================

def fmt_ragas_metric(val: Optional[float]) -> str:
    """Format an optional float to 4dp, or 'N/A'.

    Args:
        val: Optional float metric.

    Returns:
        Formatted string.
    """
    return f"{val:.4f}" if val is not None else "N/A"


def fmt_cost(val: float) -> str:
    """Format a USD cost value for display.

    Args:
        val: Cost in USD.

    Returns:
        Formatted string like '$0.000123'.
    """
    if val == 0.0:
        return "$0.00"
    if val < 0.001:
        return f"${val:.6f}"
    return f"${val:.4f}"


def get_improvement_row(
    metric_name: str, val_a: Optional[float], val_b: Optional[float]
) -> Dict[str, str]:
    """Build an improvement comparison row dict.

    Args:
        metric_name: Display name for the metric.
        val_a: Previous/baseline value.
        val_b: Current/optimized value.

    Returns:
        Dict with Metric, Previous, Current, Improvement keys.
    """
    if val_a is None or val_b is None:
        return {"Metric": metric_name, "Previous": "N/A", "Current": "N/A", "Improvement": "N/A"}
    diff = val_b - val_a
    pct = (diff / val_a * 100) if val_a else 0.0
    if diff > 0.0001:
        arrow = f"🟢 +{pct:.1f}% ⬆️"
    elif diff < -0.0001:
        arrow = f"🔴 {pct:.1f}% ⬇️"
    else:
        arrow = "⚪ 0.0%"
    return {
        "Metric": metric_name,
        "Previous": f"{val_a:.4f}",
        "Current": f"{val_b:.4f}",
        "Improvement": arrow,
    }
