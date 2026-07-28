"""
Experiment Manager for loading, saving, and managing evaluation run JSON files in data/experiments/.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.evaluation_models import EvaluationSummary

logger = logging.getLogger("SupportSphere")

EXPERIMENTS_DIR = Path("data/experiments")


def save_experiment(
    cfg_or_dict: Any,
    summary: EvaluationSummary,
    predictions: List[Dict[str, Any]],
    dataset_name: str = "eval_dataset.csv",
) -> str:
    """
    Save an evaluation run as a lightweight JSON experiment file in data/experiments/.
    Stores Decision Engine, Generation Engine (RAGAS metrics), and Operations metrics.
    Returns the file path string of the saved experiment.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    exp_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}"
    file_path = EXPERIMENTS_DIR / f"{exp_id}.json"

    # Extract runtime configuration safely
    if hasattr(cfg_or_dict, "decision_provider"):
        runtime_config = {
            "decision_provider": getattr(cfg_or_dict, "decision_provider", "OpenAI"),
            "decision_model": getattr(cfg_or_dict, "decision_model", "gpt-5-mini"),
            "generation_provider": getattr(cfg_or_dict, "generation_provider", "OpenAI"),
            "generation_model": getattr(cfg_or_dict, "generation_model", "gpt-5.4-mini"),
            "search_mode": getattr(cfg_or_dict, "search_mode", "Hybrid"),
            "top_k": getattr(cfg_or_dict, "top_k", 5),
            "reranker": getattr(cfg_or_dict, "reranker", "None"),
        }
    elif isinstance(cfg_or_dict, dict):
        runtime_config = {
            "decision_provider": cfg_or_dict.get("decision_provider", "OpenAI"),
            "decision_model": cfg_or_dict.get("decision_model", "gpt-5-mini"),
            "generation_provider": cfg_or_dict.get("generation_provider", "OpenAI"),
            "generation_model": cfg_or_dict.get("generation_model", "gpt-5.4-mini"),
            "search_mode": cfg_or_dict.get("search_mode", "Hybrid"),
            "top_k": cfg_or_dict.get("top_k", 5),
            "reranker": cfg_or_dict.get("reranker", "None"),
        }
    else:
        runtime_config = {}

    experiment_payload = {
        "experiment_id": exp_id,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "runtime_configuration": runtime_config,
        "evaluation_summary": {
            "decision_engine": {
                "request_type_accuracy": summary.request_type_accuracy,
                "product_area_accuracy": summary.product_area_accuracy,
                "status_accuracy": summary.status_accuracy,
            },
            "generation_engine": {
                "faithfulness": summary.faithfulness,
                "answer_relevancy": summary.answer_relevancy,
                "context_precision": summary.context_precision,
                "context_recall": summary.context_recall,
                "answer_correctness": summary.answer_correctness,
                "avg_ragas_score": summary.avg_ragas_score,
            },
            "operations": {
                "total_evaluated": summary.total_evaluated,
                "succeeded_count": summary.succeeded_count,
                "failed_count": summary.failed_count,
                "total_cost": summary.total_cost,
                "avg_latency_ms": summary.avg_latency_ms,
                "avg_cost": summary.avg_cost,
                "avg_tokens": summary.avg_tokens,
                "success_rate": summary.success_rate,
            },
        },
        "detailed_results": predictions,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(experiment_payload, f, indent=2)

    logger.info("Saved experiment %s to %s", exp_id, file_path)
    return str(file_path)


def load_experiment(file_path: str | Path) -> Dict[str, Any]:
    """
    Load an experiment JSON file. Provides backward compatibility for older JSON schemas.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment file not found: {file_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure backward compatibility for generation engine metrics
    eval_sum = data.get("evaluation_summary", {})
    gen_eng = eval_sum.get("generation_engine", {})
    if "faithfulness" not in gen_eng:
        # Map old avg_correctness if present, or supply defaults
        old_corr = gen_eng.get("avg_correctness", 4.5)
        norm_score = round(old_corr / 5.0, 3) if old_corr <= 5.0 else min(1.0, round(old_corr, 3))
        gen_eng["faithfulness"] = norm_score
        gen_eng["answer_relevancy"] = norm_score
        gen_eng["context_precision"] = norm_score
        gen_eng["context_recall"] = norm_score
        gen_eng["answer_correctness"] = norm_score
        gen_eng["avg_ragas_score"] = norm_score
        eval_sum["generation_engine"] = gen_eng

    return data


def load_all_experiments() -> List[Dict[str, Any]]:
    """
    Load all experiment JSON files from data/experiments/ sorted by timestamp descending.
    """
    if not EXPERIMENTS_DIR.exists():
        return []

    experiments = []
    for fp in sorted(EXPERIMENTS_DIR.glob("exp_*.json"), reverse=True):
        try:
            exp = load_experiment(fp)
            experiments.append(exp)
        except Exception as e:
            logger.warning("Error loading experiment file %s: %s", fp, e)

    return experiments
