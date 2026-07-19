"""
Module for organizing evaluation runs as historical experiments.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages historical evaluation reports, loading, and formatting comparison metrics.
    """

    def __init__(self, reports_dir: Union[str, Path] = "data/reports") -> None:
        self.reports_dir = Path(reports_dir)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        Scans registry index.json for experiments and loads their metrics.

        Returns:
            List of dictionaries containing experiment summaries and metadata.
        """
        index_path = self.reports_dir / "experiments" / "index.json"
        if not index_path.exists():
            return []
            
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry index.json: {e}")
            return []
            
        experiments = []
        for item in registry:
            exp_id = item.get("id")
            report_path_str = item.get("report_path")
            if not report_path_str:
                continue
            
            report_dir = Path(report_path_str)
            if not report_dir.is_absolute():
                # check relative to workspace root or reports parent
                report_dir = Path(report_path_str)
                
            summary_path = report_dir / "summary.json"
            if not summary_path.exists():
                summary_path = self.reports_dir / "experiments" / exp_id / "summary.json"
                report_dir = self.reports_dir / "experiments" / exp_id
                
            if not summary_path.exists():
                continue

            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                meta = data.get("metadata", {})
                ragas = data.get("ragas", {})
                req_type = data.get("classification", {}).get("request_type", {})
                prod_area = data.get("classification", {}).get("product_area", {})
                status = data.get("classification", {}).get("status", {})
                routing = data.get("routing", {})
                billing = data.get("billing", {})

                experiments.append({
                    "id": exp_id,
                    "timestamp": item.get("timestamp", "unknown"),
                    "friendly_name": item.get("friendly_name", meta.get("strategy_used", exp_id)),
                    "strategy_used": meta.get("strategy_used", exp_id),
                    "provider": item.get("provider", meta.get("provider", "unknown")),
                    "model": item.get("model", meta.get("model", "unknown")),
                    "decision_provider": item.get("decision_provider", meta.get("decision_provider", "unknown")),
                    "decision_model": item.get("decision_model", meta.get("decision_model", "unknown")),
                    "generation_provider": item.get("generation_provider", meta.get("generation_provider", "unknown")),
                    "generation_model": item.get("generation_model", meta.get("generation_model", "unknown")),
                    "search_mode": item.get("search_mode", meta.get("search_mode", "unknown")),
                    "reranker": item.get("reranker", meta.get("reranker", "unknown")),
                    "top_k": item.get("top_k", meta.get("top_k", 0)),
                    "dataset": item.get("dataset", meta.get("dataset", "unknown")),
                    "total_tickets": item.get("total_tickets", meta.get("total_tickets", 0)),
                    "elapsed_time": item.get("elapsed_time", meta.get("elapsed_time_seconds", 0.0)),
                    "avg_latency": meta.get("avg_latency_seconds", 0.0),
                    "avg_chunks": meta.get("avg_chunks_retrieved", 0.0),

                    # Accuracies
                    "request_type_accuracy": req_type.get("accuracy", 0.0),
                    "product_area_accuracy": prod_area.get("accuracy", 0.0),
                    "status_accuracy": status.get("accuracy", 0.0),

                    # Routing
                    "retrieval_decision_accuracy": routing.get("retrieval_decision_accuracy", 1.0),
                    "escalation_accuracy": routing.get("escalation_accuracy", 1.0),
                    "out_of_scope_accuracy": routing.get("out_of_scope_accuracy", 1.0),
                    "greeting_accuracy": routing.get("greeting_accuracy", 1.0),
                    "retrieval_skip_rate": routing.get("retrieval_skip_rate", 0.0),

                    # Ragas Scores
                    "context_precision": ragas.get("avg_context_precision"),
                    "context_recall": ragas.get("avg_context_recall"),
                    "faithfulness": ragas.get("avg_faithfulness"),
                    "answer_correctness": ragas.get("avg_answer_correctness"),
                    "answer_relevancy": ragas.get("avg_answer_relevancy"),

                    # Billing
                    "avg_cost_per_ticket": item.get("avg_cost_per_ticket", billing.get("avg_cost_per_ticket", 0.0)),
                    "total_cost": item.get("total_cost", billing.get("total_experiment_cost", 0.0)),
                    "avg_input_tokens": billing.get("avg_input_tokens", 0.0),
                    "avg_output_tokens": billing.get("avg_output_tokens", 0.0),
                    "avg_total_tokens": billing.get("avg_total_tokens", 0.0),
                    "pricing_version": item.get("pricing_version", "unknown"),

                    "summary_json_path": summary_path.absolute(),
                    "report_csv_path": report_dir / "evaluation_report.csv",
                    "evaluation_error": meta.get("evaluation_error"),
                })
            except Exception as e:
                logger.error(f"Failed to load experiment {exp_id} summary: {e}")

        return experiments


    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Removes an experiment from the registry index.json and deletes its files from disk.
        """
        index_path = self.reports_dir / "experiments" / "index.json"
        if not index_path.exists():
            return False
            
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry index.json: {e}")
            return False
            
        # Find entry to remove
        new_registry = []
        target_path_str = None
        for item in registry:
            if item.get("id") == experiment_id:
                target_path_str = item.get("report_path")
            else:
                new_registry.append(item)
                
        # Write back new registry
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(new_registry, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save registry index.json: {e}")
            return False
            
        # Delete directory from disk
        if target_path_str:
            target_dir = Path(target_path_str)
            if not target_dir.is_absolute():
                target_dir = Path(target_path_str)
            if not target_dir.exists():
                target_dir = self.reports_dir / "experiments" / experiment_id
                
            if target_dir.exists() and target_dir.is_dir():
                import shutil
                try:
                    shutil.rmtree(target_dir)
                    logger.info(f"Successfully deleted experiment files for {experiment_id} from {target_dir}")
                except Exception as e:
                    logger.error(f"Failed to delete files for {experiment_id} at {target_dir}: {e}")
                    return False
                    
        return True
