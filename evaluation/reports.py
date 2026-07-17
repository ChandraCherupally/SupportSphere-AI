"""
Report generator for exporting CSV, JSON, and TXT metrics of evaluation runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluation.models import ClassificationMetrics
from evaluation.models import EvaluationResult
from evaluation.models import EvaluationSummary

logger = logging.getLogger(__name__)


def generate_reports(
    results: List[EvaluationResult],
    summary: EvaluationSummary,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Exports evaluation run metrics to evaluation_report.csv, summary.json, and summary.txt.

    Args:
        results: List of detailed EvaluationResult objects.
        summary: Accumulated EvaluationSummary object.
        output_dir: Directory where the reports will be written.

    Returns:
        Dict mapping report type name to the written Path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating evaluation reports in directory: {output_dir.absolute()}")

    csv_path = output_dir / "evaluation_report.csv"
    json_path = output_dir / "summary.json"
    txt_path = output_dir / "summary.txt"

    # ---------------------------------------------------------
    # 1. Export Detailed CSV
    # ---------------------------------------------------------
    rows = []
    strategy_used = f"{summary.provider}_{summary.model}_{summary.search_mode}_{summary.reranker}_{summary.top_k}"
    
    for r in results:
        rows.append({
            "company": r.sample.company,
            "subject": r.sample.subject,
            "issue": r.sample.issue,
            "expected_response": r.sample.expected_response,
            "generated_response": r.generated_response,
            "strategy_used": strategy_used,
            
            # Classification details
            "expected_request_type": r.sample.expected_request_type,
            "predicted_request_type": r.prediction.predicted_request_type,
            "request_type_accuracy": r.prediction.request_type_accuracy,
            
            "expected_product_area": r.sample.expected_product_area,
            "predicted_product_area": r.prediction.predicted_product_area,
            "product_area_accuracy": r.prediction.product_area_accuracy,
            
            "expected_status": r.sample.expected_status,
            "predicted_status": r.prediction.predicted_status,
            "status_accuracy": r.prediction.status_accuracy,
            
            # Decision Gate details (NEW)
            "retrieval_required": r.retrieval_required,
            "normalized_issue": r.normalized_issue,
            "normalized_subject": r.normalized_subject,
            "routing_reason": r.routing_reason,
            "routing_confidence": r.confidence,
            
            # Latency and chunks
            "latency": r.latency,
            "num_chunks": r.num_chunks,
            
            # Ragas metrics
            "ragas_context_precision": r.ragas_metrics.context_precision,
            "ragas_context_recall": r.ragas_metrics.context_recall,
            "ragas_faithfulness": r.ragas_metrics.faithfulness,
            "ragas_answer_correctness": r.ragas_metrics.answer_correctness,
            "ragas_answer_relevancy": r.ragas_metrics.answer_relevancy,
        })

    df = pd.DataFrame(rows)
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Successfully saved detailed evaluation report to {csv_path.name}")
    except PermissionError:
        backup_path = csv_path.parent / f"{csv_path.stem}_backup.csv"
        logger.warning(f"Permission denied writing to {csv_path.name} (locked file). Saving to backup: {backup_path.name}")
        df.to_csv(backup_path, index=False)

    # ---------------------------------------------------------
    # 2. Export Summary JSON
    # ---------------------------------------------------------
    def serialize_class_metrics(m: ClassificationMetrics) -> Dict[str, Any]:
        return {
            "accuracy": m.accuracy,
            "precision_macro": m.precision_macro,
            "recall_macro": m.recall_macro,
            "f1_macro": m.f1_macro,
            "precision_weighted": m.precision_weighted,
            "recall_weighted": m.recall_weighted,
            "f1_weighted": m.f1_weighted,
            "confusion_matrix": m.confusion_matrix,
            "per_label_metrics": m.per_label_metrics,
        }

    json_dict = {
        "metadata": {
            "provider": summary.provider,
            "model": summary.model,
            "search_mode": summary.search_mode,
            "reranker": summary.reranker,
            "top_k": summary.top_k,
            "strategy_used": strategy_used,
            "total_tickets": summary.total_tickets,
            "elapsed_time_seconds": summary.elapsed_time_seconds,
            "avg_latency_seconds": summary.avg_latency,
            "avg_chunks_retrieved": summary.avg_chunks,
            "retrieval_skip_rate": summary.retrieval_skip_rate,
            "evaluation_error": summary.evaluation_error,
        },
        "classification": {
            "request_type": serialize_class_metrics(summary.request_type_metrics),
            "product_area": serialize_class_metrics(summary.product_area_metrics),
            "status": serialize_class_metrics(summary.status_metrics),
        },
        "routing": {
            "retrieval_decision_accuracy": summary.retrieval_decision_accuracy,
            "escalation_accuracy": summary.escalation_accuracy,
            "out_of_scope_accuracy": summary.out_of_scope_accuracy,
            "greeting_accuracy": summary.greeting_accuracy,
            "retrieval_skip_rate": summary.retrieval_skip_rate,
        },
        "ragas": {
            "avg_context_precision": summary.avg_context_precision,
            "avg_context_recall": summary.avg_context_recall,
            "avg_faithfulness": summary.avg_faithfulness,
            "avg_answer_correctness": summary.avg_answer_correctness,
            "avg_answer_relevancy": summary.avg_answer_relevancy,
        }
    }

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)
        logger.info(f"Successfully saved evaluation summary JSON to {json_path.name}")
    except PermissionError:
        backup_json = json_path.parent / f"{json_path.stem}_backup.json"
        logger.warning(f"Permission denied writing to {json_path.name}. Saving to backup: {backup_json.name}")
        with open(backup_json, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)

    # ---------------------------------------------------------
    # 3. Export Human-Readable TXT
    # ---------------------------------------------------------
    txt_lines = [
        "==============================================================================",
        "                     SUPPORTSPHERE AI EVALUATION REPORT                       ",
        "==============================================================================",
        f"Strategy Used:        {strategy_used}",
        f"LLM Provider:         {summary.provider.upper()}",
        f"LLM Model:            {summary.model}",
        f"Retrieval Mode:       {summary.search_mode.upper()}",
        f"Reranker Type:        {summary.reranker.title()}",
        f"Top-K Chunks:         {summary.top_k}",
        f"Total Tickets:        {summary.total_tickets}",
        f"Avg Chunks Retrieved: {summary.avg_chunks:.2f}",
        f"Avg Latency (sec):    {summary.avg_latency:.2f}s",
        f"Total Runtime (sec):  {summary.elapsed_time_seconds:.2f}s",
        "------------------------------------------------------------------------------",
        "                             CLASSIFICATION ACCURACIES                        ",
        "------------------------------------------------------------------------------",
        f"Request Type Classification Accuracy:  {summary.request_type_metrics.accuracy:.2%}",
        f"Product Area Classification Accuracy:  {summary.product_area_metrics.accuracy:.2%}",
        f"Status Triage Classification Accuracy:  {summary.status_metrics.accuracy:.2%}",
        "------------------------------------------------------------------------------",
        "                             ROUTING ACCURACIES                               ",
        "------------------------------------------------------------------------------",
        f"Retrieval Decision Accuracy:           {summary.retrieval_decision_accuracy:.2%}",
        f"Escalation Decision Accuracy:          {summary.escalation_accuracy:.2%}",
        f"Out-of-Scope Decision Accuracy:        {summary.out_of_scope_accuracy:.2%}",
        f"Greeting Decision Accuracy:            {summary.greeting_accuracy:.2%}",
        f"Retrieval Skip Rate:                   {summary.retrieval_skip_rate:.2%}",
        "------------------------------------------------------------------------------",
        "                             RAGAS METRICS PERFORMANCE                        ",
        "------------------------------------------------------------------------------",
        f"Avg Context Precision:  {f'{summary.avg_context_precision:.4f}' if summary.avg_context_precision is not None else 'N/A'}",
        f"Avg Context Recall:     {f'{summary.avg_context_recall:.4f}' if summary.avg_context_recall is not None else 'N/A'}",
        f"Avg Faithfulness:       {f'{summary.avg_faithfulness:.4f}' if summary.avg_faithfulness is not None else 'N/A'}",
        f"Avg Answer Correctness: {f'{summary.avg_answer_correctness:.4f}' if summary.avg_answer_correctness is not None else 'N/A'}",
        f"Avg Answer Relevancy:   {f'{summary.avg_answer_relevancy:.4f}' if summary.avg_answer_relevancy is not None else 'N/A'}",
        "==============================================================================",
    ]
    if summary.evaluation_error:
        txt_lines.extend([
            f"Evaluation Error Details:",
            f"  {summary.evaluation_error}",
            "==============================================================================",
        ])

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
        logger.info(f"Successfully saved evaluation summary text report to {txt_path.name}")
    except PermissionError:
        backup_txt = txt_path.parent / f"{txt_path.stem}_backup.txt"
        logger.warning(f"Permission denied writing to {txt_path.name}. Saving to backup: {backup_txt.name}")
        with open(backup_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

    return {
        "csv": csv_path,
        "json": json_path,
        "txt": txt_path,
    }
