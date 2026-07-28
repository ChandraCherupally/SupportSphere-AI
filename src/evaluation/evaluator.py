"""
Orchestration layer for SupportSphere AI evaluation framework.
Coordinates ticket predictions, Decision Engine metrics, RAGAS Generation quality, and Operations metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

import src.config as src_config
from src.ai.models import AgentResponse, SupportTicket
from src.billing.calculator import BillingCalculator
from src.billing.models import TokenUsage
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.evaluation_models import EvaluationSummary
from src.evaluation.experiment_manager import save_experiment
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.evaluation.runtime_metrics import compute_runtime_metrics
from src.evaluation.utils import extract_col, match_product_area, normalize_val
from src.graph.support_agent import SupportAgent

logger = logging.getLogger("SupportSphere")


class SupportSphereEvaluator:
    """Orchestrates evaluation runs across datasets using SupportAgent, RAGAS, and classification metrics."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.agent = SupportAgent()
        self.billing_calculator = BillingCalculator(
            decision_provider=self.config.get("decision_provider", "openai"),
            decision_model=self.config.get("decision_model", "gpt-5-mini"),
            generation_provider=self.config.get("generation_provider", "openai"),
            generation_model=self.config.get("generation_model", "gpt-5.4-mini"),
        )
        eval_provider = self.config.get("eval_provider") or self.config.get("decision_provider") or "openai"
        eval_model = self.config.get("eval_model") or self.config.get("decision_model") or "gpt-5-mini"
        self.ragas_evaluator = RagasEvaluator(
            provider=eval_provider,
            model_name=eval_model,
            api_key=src_config.get_api_key(eval_provider),
        )

    def _eval_row(self, row: Any) -> Dict[str, Any]:
        issue = extract_col(row, "issue", "Issue")
        subject = extract_col(row, "subject", "Subject")
        company = extract_col(row, "company", "Company")

        raw_req = extract_col(row, "expected_request_type", "request_type", "Request Type")
        raw_prod = extract_col(row, "expected_product_area", "product_area", "Product Area")
        raw_status = extract_col(row, "expected_status", "status", "Status")
        raw_resp = extract_col(row, "expected_response", "response", "Response", "Ground Truth Response")
        norm_req, norm_prod, norm_status = normalize_val(raw_req), normalize_val(raw_prod), normalize_val(raw_status)
        search_mode = str(self.config.get("search_mode") or getattr(src_config, "SEARCH_MODE", "hybrid")).lower()
        reranker = str(self.config.get("reranker") or getattr(src_config, "RERANKER", "none")).lower()
        ticket = SupportTicket(
            issue=issue,
            subject=subject,
            company=company,
            search_mode=search_mode,
            reranker=reranker,
        )
        t_start = time.time()

        try:
            agent_res: AgentResponse = self.agent.invoke(ticket)
            ms = (time.time() - t_start) * 1000.0
            p_req = (
                agent_res.decision_result.request_type
                if (agent_res.decision_result and getattr(agent_res.decision_result, "request_type", None))
                else (getattr(agent_res.response, "request_type", "invalid") if agent_res.response else "invalid")
            )
            p_prod = agent_res.response.product_area if agent_res.response else "Conversation_Management"
            p_status = agent_res.response.status if agent_res.response else "Replied"
            np_req, np_prod, np_status = normalize_val(p_req), normalize_val(p_prod), normalize_val(p_status)
            toks = agent_res.decision_input_tokens + agent_res.decision_output_tokens + agent_res.generation_input_tokens + agent_res.generation_output_tokens

            try:
                dec_usage = TokenUsage(
                    input_tokens=agent_res.decision_input_tokens,
                    output_tokens=agent_res.decision_output_tokens,
                )
                gen_usage = TokenUsage(
                    input_tokens=agent_res.generation_input_tokens,
                    output_tokens=agent_res.generation_output_tokens,
                )
                ticket_cost_obj = self.billing_calculator.calculate_ticket_cost(
                    decision_usage=dec_usage,
                    generation_usage=gen_usage,
                    retrieval_required=agent_res.retrieval_required,
                )
                cost = ticket_cost_obj.total_cost
            except Exception as exc:
                logger.warning("Cost calculation fallback: %s", exc)
                cost = 0.0

            return {
                "issue": issue, "subject": subject, "company": company, "expected_response": raw_resp,
                "predicted_response": agent_res.response.response if agent_res.response else "",
                "expected_request_type": raw_req, "predicted_request_type": p_req,
                "norm_expected_request_type": norm_req, "norm_predicted_request_type": np_req,
                "req_match": (norm_req == np_req) if norm_req else True,
                "expected_product_area": raw_prod, "predicted_product_area": p_prod,
                "norm_expected_product_area": norm_prod, "norm_predicted_product_area": np_prod,
                "prod_match": match_product_area(raw_prod, p_prod) if norm_prod else True,
                "expected_status": raw_status, "predicted_status": p_status,
                "norm_expected_status": norm_status, "norm_predicted_status": np_status,
                "status_match": (norm_status == np_status) if norm_status else True,
                "latency": ms, "tokens": toks, "cost": cost,
                "retrieved_context": agent_res.retrieved_context,
                "execution_status": "SUCCESS", "status": "SUCCESS",
            }
        except Exception as exc:
            ms = (time.time() - t_start) * 1000.0
            logger.error("Error evaluating ticket: %s", exc)
            fallback_req = raw_req if raw_req else "product_issue"
            return {
                "issue": issue, "subject": subject, "company": company, "expected_response": raw_resp,
                "predicted_response": f"ERROR: {exc}",
                "expected_request_type": raw_req, "predicted_request_type": fallback_req,
                "norm_expected_request_type": norm_req, "norm_predicted_request_type": normalize_val(fallback_req),
                "req_match": False, "expected_product_area": raw_prod, "predicted_product_area": "Conversation_Management",
                "norm_expected_product_area": norm_prod, "norm_predicted_product_area": "conversation_management",
                "prod_match": False, "expected_status": raw_status, "predicted_status": "Escalated",
                "norm_expected_status": norm_status, "norm_predicted_status": "escalated",
                "status_match": False, "latency": ms, "tokens": 0, "cost": 0.0,
                "retrieved_context": [], "execution_status": "FAILED", "status": "FAILED",
            }

    def run_evaluation(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Run benchmark predictions, classification metrics, RAGAS generation metrics, and operations summary."""
        results: List[Dict[str, Any]] = []
        total = len(df)

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            res = self._eval_row(row)
            results.append(res)
            if progress_callback:
                progress_callback(idx, total)

        succ_cnt = sum(1 for r in results if r.get("execution_status") == "SUCCESS")
        fail_cnt = total - succ_cnt

        class_metrics = compute_classification_metrics(results)
        ops_metrics = compute_runtime_metrics(results, total_tickets=total, succeeded_count=succ_cnt, failed_count=fail_cnt)
        ragas_metrics = self.ragas_evaluator.evaluate_generation(results)

        summary = EvaluationSummary(
            total_evaluated=ops_metrics["total_evaluated"],
            succeeded_count=ops_metrics["succeeded_count"],
            failed_count=ops_metrics["failed_count"],
            success_rate=ops_metrics["success_rate"],
            request_type_accuracy=class_metrics["request_type_accuracy"],
            product_area_accuracy=class_metrics["product_area_accuracy"],
            status_accuracy=class_metrics["status_accuracy"],
            faithfulness=ragas_metrics["faithfulness"],
            answer_relevancy=ragas_metrics["answer_relevancy"],
            context_precision=ragas_metrics["context_precision"],
            context_recall=ragas_metrics["context_recall"],
            answer_correctness=ragas_metrics["answer_correctness"],
            avg_ragas_score=ragas_metrics["avg_ragas_score"],
            avg_latency_ms=ops_metrics["avg_latency_ms"],
            avg_cost=ops_metrics["avg_cost"],
            avg_tokens=ops_metrics["avg_tokens"],
            total_cost=ops_metrics["total_cost"],
        )

        all_metrics = {
            **ops_metrics,
            **class_metrics,
            **ragas_metrics,
        }

        return {
            "summary": summary,
            "metrics": all_metrics,
            "predictions": results,
        }

    def run_benchmark(
        self,
        filepath: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[EvaluationSummary, List[Dict[str, Any]]]:
        """Convenience method running benchmark from CSV path."""
        df = pd.read_csv(filepath)
        eval_res = self.run_evaluation(df, progress_callback=progress_callback)
        return eval_res["summary"], eval_res["predictions"]


# Backward compatibility alias
SimpleEvaluator = SupportSphereEvaluator
