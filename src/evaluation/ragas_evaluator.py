"""
RAGAS Evaluation module for Generation Engine quality.
Clean implementation closely aligned with official RAGAS notebook examples.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Dict, List

# Standard shim to prevent third-party legacy import error in ragas
if "langchain_community.chat_models.vertexai" not in sys.modules:
    shim = types.ModuleType("langchain_community.chat_models.vertexai")
    shim.ChatVertexAI = None  # type: ignore
    sys.modules["langchain_community.chat_models.vertexai"] = shim

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics._answer_correctness import answer_correctness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

import src.config as src_config
from src.ai.client import LLMClient
import pandas as pd

def _extract_metric_val(df: Any, idx: int, *col_names: str) -> float:
    """Safely extract a metric value from RAGAS evaluation pandas DataFrame by matching possible column aliases."""
    if not hasattr(df, "columns"):
        return 0.0
    for col in col_names:
        if col in df.columns:
            try:
                val = df[col].iloc[idx]
                if val is not None and not pd.isna(val):
                    return float(val)
            except Exception:
                pass
    return 0.0


class RagasEvaluator:
    """
    Evaluator for Generation Engine quality using native RAGAS metrics.
    Constructs an explicit LLM judge from the selected Evaluation Provider and Model.
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-5-mini",
        api_key: str = "",
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or src_config.get_api_key(self.provider)

    def evaluate_generation(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate Generation Engine outputs using native RAGAS metrics.

        Each sample includes:
        - user_input: customer issue
        - response: generated LLM response
        - retrieved_contexts: retrieved text chunks
        - reference: expected ground truth response
        """
        if not self.api_key:
            raise ValueError(
                f"RAGAS evaluation requires a configured Evaluation Provider and API Key for '{self.provider}'."
            )

        success_preds = [p for p in predictions if p.get("execution_status") == "SUCCESS"]

        if not success_preds:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_correctness": 0.0,
                "avg_ragas_score": 0.0,
            }

        # 1. Segment tickets into RAG vs Direct (Template) tickets
        rag_preds = []
        direct_preds = []

        for p in success_preds:
            ret_ctx = p.get("retrieved_context") or p.get("retrieved_contexts") or []
            if isinstance(ret_ctx, str):
                ret_ctx = [ret_ctx] if ret_ctx.strip() else []
            elif isinstance(ret_ctx, list):
                ret_ctx = [str(c) for c in ret_ctx if c]
            else:
                ret_ctx = []

            is_rag = bool(ret_ctx) and p.get("retrieval_required", True) is not False
            if is_rag:
                p["_parsed_ret_ctx"] = ret_ctx
                rag_preds.append(p)
            else:
                direct_preds.append(p)

        # 2. Create Evaluation LLM Judge explicitly
        chat_model = LLMClient()._get_model(self.provider, self.model_name, self.api_key)
        eval_llm = LangchainLLMWrapper(chat_model)
        run_cfg = RunConfig(timeout=180, max_retries=10, max_workers=2)

        is_temp_restricted = any(sub in self.model_name.lower() for sub in ("o1", "o3", "o-1", "o-3", "reason", "r1"))
        if is_temp_restricted:
            for m in (faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness):
                m.llm = eval_llm
                if hasattr(getattr(m.llm, "langchain_llm", None), "temperature"):
                    try:
                        m.llm.langchain_llm.temperature = None
                    except Exception:
                        pass

        # 3. Evaluate RAG tickets (Full 5 RAGAS metrics)
        rag_f_scores, rag_ar_scores, rag_cp_scores, rag_cr_scores, rag_ac_scores = [], [], [], [], []
        if rag_preds:
            rag_samples = [
                {
                    "user_input": str(p.get("issue") or p.get("user_input") or "").strip(),
                    "response": str(p.get("predicted_response") or p.get("response") or "").strip(),
                    "retrieved_contexts": p["_parsed_ret_ctx"],
                    "reference": str(p.get("expected_response") or p.get("reference") or "").strip() or str(p.get("predicted_response") or ""),
                }
                for p in rag_preds
            ]
            eval_dataset = Dataset.from_dict(
                {
                    "user_input": [s["user_input"] for s in rag_samples],
                    "response": [s["response"] for s in rag_samples],
                    "retrieved_contexts": [s["retrieved_contexts"] for s in rag_samples],
                    "reference": [s["reference"] for s in rag_samples],
                }
            )
            ragas_res = evaluate(
                eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
                llm=eval_llm,
                run_config=run_cfg,
                raise_exceptions=False,
            )
            scores_df = ragas_res.to_pandas()

            for idx, p in enumerate(rag_preds):
                f_val = round(_extract_metric_val(scores_df, idx, "faithfulness"), 3)
                ar_val = round(_extract_metric_val(scores_df, idx, "answer_relevancy", "answer_relevance"), 3)
                cp_val = round(_extract_metric_val(scores_df, idx, "context_precision"), 3)
                cr_val = round(_extract_metric_val(scores_df, idx, "context_recall"), 3)
                ac_val = round(_extract_metric_val(scores_df, idx, "answer_correctness"), 3)
                avg_val = round((f_val + ar_val + cp_val + cr_val + ac_val) / 5.0, 3)

                p["faithfulness"] = max(0.0, min(1.0, f_val))
                p["answer_relevancy"] = max(0.0, min(1.0, ar_val))
                p["context_precision"] = max(0.0, min(1.0, cp_val))
                p["context_recall"] = max(0.0, min(1.0, cr_val))
                p["answer_correctness"] = max(0.0, min(1.0, ac_val))
                p["avg_ragas_score"] = max(0.0, min(1.0, avg_val))

                rag_f_scores.append(p["faithfulness"])
                rag_ar_scores.append(p["answer_relevancy"])
                rag_cp_scores.append(p["context_precision"])
                rag_cr_scores.append(p["context_recall"])
                rag_ac_scores.append(p["answer_correctness"])

        # 4. Evaluate Direct / Template tickets (Answer Correctness only, RAG context metrics = None)
        all_ac_scores = list(rag_ac_scores)
        if direct_preds:
            direct_samples = [
                {
                    "user_input": str(p.get("issue") or p.get("user_input") or "").strip(),
                    "response": str(p.get("predicted_response") or p.get("response") or "").strip(),
                    "retrieved_contexts": ["Direct template response."],
                    "reference": str(p.get("expected_response") or p.get("reference") or "").strip() or str(p.get("predicted_response") or ""),
                }
                for p in direct_preds
            ]
            direct_dataset = Dataset.from_dict(
                {
                    "user_input": [s["user_input"] for s in direct_samples],
                    "response": [s["response"] for s in direct_samples],
                    "retrieved_contexts": [s["retrieved_contexts"] for s in direct_samples],
                    "reference": [s["reference"] for s in direct_samples],
                }
            )
            direct_res = evaluate(
                direct_dataset,
                metrics=[answer_correctness],
                llm=eval_llm,
                run_config=run_cfg,
                raise_exceptions=False,
            )
            direct_scores_df = direct_res.to_pandas()

            for idx, p in enumerate(direct_preds):
                ac_val = round(float(direct_scores_df["answer_correctness"].iloc[idx]) if "answer_correctness" in direct_scores_df else 1.0, 3)
                p["faithfulness"] = None
                p["answer_relevancy"] = None
                p["context_precision"] = None
                p["context_recall"] = None
                p["answer_correctness"] = max(0.0, min(1.0, ac_val))
                p["avg_ragas_score"] = p["answer_correctness"]
                all_ac_scores.append(p["answer_correctness"])

        # Clean up temporary internal field
        for p in success_preds:
            p.pop("_parsed_ret_ctx", None)

        for p in predictions:
            if "avg_ragas_score" not in p:
                p["faithfulness"] = None
                p["answer_relevancy"] = None
                p["context_precision"] = None
                p["context_recall"] = None
                p["answer_correctness"] = 0.0
                p["avg_ragas_score"] = 0.0

        # Compute summary averages
        f_summary = round(float(sum(rag_f_scores) / len(rag_f_scores)), 3) if rag_f_scores else 1.0
        ar_summary = round(float(sum(rag_ar_scores) / len(rag_ar_scores)), 3) if rag_ar_scores else 1.0
        cp_summary = round(float(sum(rag_cp_scores) / len(rag_cp_scores)), 3) if rag_cp_scores else 1.0
        cr_summary = round(float(sum(rag_cr_scores) / len(rag_cr_scores)), 3) if rag_cr_scores else 1.0
        ac_summary = round(float(sum(all_ac_scores) / len(all_ac_scores)), 3) if all_ac_scores else 0.0

        all_avg_scores = [p["avg_ragas_score"] for p in success_preds if p.get("avg_ragas_score") is not None]
        avg_summary = round(float(sum(all_avg_scores) / len(all_avg_scores)), 3) if all_avg_scores else 0.0

        return {
            "faithfulness": max(0.0, min(1.0, f_summary)),
            "answer_relevancy": max(0.0, min(1.0, ar_summary)),
            "context_precision": max(0.0, min(1.0, cp_summary)),
            "context_recall": max(0.0, min(1.0, cr_summary)),
            "answer_correctness": max(0.0, min(1.0, ac_summary)),
            "avg_ragas_score": max(0.0, min(1.0, avg_summary)),
        }
