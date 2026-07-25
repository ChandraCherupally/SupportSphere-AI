"""
Evaluation & Experiments UI module.
Tab 1: Evaluation Summary - Production evaluation workflow (Benchmark Dataset -> Evaluation Config -> Run Evaluation -> Decision & RAGAS Summary -> Detailed Results -> Export CSV -> Save Experiment).
Tab 2: Experiments - Lightweight Experiment Management & Zero-Token Comparison tool.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

import src.config as src_config
from src.billing.catalog import get_supported_models
from src.evaluation.evaluation_models import EvaluationSummary
from src.evaluation.evaluator import SupportSphereEvaluator
from src.evaluation.experiment_manager import load_all_experiments, save_experiment
from src.streamlit.utils import AppConfig, patch_agent_configs

logger = logging.getLogger(__name__)

DEMO_EVAL_PATH = Path("data/eval_dataset.csv")
EXPERIMENTS_DIR = Path("data/experiments")


def _get_active_benchmark_dataset(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None,
) -> tuple[pd.DataFrame | None, str | None, bool]:
    """
    Resolve active benchmark dataset based on priority:
    1. Uploaded dataset
    2. Demo benchmark dataset (data/eval_dataset.csv)
    3. None (silently fallback)
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, uploaded_file.name, False
        except Exception as e:
            logger.error("Failed to parse uploaded benchmark file: %s", e)
            st.error(f"Error loading uploaded dataset: {e}")
            return None, None, False

    if DEMO_EVAL_PATH.exists():
        try:
            df = pd.read_csv(DEMO_EVAL_PATH)
            return df, DEMO_EVAL_PATH.name, True
        except Exception as e:
            logger.warning("Silent fallback: Failed to load demo eval dataset: %s", e)
            return None, None, False

def _format_ragas_metric(val: Any) -> str:
    """Format RAGAS metric value into 3-decimal float string or N/A for template tickets."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        return f"{float(val):.3f}"
    except (ValueError, TypeError):
        return "N/A"


def render_evaluation(cfg: AppConfig) -> None:
    """Render the Evaluation & Experiments Engine."""
    st.header("📊 Evaluation Engine")
    st.caption("Phase 1 System Evaluation validating Decision Engine accuracy, RAGAS Generation quality, Operations, and Experiment Tracking.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Evaluation Summary", "🧪 Experiments"])

    with tab1:
        _render_evaluation_summary_tab(cfg)

    with tab2:
        _render_experiments_tab()


def _render_evaluation_summary_tab(cfg: AppConfig) -> None:
    """Tab 1: Evaluation Summary workflow."""
    # -------------------------------------------------------------------------
    # 1. Benchmark Dataset & Compact Preview
    # -------------------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload Benchmark Dataset", type=["csv"], help="Upload custom ground-truth benchmark CSV or use preloaded system dataset.")

    bench_df, dataset_name, is_demo = _get_active_benchmark_dataset(uploaded_file)
    dataset_exists = bench_df is not None

    if dataset_exists and bench_df is not None:
        demo_tag = " (Demo Dataset)" if is_demo else ""
        st.success(f"Loaded {len(bench_df)} benchmark tickets from `{dataset_name}`{demo_tag}.")

        # Compact Dataset Preview (Issue, Subject, Company, Ground Truth Response, Product Area, Status, Request Type)
        col_map = {}
        for c in bench_df.columns:
            c_norm = c.strip().lower().replace(" ", "_").replace("-", "_")
            if c_norm == "issue":
                col_map["Issue"] = bench_df[c]
            elif c_norm == "subject":
                col_map["Subject"] = bench_df[c]
            elif c_norm == "company":
                col_map["Company"] = bench_df[c]
            elif c_norm in ("response", "expected_response", "ground_truth_response"):
                col_map["Ground Truth Response"] = bench_df[c]
            elif c_norm in ("product_area", "expected_product_area"):
                col_map["Product Area"] = bench_df[c]
            elif c_norm in ("status", "expected_status"):
                col_map["Status"] = bench_df[c]
            elif c_norm in ("request_type", "expected_request_type"):
                col_map["Request Type"] = bench_df[c]

        ordered_cols = ["Issue", "Subject", "Company", "Ground Truth Response", "Product Area", "Status", "Request Type"]
        compact_preview = pd.DataFrame({col: col_map[col] for col in ordered_cols if col in col_map})

        st.caption("🔍 **Benchmark Dataset Preview**")
        st.dataframe(compact_preview.head(5), width="stretch")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Evaluation Configuration (Provider & Model Dropdowns)
    # -------------------------------------------------------------------------
    st.markdown("#### ⚙️ Evaluation Configuration")
    st.caption("Select the LLM judge provider and model used exclusively for RAGAS evaluation.")

    model_catalog = get_supported_models()
    provider_list = list(model_catalog.keys())

    col_ep1, col_ep2 = st.columns(2)

    with col_ep1:
        default_p_idx = provider_list.index("OpenAI") if "OpenAI" in provider_list else 0
        eval_provider = st.selectbox(
            "Evaluation Provider",
            provider_list,
            index=default_p_idx,
            help="Select LLM provider for RAGAS evaluation judge.",
            key="eval_provider_select",
        )

    available_eval_models = model_catalog.get(eval_provider, ["gpt-5-mini"])

    with col_ep2:
        eval_model = st.selectbox(
            "Evaluation Model",
            available_eval_models,
            index=0,
            help="Select LLM model for RAGAS evaluation judge.",
            key="eval_model_select",
        )

    eval_api_key = src_config.get_api_key(eval_provider)
    has_eval_key = bool(eval_api_key)

    if not has_eval_key:
        st.warning(f"⚠️ RAGAS evaluation requires a configured Evaluation Provider and API Key for '{eval_provider}'.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. Run Evaluation Execution
    # -------------------------------------------------------------------------
    can_run = dataset_exists and has_eval_key

    if st.button("🚀 Run Evaluation", width="stretch", disabled=not can_run):
        patch_agent_configs(cfg)
        p_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(current: int, total: int) -> None:
            p_bar.progress(current / total)
            status_text.markdown(f"**Evaluating benchmark ticket {current}/{total}**...")

        try:
            evaluator = SupportSphereEvaluator(config={
                "decision_provider": cfg.decision_provider,
                "decision_model": cfg.decision_model,
                "generation_provider": cfg.generation_provider,
                "generation_model": cfg.generation_model,
                "eval_provider": eval_provider,
                "eval_model": eval_model,
                "search_mode": cfg.search_mode,
                "top_k": cfg.top_k,
                "reranker": cfg.reranker,
            })

            eval_res = evaluator.run_evaluation(bench_df, progress_callback=update_progress)
            summary_obj = eval_res["summary"]
            results = eval_res["predictions"]

            p_bar.progress(1.0)
            status_text.success("Finished evaluating benchmark dataset!")

            exp_path = save_experiment(cfg, summary_obj, results, dataset_name=dataset_name or "eval_dataset.csv")

            # Store in session state for persistence across reruns
            st.session_state["eval_summary"] = summary_obj
            st.session_state["eval_predictions"] = results
            st.session_state["last_experiment_path"] = exp_path

        except Exception as e:
            st.error(f"Evaluation benchmark failed: {e}")
            logger.error("Evaluation benchmark error: %s", e, exc_info=True)

    # -------------------------------------------------------------------------
    # 4. Evaluation Summary (Grouped by Architecture)
    # -------------------------------------------------------------------------
    summary_obj: EvaluationSummary | None = st.session_state.get("eval_summary")
    predictions: list[dict] | None = st.session_state.get("eval_predictions")
    last_exp_path: str | None = st.session_state.get("last_experiment_path")

    if summary_obj is not None and predictions is not None:
        st.markdown("---")
        st.subheader("📊 Evaluation Summary")

        # --- Decision Engine ---
        st.markdown("#### 🎯 Decision Engine")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Request Type Accuracy", f"{summary_obj.request_type_accuracy:.1%}")
        with col_d2:
            st.metric("Product Area Accuracy", f"{summary_obj.product_area_accuracy:.1%}")
        with col_d3:
            st.metric("Status Accuracy", f"{summary_obj.status_accuracy:.1%}")

        # --- Generation Engine Quality (RAGAS Metrics) ---
        st.markdown("#### 🤖 Generation Engine (RAGAS Quality)")
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.metric("Faithfulness", f"{summary_obj.faithfulness:.3f}")
            st.metric("Context Precision", f"{summary_obj.context_precision:.3f}")
        with col_g2:
            st.metric("Answer Relevancy", f"{summary_obj.answer_relevancy:.3f}")
            st.metric("Context Recall", f"{summary_obj.context_recall:.3f}")
        with col_g3:
            st.metric("Answer Correctness", f"{summary_obj.answer_correctness:.3f}")
            st.metric("Average RAGAS Score", f"{summary_obj.avg_ragas_score:.3f}")

        # --- Operations ---
        st.markdown("#### ⚙️ Operations")
        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        with col_o1:
            succ_pct = summary_obj.success_rate if summary_obj.success_rate > 0 else (
                (summary_obj.succeeded_count / summary_obj.total_evaluated * 100.0) if summary_obj.total_evaluated else 0.0
            )
            st.metric(
                "Execution Status (Pass / Fail)",
                f"{summary_obj.succeeded_count} / {summary_obj.failed_count}",
                delta=f"{succ_pct:.1f}% Pass Rate",
            )
        with col_o2:
            st.metric("Average Latency", f"{summary_obj.avg_latency_ms:.1f} ms")
        with col_o3:
            st.metric("Average Tokens", f"{int(summary_obj.avg_tokens):,}")
        with col_o4:
            st.metric("Average Cost", f"${summary_obj.avg_cost:.6f}")

        # -------------------------------------------------------------------------
        # 5. Detailed Results
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📋 Detailed Results")

        df_display = pd.DataFrame(
            [
                {
                    # Business Output
                    "Issue": r.get("issue", ""),
                    "Subject": r.get("subject", ""),
                    "Company": r.get("company", ""),
                    "Ground Truth Response": r.get("expected_response", ""),
                    "Predicted Response": r.get("predicted_response", ""),
                    # Technical Metadata - Decision Engine
                    "Expected Request Type": r.get("expected_request_type", ""),
                    "Predicted Request Type": r.get("predicted_request_type", ""),
                    "Request Type Match": "✅ Match" if r.get("req_match") else "❌ Mismatch",
                    "Expected Product Area": r.get("expected_product_area", ""),
                    "Predicted Product Area": r.get("predicted_product_area", ""),
                    "Product Area Match": "✅ Match" if r.get("prod_match") else "❌ Mismatch",
                    "Expected Status": r.get("expected_status", ""),
                    "Predicted Status": r.get("predicted_status", ""),
                    "Status Match": "✅ Match" if r.get("status_match") else "❌ Mismatch",
                    # Technical Metadata - RAGAS Quality Metrics
                    "Faithfulness": _format_ragas_metric(r.get("faithfulness")),
                    "Answer Relevancy": _format_ragas_metric(r.get("answer_relevancy")),
                    "Context Precision": _format_ragas_metric(r.get("context_precision")),
                    "Context Recall": _format_ragas_metric(r.get("context_recall")),
                    "Answer Correctness": _format_ragas_metric(r.get("answer_correctness")),
                    "Avg RAGAS Score": _format_ragas_metric(r.get("avg_ragas_score")),
                    # Technical Metadata - Operations
                    "Latency (ms)": round(r.get("latency", 0.0), 1),
                    "Tokens": r.get("tokens", 0),
                    "Cost ($)": round(r.get("cost", 0.0), 6),
                    "Execution Status": r.get("execution_status", "SUCCESS"),
                }
                for r in predictions
            ]
        )

        st.dataframe(df_display, width="stretch")

        # Lowest Score Ticket Analysis Assistant
        with st.expander("🔍 **Analyze Lowest Scoring Tickets by Metric**", expanded=False):
            st.caption("Sort tickets by any RAGAS metric to pinpoint low-performing queries (N/A template tickets placed at bottom).")
            metric_sort = st.selectbox(
                "Sort Tickets By (Ascending)",
                [
                    "Avg RAGAS Score",
                    "Faithfulness",
                    "Answer Relevancy",
                    "Context Precision",
                    "Context Recall",
                    "Answer Correctness",
                    "Latency (ms)",
                    "Cost ($)",
                ],
                index=0,
                key="lowest_score_sort_select",
            )
            if metric_sort in df_display.columns:
                df_sorted = df_display.copy()
                df_sorted["_sort_val"] = pd.to_numeric(df_sorted[metric_sort], errors="coerce")
                df_sorted = df_sorted.sort_values(by="_sort_val", ascending=True, na_position="last").drop(columns=["_sort_val"])
                st.dataframe(df_sorted, width="stretch")

        # -------------------------------------------------------------------------
        # 6. Download Evaluation CSV
        # -------------------------------------------------------------------------
        st.markdown("---")
        csv_bytes = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Evaluation CSV",
            data=csv_bytes,
            file_name="supportsphere_evaluation_results.csv",
            mime="text/csv",
            width="stretch",
        )

        # -------------------------------------------------------------------------
        # 7. Save Experiment Notification
        # -------------------------------------------------------------------------
        if last_exp_path:
            st.markdown("---")
            st.success(f"💾 **Experiment Saved**: `{last_exp_path}`")


def _get_config_fingerprint(exp_data: Dict[str, Any]) -> str:
    """Generate a compact readable fingerprint for an experiment configuration."""
    cfg = exp_data.get("runtime_configuration", {})
    dec_m = str(cfg.get("decision_model", "gpt-5-mini")).replace("gpt-", "").replace("-mini", "m")
    gen_m = str(cfg.get("generation_model", "gpt-5.4-mini")).replace("gpt-", "").replace("-mini", "m")
    sm = str(cfg.get("search_mode", "Hybrid"))[:3].upper()
    top_k = f"T{cfg.get('top_k', 5)}"
    return f"{dec_m}-{gen_m}-{sm}-{top_k}"


def _render_experiments_tab() -> None:
    """Tab 2: Experiments Management & Zero-Token Comparison."""
    st.subheader("🧪 Experiment Management & Comparison")
    st.caption("Compare evaluation snapshots and runtime configurations without re-running predictions or consuming LLM tokens.")

    experiments = load_all_experiments()

    if not experiments:
        st.info("ℹ️ No saved experiments found in `data/experiments/`. Run an evaluation benchmark in the **Evaluation Summary** tab to generate your first experiment snapshot.")
        return

    st.markdown(f"**Saved Experiments Available:** `{len(experiments)}`")

    exp_options = {}
    for exp in experiments:
        exp_id = exp.get("experiment_id", "exp_unknown")
        ts = exp.get("timestamp", "")
        fp = _get_config_fingerprint(exp)
        label = f"{exp_id} | {ts} | {fp}"
        exp_options[label] = exp

    option_labels = list(exp_options.keys())

    st.markdown("---")
    st.markdown("#### 🔬 Select Experiments for Comparison")
    col_sel1, col_sel2 = st.columns(2)

    with col_sel1:
        idx_a = 1 if len(option_labels) > 1 else 0
        label_a = st.selectbox("Experiment A (Baseline / Reference)", option_labels, index=idx_a)
        exp_a = exp_options[label_a]

    with col_sel2:
        idx_b = 0
        label_b = st.selectbox("Experiment B (New Candidate)", option_labels, index=idx_b)
        exp_b = exp_options[label_b]

    # Configuration Summary Cards
    st.markdown("---")
    col_ca, col_cb = st.columns(2)

    with col_ca:
        with st.container(border=True):
            cfg_a = exp_a.get("runtime_configuration", {})
            st.markdown(f"##### 🧪 **Experiment A**: `{exp_a.get('experiment_id')}`")
            st.caption(f"📅 Timestamp: {exp_a.get('timestamp', 'N/A')}")
            st.markdown(f"• **Decision Engine:** `{cfg_a.get('decision_provider', 'OpenAI')} / {cfg_a.get('decision_model', 'N/A')}`")
            st.markdown(f"• **Generation Engine:** `{cfg_a.get('generation_provider', 'OpenAI')} / {cfg_a.get('generation_model', 'N/A')}`")
            st.markdown(f"• **Retrieval:** `{cfg_a.get('search_mode', 'Hybrid')} | Top-{cfg_a.get('top_k', 5)} | Reranker: {cfg_a.get('reranker', 'None')}`")
            st.caption(f"Fingerprint: `{_get_config_fingerprint(exp_a)}`")

    with col_cb:
        with st.container(border=True):
            cfg_b = exp_b.get("runtime_configuration", {})
            st.markdown(f"##### 🧪 **Experiment B**: `{exp_b.get('experiment_id')}`")
            st.caption(f"📅 Timestamp: {exp_b.get('timestamp', 'N/A')}")
            st.markdown(f"• **Decision Engine:** `{cfg_b.get('decision_provider', 'OpenAI')} / {cfg_b.get('decision_model', 'N/A')}`")
            st.markdown(f"• **Generation Engine:** `{cfg_b.get('generation_provider', 'OpenAI')} / {cfg_b.get('generation_model', 'N/A')}`")
            st.markdown(f"• **Retrieval:** `{cfg_b.get('search_mode', 'Hybrid')} | Top-{cfg_b.get('top_k', 5)} | Reranker: {cfg_b.get('reranker', 'None')}`")
            st.caption(f"Fingerprint: `{_get_config_fingerprint(exp_b)}`")

    # Comparison Table
    st.markdown("---")
    st.markdown("#### ⚖️ Experiment Comparison Summary")

    sum_a = exp_a.get("evaluation_summary", {})
    sum_b = exp_b.get("evaluation_summary", {})

    dec_a = sum_a.get("decision_engine", {})
    dec_b = sum_b.get("decision_engine", {})

    gen_a = sum_a.get("generation_engine", {})
    gen_b = sum_b.get("generation_engine", {})

    ops_a = sum_a.get("operations", {})
    ops_b = sum_b.get("operations", {})

    # Helper functions for metric comparison visual badges
    def diff_higher_pct(v_b: float, v_a: float) -> str:
        d = (v_b - v_a) * 100
        if abs(d) < 0.001:
            return "🟠 ─ 0.0%"
        elif d > 0:
            return f"🟢 ▲ +{d:.1f}%"
        else:
            return f"🔴 ▼ {d:.1f}%"

    def diff_higher_val(v_b: float, v_a: float) -> str:
        d = v_b - v_a
        if abs(d) < 0.001:
            return "🟠 ─ 0.000"
        elif d > 0:
            return f"🟢 ▲ +{d:.3f}"
        else:
            return f"🔴 ▼ {d:.3f}"

    def diff_lower_ms(v_b: float, v_a: float) -> str:
        d = v_b - v_a
        if abs(d) < 0.01:
            return "🟠 ─ 0.0 ms"
        elif d < 0:
            return f"🟢 ▼ {d:.1f} ms"
        else:
            return f"🔴 ▲ +{d:.1f} ms"

    def diff_lower_cost(v_b: float, v_a: float) -> str:
        d = v_b - v_a
        if abs(d) < 1e-7:
            return "🟠 ─ $0.000000"
        elif d < 0:
            return f"🟢 ▼ -${abs(d):.6f}"
        else:
            return f"🔴 ▲ +${d:.6f}"

    def diff_lower_tokens(v_b: float, v_a: float) -> str:
        d = int(v_b) - int(v_a)
        if d == 0:
            return "🟠 ─ 0"
        elif d < 0:
            return f"🟢 ▼ {d}"
        else:
            return f"🔴 ▲ +{d}"

    comp_rows = [
        {
            "Metric": "Request Type Accuracy",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{dec_a.get('request_type_accuracy', 0.0):.1%}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{dec_b.get('request_type_accuracy', 0.0):.1%}",
            "Difference": diff_higher_pct(dec_b.get("request_type_accuracy", 0.0), dec_a.get("request_type_accuracy", 0.0)),
        },
        {
            "Metric": "Product Area Accuracy",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{dec_a.get('product_area_accuracy', 0.0):.1%}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{dec_b.get('product_area_accuracy', 0.0):.1%}",
            "Difference": diff_higher_pct(dec_b.get("product_area_accuracy", 0.0), dec_a.get("product_area_accuracy", 0.0)),
        },
        {
            "Metric": "Status Accuracy",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{dec_a.get('status_accuracy', 0.0):.1%}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{dec_b.get('status_accuracy', 0.0):.1%}",
            "Difference": diff_higher_pct(dec_b.get("status_accuracy", 0.0), dec_a.get("status_accuracy", 0.0)),
        },
        {
            "Metric": "Faithfulness (RAGAS)",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{gen_a.get('faithfulness', 0.0):.3f}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{gen_b.get('faithfulness', 0.0):.3f}",
            "Difference": diff_higher_val(gen_b.get("faithfulness", 0.0), gen_a.get("faithfulness", 0.0)),
        },
        {
            "Metric": "Answer Relevancy (RAGAS)",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{gen_a.get('answer_relevancy', 0.0):.3f}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{gen_b.get('answer_relevancy', 0.0):.3f}",
            "Difference": diff_higher_val(gen_b.get("answer_relevancy", 0.0), gen_a.get("answer_relevancy", 0.0)),
        },
        {
            "Metric": "Context Precision (RAGAS)",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{gen_a.get('context_precision', 0.0):.3f}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{gen_b.get('context_precision', 0.0):.3f}",
            "Difference": diff_higher_val(gen_b.get("context_precision", 0.0), gen_a.get("context_precision", 0.0)),
        },
        {
            "Metric": "Context Recall (RAGAS)",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{gen_a.get('context_recall', 0.0):.3f}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{gen_b.get('context_recall', 0.0):.3f}",
            "Difference": diff_higher_val(gen_b.get("context_recall", 0.0), gen_a.get("context_recall", 0.0)),
        },
        {
            "Metric": "Answer Correctness (RAGAS)",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{gen_a.get('answer_correctness', 0.0):.3f}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{gen_b.get('answer_correctness', 0.0):.3f}",
            "Difference": diff_higher_val(gen_b.get("answer_correctness", 0.0), gen_a.get("answer_correctness", 0.0)),
        },
        {
            "Metric": "Average RAGAS Score",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{gen_a.get('avg_ragas_score', 0.0):.3f}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{gen_b.get('avg_ragas_score', 0.0):.3f}",
            "Difference": diff_higher_val(gen_b.get("avg_ragas_score", 0.0), gen_a.get("avg_ragas_score", 0.0)),
        },
        {
            "Metric": "Average Latency",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{ops_a.get('avg_latency_ms', 0.0):.1f} ms",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{ops_b.get('avg_latency_ms', 0.0):.1f} ms",
            "Difference": diff_lower_ms(ops_b.get("avg_latency_ms", 0.0), ops_a.get("avg_latency_ms", 0.0)),
        },
        {
            "Metric": "Average Cost",
            f"Experiment A ({exp_a.get('experiment_id')})": f"${ops_a.get('avg_cost', 0.0):.6f}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"${ops_b.get('avg_cost', 0.0):.6f}",
            "Difference": diff_lower_cost(ops_b.get("avg_cost", 0.0), ops_a.get("avg_cost", 0.0)),
        },
        {
            "Metric": "Average Tokens",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{int(ops_a.get('avg_tokens', 0))}",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{int(ops_b.get('avg_tokens', 0))}",
            "Difference": diff_lower_tokens(ops_b.get("avg_tokens", 0), ops_a.get("avg_tokens", 0)),
        },
        {
            "Metric": "Success Rate",
            f"Experiment A ({exp_a.get('experiment_id')})": f"{ops_a.get('success_rate', 0.0):.1f}%",
            f"Experiment B ({exp_b.get('experiment_id')})": f"{ops_b.get('success_rate', 0.0):.1f}%",
            "Difference": diff_higher_pct(ops_b.get("success_rate", 0.0) / 100.0, ops_a.get("success_rate", 0.0) / 100.0),
        },
    ]

    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, width="stretch")
