"""
Evaluation Analytics Suite tab for SupportSphere AI Streamlit application.

Renders the complete evaluation workflow:
- Dataset upload & configuration
- Run New Experiment
- Experiment Registry (history, view, delete, compare)
- Single Run View with KPI cards, 5 metric tabs, ticket log, inspector, exports
- Comparison mode with radar, grouped bar, improvement panel
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from evaluation.experiment import ExperimentManager
from evaluation.runner import EvaluationRunner
from src.streamlit_app.charts import (
    get_actual_charts,
    get_comparison_charts,
    get_cost_comparison_chart,
    get_placeholder_charts,
    get_timeline_chart,
)
from src.streamlit_app.utils import (
    AppConfig,
    fmt_cost,
    fmt_ragas_metric,
    get_improvement_row,
    load_experiment_from_paths,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Internal Helpers
# =============================================================================

def _render_kpi_cards(summary_obj: Any) -> None:
    """Render 3×3 KPI metric cards.

    Args:
        summary_obj: EvaluationSummary or None.
    """
    st.markdown("### 🏆 KPI Performance Metrics")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col4, kpi_col5, kpi_col6 = st.columns(3)
    kpi_col7, kpi_col8, kpi_col9 = st.columns(3)

    if summary_obj is not None:
        kpi_col1.metric("Overall Accuracy", f"{summary_obj.request_type_metrics.accuracy:.2%}")
        kpi_col2.metric("Avg Faithfulness", fmt_ragas_metric(summary_obj.avg_faithfulness))
        kpi_col3.metric("Avg Correctness", fmt_ragas_metric(summary_obj.avg_answer_correctness))
        kpi_col4.metric("Avg Context Precision", fmt_ragas_metric(summary_obj.avg_context_precision))
        kpi_col5.metric("Avg Context Recall", fmt_ragas_metric(summary_obj.avg_context_recall))
        kpi_col6.metric("Execution Runtime", f"{summary_obj.elapsed_time_seconds:.2f}s")
        kpi_col7.metric("Avg Cost / Ticket", fmt_cost(summary_obj.avg_cost_per_ticket))
        kpi_col8.metric("Routing Accuracy", f"{summary_obj.retrieval_decision_accuracy:.2%}")
        kpi_col9.metric("Total Experiment Cost", fmt_cost(summary_obj.total_experiment_cost))
    else:
        st.info("ℹ️ No active experiment metrics loaded. Select a historical experiment or run evaluation above.")
        for col in [kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7, kpi_col8, kpi_col9]:
            col.metric("Metric Value", "N/A ℹ️")


def _render_metric_tabs(summary_obj: Any, results_list: List[Any]) -> None:
    """Render the 5 evaluation metric breakdown tabs.

    Args:
        summary_obj: EvaluationSummary.
        results_list: List of EvaluationResult.
    """
    st.markdown("### 🔍 Evaluation Metrics Breakdowns")

    fig_bar, fig_radar, fig_hist, fig_cm = get_actual_charts(summary_obj, results_list)

    sec_tab1, sec_tab2, sec_tab3, sec_tab4, sec_tab5 = st.tabs([
        "🎯 1. Classification Metrics",
        "🔀 2. Routing Metrics",
        "🧠 3. Knowledge (Ragas)",
        "📊 4. Generation Quality",
        "⚙️ 5. System Performance",
    ])

    with sec_tab1:
        st.markdown("#### Classification Accuracies")
        c1, c2, c3 = st.columns(3)
        c1.metric("Request Type Accuracy", f"{summary_obj.request_type_metrics.accuracy:.2%}")
        c2.metric("Product Area Accuracy", f"{summary_obj.product_area_metrics.accuracy:.2%}")
        c3.metric("Status Triage Accuracy", f"{summary_obj.status_metrics.accuracy:.2%}")

        class_metrics_data = []
        for name, metrics in [
            ("Request Type", summary_obj.request_type_metrics),
            ("Product Area", summary_obj.product_area_metrics),
            ("Status Triage", summary_obj.status_metrics),
        ]:
            class_metrics_data.append({
                "Category": name,
                "Accuracy": f"{metrics.accuracy:.2%}",
                "Precision (Macro)": f"{metrics.precision_macro:.4f}",
                "Recall (Macro)": f"{metrics.recall_macro:.4f}",
                "F1 Score (Macro)": f"{metrics.f1_macro:.4f}",
            })
        st.dataframe(pd.DataFrame(class_metrics_data), width="stretch")
        st.plotly_chart(fig_bar, width="stretch")

    with sec_tab2:
        st.markdown("#### Routing Accuracies")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Retrieval Decision", f"{summary_obj.retrieval_decision_accuracy:.2%}")
        r2.metric("Escalation Accuracy", f"{summary_obj.escalation_accuracy:.2%}")
        r3.metric("Out-of-Scope Accuracy", f"{summary_obj.out_of_scope_accuracy:.2%}")
        r4.metric("Greeting Accuracy", f"{summary_obj.greeting_accuracy:.2%}")

    with sec_tab3:
        st.markdown("#### Ragas Scores")
        rk1, rk2, rk3, rk4, rk5 = st.columns(5)
        rk1.metric("Context Precision", fmt_ragas_metric(summary_obj.avg_context_precision))
        rk2.metric("Context Recall", fmt_ragas_metric(summary_obj.avg_context_recall))
        rk3.metric("Faithfulness", fmt_ragas_metric(summary_obj.avg_faithfulness))
        rk4.metric("Answer Correctness", fmt_ragas_metric(summary_obj.avg_answer_correctness))
        rk5.metric("Answer Relevancy", fmt_ragas_metric(summary_obj.avg_answer_relevancy))

    with sec_tab4:
        st.markdown("#### Performance Shape")
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(fig_radar, width="stretch")
        with gc2:
            st.plotly_chart(fig_hist, width="stretch")

    with sec_tab5:
        st.markdown("#### System Performance Metrics")
        s1, s2, s3 = st.columns(3)
        s1.metric("Average Latency", f"{summary_obj.avg_latency:.2f}s")
        s2.metric("Average Chunks Retrieved", f"{summary_obj.avg_chunks:.2f}")
        s3.metric("Retrieval Skip Rate", f"{summary_obj.retrieval_skip_rate:.2%}")

        st.markdown("#### 💰 Billing & Token Metrics")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Avg Input Tokens", f"{summary_obj.avg_input_tokens:.0f}")
        b2.metric("Avg Output Tokens", f"{summary_obj.avg_output_tokens:.0f}")
        b3.metric("Avg Total Tokens", f"{summary_obj.avg_total_tokens:.0f}")
        b4.metric("Total Experiment Cost", fmt_cost(summary_obj.total_experiment_cost))


def _render_ticket_log(results_list: List[Any]) -> None:
    """Render the ticket evaluation output log table.

    Args:
        results_list: List of EvaluationResult.
    """
    st.markdown("### 🎫 Ticket Evaluation Output Log")
    table_rows = []
    for i, r in enumerate(results_list):
        table_rows.append({
            "Idx": i + 1,
            "Company": r.sample.company,
            "Subject": r.sample.subject,
            "Req Type Ok": "✅" if r.prediction.request_type_accuracy == 1 else "❌",
            "Status Ok": "✅" if r.prediction.status_accuracy == 1 else "❌",
            "Retrieval": "Run" if r.retrieval_required else "Skip",
            "Reason": r.routing_reason,
            "Confidence": f"{r.confidence:.2%}" if r.confidence is not None else "100%",
            "Faithfulness": round(r.ragas_metrics.faithfulness, 4) if r.ragas_metrics.faithfulness is not None else None,
            "Correctness": round(r.ragas_metrics.answer_correctness, 4) if r.ragas_metrics.answer_correctness is not None else None,
            "Cost ($)": f"{r.billing.total_cost:.6f}" if r.billing.total_cost > 0 else "—",
            "Latency": f"{r.latency:.2f}s",
        })
    st.dataframe(pd.DataFrame(table_rows), width="stretch")


def _render_ticket_inspector(results_list: List[Any]) -> None:
    """Render the detailed single ticket inspector.

    Args:
        results_list: List of EvaluationResult.
    """
    st.markdown("### 🔍 Detailed Sample Inspector")
    selected_idx = st.selectbox(
        "Select Ticket index to inspect",
        options=range(1, len(results_list) + 1),
        key="inspect_evaluation_ticket_idx",
    )
    detail_res = results_list[selected_idx - 1]
    c_i1, c_i2 = st.columns(2)
    with c_i1:
        st.markdown("**Customer Issue Inquiry**")
        st.info(detail_res.sample.issue)
        st.markdown("**Expected Response (Ground Truth)**")
        st.code(detail_res.sample.expected_response)
    with c_i2:
        st.markdown("**Generated RAG Response**")
        st.success(detail_res.generated_response)
        st.markdown("#### Ragas Metric Scores")
        st.write(f"- Faithfulness: `{detail_res.ragas_metrics.faithfulness}`")
        st.write(f"- Correctness: `{detail_res.ragas_metrics.answer_correctness}`")
        st.write(f"- Relevancy: `{detail_res.ragas_metrics.answer_relevancy}`")
        st.write(f"- Precision: `{detail_res.ragas_metrics.context_precision}`")
        st.write(f"- Recall: `{detail_res.ragas_metrics.context_recall}`")
        st.markdown("#### 💰 Billing Breakdown")
        st.write(f"- Decision Cost: `{fmt_cost(detail_res.billing.decision_cost)}`")
        st.write(f"- Retriever Cost: `{fmt_cost(detail_res.billing.retriever_cost)}`")
        st.write(f"- Generation Cost: `{fmt_cost(detail_res.billing.generation_cost)}`")
        st.write(f"- **Total Cost**: `{fmt_cost(detail_res.billing.total_cost)}`")


def _render_export_panel(summary_obj: Any) -> None:
    """Render the export buttons for current run.

    Args:
        summary_obj: EvaluationSummary.
    """
    st.markdown("### 📈 Export Viewed Run Reports")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_report = pd.DataFrame([{
            "Strategy": f"{summary_obj.provider}_{summary_obj.model}_{summary_obj.search_mode}_{summary_obj.reranker}_{summary_obj.top_k}",
            "Total Tickets": summary_obj.total_tickets,
            "Request Type Acc": summary_obj.request_type_metrics.accuracy,
            "Product Area Acc": summary_obj.product_area_metrics.accuracy,
            "Status Acc": summary_obj.status_metrics.accuracy,
            "Avg Faithfulness": summary_obj.avg_faithfulness,
            "Avg Correctness": summary_obj.avg_answer_correctness,
            "Avg Latency": summary_obj.avg_latency,
            "Avg Cost / Ticket": summary_obj.avg_cost_per_ticket,
            "Total Cost": summary_obj.total_experiment_cost,
        }])
        st.download_button(
            "Export Summary Metrics CSV",
            data=csv_report.to_csv(index=False).encode("utf-8"),
            file_name="evaluation_summary_metrics.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_exp2:
        summary_json_str = json.dumps({
            "provider": summary_obj.provider,
            "model": summary_obj.model,
            "search_mode": summary_obj.search_mode,
            "reranker": summary_obj.reranker,
            "top_k": summary_obj.top_k,
            "total_tickets": summary_obj.total_tickets,
            "request_type_accuracy": summary_obj.request_type_metrics.accuracy,
            "product_area_accuracy": summary_obj.product_area_metrics.accuracy,
            "status_accuracy": summary_obj.status_metrics.accuracy,
            "avg_faithfulness": summary_obj.avg_faithfulness,
            "avg_answer_correctness": summary_obj.avg_answer_correctness,
            "elapsed_time_seconds": summary_obj.elapsed_time_seconds,
            "avg_cost_per_ticket": summary_obj.avg_cost_per_ticket,
            "total_experiment_cost": summary_obj.total_experiment_cost,
        }, indent=4)
        st.download_button(
            "Export Summary Metrics JSON",
            data=summary_json_str.encode("utf-8"),
            file_name="evaluation_summary_metrics.json",
            mime="application/json",
            width="stretch",
        )


def _render_comparison_mode(experiments: List[Dict[str, Any]]) -> None:
    """Render the comparison mode view for two selected experiments.

    Args:
        experiments: Full experiments list from ExperimentManager.
    """
    exp_a_id = st.session_state.compare_exp_a
    exp_b_id = st.session_state.compare_exp_b

    exp_a = next((e for e in experiments if e["id"] == exp_a_id), None)
    exp_b = next((e for e in experiments if e["id"] == exp_b_id), None)

    if not exp_a or not exp_b:
        return

    st.markdown("---")
    st.markdown(f"## ⚔️ Comparison Report: `{exp_a_id}` (Previous) vs `{exp_b_id}` (Current)")

    st.markdown("### 📊 Performance Shape Comparison")
    fig_radar, fig_bar = get_comparison_charts(exp_a, exp_b)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(fig_radar, width="stretch")
    with chart_col2:
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("### 📈 Improvement Panel")
    ragas_metrics = [
        ("Faithfulness", exp_a.get("faithfulness"), exp_b.get("faithfulness")),
        ("Answer Correctness", exp_a.get("answer_correctness"), exp_b.get("answer_correctness")),
        ("Answer Relevancy", exp_a.get("answer_relevancy"), exp_b.get("answer_relevancy")),
        ("Context Precision", exp_a.get("context_precision"), exp_b.get("context_precision")),
        ("Context Recall", exp_a.get("context_recall"), exp_b.get("context_recall")),
    ]
    improvement_data = [get_improvement_row(name, va, vb) for name, va, vb in ragas_metrics]
    st.table(pd.DataFrame(improvement_data))

    st.markdown("### 💰 Cost Comparison")
    cost_a = exp_a.get("avg_cost_per_ticket") or 0.0
    cost_b = exp_b.get("avg_cost_per_ticket") or 0.0
    cost_diff = cost_b - cost_a
    cost_pct = (cost_diff / cost_a * 100) if cost_a > 0 else 0.0
    cost_col1, cost_col2, cost_col3 = st.columns(3)
    cost_col1.metric(f"Previous Avg Cost ({exp_a_id})", fmt_cost(cost_a))
    cost_col2.metric(f"Current Avg Cost ({exp_b_id})", fmt_cost(cost_b),
                     delta=f"{cost_pct:+.1f}%", delta_color="inverse")
    cost_col3.metric("Cost Difference", fmt_cost(abs(cost_diff)))

    # Ticket-by-ticket comparison
    st.markdown("---")
    st.markdown("### 🔍 Comparative Ticket Inspector")
    try:
        summary_a, results_a = load_experiment_from_paths(exp_a["summary_json_path"], exp_a["report_csv_path"])
        summary_b, results_b = load_experiment_from_paths(exp_b["summary_json_path"], exp_b["report_csv_path"])
    except Exception as e:
        st.error(f"Could not load experiment results for comparison: {e}")
        return

    common_len = min(len(results_a), len(results_b))
    if common_len > 0:
        selected_idx = st.selectbox(
            "Select Ticket index to inspect side-by-side",
            options=range(1, common_len + 1),
            key="inspect_comparison_ticket_idx",
        )

        res_a = results_a[selected_idx - 1]
        res_b = results_b[selected_idx - 1]

        st.info(f"**Customer Issue Inquiry**:\n{res_a.sample.issue}")
        st.markdown("**Expected Response (Ground Truth)**")
        st.code(res_a.sample.expected_response)

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(f"#### Previous Response ({exp_a_id})")
            st.success(res_a.generated_response)
        with col_i2:
            st.markdown(f"#### Current Response ({exp_b_id})")
            st.success(res_b.generated_response)

        st.markdown("#### Metric Differences for this Ticket")
        ticket_diff_data = [
            get_improvement_row("Faithfulness", res_a.ragas_metrics.faithfulness, res_b.ragas_metrics.faithfulness),
            get_improvement_row("Correctness", res_a.ragas_metrics.answer_correctness, res_b.ragas_metrics.answer_correctness),
            get_improvement_row("Relevancy", res_a.ragas_metrics.answer_relevancy, res_b.ragas_metrics.answer_relevancy),
            get_improvement_row("Precision", res_a.ragas_metrics.context_precision, res_b.ragas_metrics.context_precision),
            get_improvement_row("Recall", res_a.ragas_metrics.context_recall, res_b.ragas_metrics.context_recall),
        ]
        st.table(pd.DataFrame(ticket_diff_data))

    # Export comparison report
    st.markdown("#### Export Comparison Report")
    comparison_summary = {
        "experiment_a": exp_a_id,
        "experiment_b": exp_b_id,
        "metrics_comparison": improvement_data,
        "cost_comparison": {"previous": cost_a, "current": cost_b, "diff": cost_diff},
    }
    comp_json_str = json.dumps(comparison_summary, indent=4)
    st.download_button(
        "Export Comparison Metrics JSON",
        data=comp_json_str.encode("utf-8"),
        file_name=f"comparison_{exp_a_id}_vs_{exp_b_id}.json",
        mime="application/json",
        width="stretch",
    )


def _render_single_run_view(
    summary_obj: Any,
    results_list: List[Any],
    active_id: Optional[str],
) -> None:
    """Render the single experiment view mode.

    Args:
        summary_obj: EvaluationSummary or None.
        results_list: List of EvaluationResult.
        active_id: Current active experiment ID string, or None.
    """
    if active_id:
        st.markdown(f"### 👁️ Viewing Experiment Run: `{active_id}`")
    else:
        st.markdown("### 👁️ Viewing Active/Current Run")

    st.markdown("---")
    _render_kpi_cards(summary_obj)

    if summary_obj is not None:
        st.markdown("---")
        _render_metric_tabs(summary_obj, results_list)

        st.markdown("---")
        _render_ticket_log(results_list)

        _render_ticket_inspector(results_list)

        st.markdown("---")
        _render_export_panel(summary_obj)


# =============================================================================
# Main Entry Point
# =============================================================================

def render_evaluation(cfg: AppConfig) -> None:
    """Render the complete Evaluation Analytics Suite tab.

    Args:
        cfg: Current AppConfig.
    """
    st.title("Evaluation Analytics Suite")
    st.write("Compare system triage predictions and RAGAS accuracy against ground-truth validation sets.")
    st.markdown("---")

    exp_mgr = ExperimentManager()
    experiments = exp_mgr.list_experiments()

    # -----------------------------------------------------------------------
    # 1. Dataset Upload & Configuration
    # -----------------------------------------------------------------------
    st.markdown("### 📂 Evaluation Dataset & Configuration")

    col_u1, col_u2 = st.columns([2, 3])
    eval_filepath: Optional[str] = None

    with col_u1:
        eval_gt_file = st.file_uploader(
            "Upload Ground Truth CSV (Evaluation)",
            type=["csv"],
            key="eval_gt_file_uploader",
            help="Upload a CSV with true classifications and expected answers.",
        )

        if eval_gt_file is not None:
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"eval_{eval_gt_file.name}"
            with open(temp_path, "wb") as f:
                f.write(eval_gt_file.getbuffer())
            eval_filepath = str(temp_path)

            try:
                eval_df = pd.read_csv(eval_filepath)
                st.write(f"**Selected File**: `{eval_gt_file.name}`")
                st.write(f"**Dataset Size**: `{eval_df.memory_usage(deep=True).sum() / 1024:.1f} KB`")
                st.write(f"**Ticket Count**: `{len(eval_df)} rows`")
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        else:
            st.info("Upload a ground truth dataset to begin evaluation.")

    with col_u2:
        st.markdown("#### Current Run Configuration")
        cc1, cc2, cc3 = st.columns(3)
        cc1.write(f"**Provider**: `{cfg.provider}`")
        cc2.write(f"**Model**: `{cfg.selected_model}`")
        cc3.write(f"**Search Mode**: `{cfg.search_mode}`")

        cc4, cc5 = st.columns(2)
        cc4.write(f"**Top-K Chunks**: `{cfg.top_k}`")
        cc5.write(f"**Reranker**: `{cfg.reranker}`")

    # -----------------------------------------------------------------------
    # 2. Run New Experiment
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚙️ Run New Experiment")
    run_btn_disabled = (eval_filepath is None) or (not cfg.keys_provided)
    run_eval_pipeline = st.button("🚀 Run Evaluation", disabled=run_btn_disabled, width="stretch")

    if run_eval_pipeline:
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_elapsed_text = st.empty()
        t_start_pipeline = time.time()

        def _update_runner_status(current: int, total: int, status_msg: str) -> None:
            ratio = current / total if total > 0 else 0
            progress_bar.progress(ratio)
            elapsed = time.time() - t_start_pipeline
            status_text.text(status_msg)
            time_elapsed_text.text(f"⏱️ Elapsed: {elapsed:.1f}s")

        try:
            runner = EvaluationRunner()
            summary, results = runner.run(
                dataset_path=eval_filepath,
                provider=cfg.provider.lower(),
                model_name=cfg.selected_model,
                search_mode=cfg.search_mode.lower(),
                reranker=cfg.reranker.lower(),
                top_k=cfg.top_k,
                google_key=cfg.google_key,
                openai_key=cfg.openai_key,
                anthropic_key=cfg.anthropic_key,
                groq_key=cfg.groq_key,
                progress_callback=_update_runner_status,
            )

            st.session_state.eval_summary = summary
            st.session_state.eval_results = results
            st.session_state.active_experiment_id = None
            st.session_state.show_comparison = False
            st.session_state.eval_exception = None

            progress_bar.progress(1.0)
            status_text.success("✅ Evaluation pipeline completed successfully!")

        except Exception as exc:
            logger.error("Evaluation pipeline failed: %s", exc, exc_info=True)
            st.session_state.eval_exception = str(exc)
            st.error(f"Evaluation pipeline failed: {exc}")

    if st.session_state.get("eval_exception"):
        st.error(f"Last run error: {st.session_state.eval_exception}")

    # -----------------------------------------------------------------------
    # 3. Timeline Chart
    # -----------------------------------------------------------------------
    if experiments:
        st.markdown("---")
        st.markdown("### 📈 Experiment Performance Timeline")
        fig_timeline = get_timeline_chart(experiments)
        st.plotly_chart(fig_timeline, width="stretch")

        # Cost chart
        fig_cost = get_cost_comparison_chart(experiments)
        st.plotly_chart(fig_cost, width="stretch")

    # -----------------------------------------------------------------------
    # 4. Experiment Registry
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🗂️ Experiment Registry")

    if experiments:
        history_rows = []
        for e in experiments:
            history_rows.append({
                "ID": e["id"],
                "Name": e.get("friendly_name", e["id"]),
                "Timestamp": e.get("timestamp", "unknown"),
                "Provider": e.get("provider", ""),
                "Model": e.get("model", ""),
                "Mode": e.get("search_mode", ""),
                "Reranker": e.get("reranker", ""),
                "TopK": e.get("top_k", ""),
                "Faithfulness": f"{e['faithfulness']:.4f}" if e.get("faithfulness") is not None else "N/A",
                "Correctness": f"{e['answer_correctness']:.4f}" if e.get("answer_correctness") is not None else "N/A",
                "Avg Cost ($)": f"{e.get('avg_cost_per_ticket', 0.0):.6f}",
                "Tickets": e.get("total_tickets"),
                "Runtime": f"{e.get('elapsed_time', 0.0):.2f}s",
            })
        st.dataframe(pd.DataFrame(history_rows), width="stretch")

        act_col1, act_col2 = st.columns(2)
        with act_col1:
            st.markdown("#### Registry Actions")
            selected_action_exp = st.selectbox(
                "Select Experiment for Action",
                options=[e["id"] for e in experiments],
                key="dashboard_action_exp_selectbox",
            )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("👁️ View Run Details", width="stretch"):
                    exp = next(e for e in experiments if e["id"] == selected_action_exp)
                    st.session_state.active_experiment_id = exp["id"]
                    st.session_state.show_comparison = False
                    summary, results = load_experiment_from_paths(exp["summary_json_path"], exp["report_csv_path"])
                    st.session_state.eval_summary = summary
                    st.session_state.eval_results = results
                    st.success(f"Successfully loaded {exp['id']}!")
                    st.rerun()

            with btn_col2:
                if st.button("🗑️ Delete Run", width="stretch"):
                    if exp_mgr.delete_experiment(selected_action_exp):
                        st.success(f"Deleted experiment {selected_action_exp}!")
                        if st.session_state.get("active_experiment_id") == selected_action_exp:
                            st.session_state.active_experiment_id = None
                            st.session_state.eval_summary = None
                            st.session_state.eval_results = []
                        st.rerun()
                    else:
                        st.error(f"Failed to delete experiment {selected_action_exp}.")

        with act_col2:
            st.markdown("#### Compare Experiments")
            exp_a_opt = st.selectbox(
                "Experiment A (Baseline / Previous)",
                options=[e["id"] for e in experiments],
                key="dashboard_compare_a_selectbox",
            )
            exp_b_opt = st.selectbox(
                "Experiment B (Current / Optimized)",
                options=[e["id"] for e in experiments],
                key="dashboard_compare_b_selectbox",
            )

            if st.button("⚔️ Run Comparison", width="stretch"):
                if exp_a_opt == exp_b_opt:
                    st.warning("Please choose two different experiments to compare.")
                else:
                    st.session_state.compare_exp_a = exp_a_opt
                    st.session_state.compare_exp_b = exp_b_opt
                    st.session_state.show_comparison = True
                    st.success(f"Set comparison: {exp_a_opt} vs {exp_b_opt}")
                    st.rerun()
    else:
        st.info("The experiment registry is empty. Run an evaluation pipeline to register your first experiment!")

    # -----------------------------------------------------------------------
    # 5. Main Display: Comparison Mode vs Single Run View
    # -----------------------------------------------------------------------
    if st.session_state.get("show_comparison") and experiments:
        _render_comparison_mode(experiments)
    else:
        _render_single_run_view(
            summary_obj=st.session_state.get("eval_summary"),
            results_list=st.session_state.get("eval_results", []),
            active_id=st.session_state.get("active_experiment_id"),
        )
