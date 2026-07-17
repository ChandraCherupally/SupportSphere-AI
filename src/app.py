"""
Production-grade Streamlit application for SupportSphere AI.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root directory to python path to resolve 'src' and 'evaluation' packages correctly when running streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

import src.config as src_config
from evaluation.experiment import ExperimentManager
from evaluation.models import (
    ClassificationPrediction,
    RagasMetrics,
    EvaluationSummary,
    EvaluationSample,
    EvaluationResult,
    ClassificationMetrics,
)
from evaluation.runner import EvaluationRunner
from src.ai.models import SupportTicket
from src.graph.support_agent import SupportAgent

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s| %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Configure Streamlit page settings
st.set_page_config(
    page_title="SupportSphere AI - Dashboard & Evaluation Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State values
if "eval_summary" not in st.session_state:
    st.session_state.eval_summary = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []
if "batch_df" not in st.session_state:
    st.session_state.batch_df = None
if "batch_outputs" not in st.session_state:
    st.session_state.batch_outputs = None
if "eval_exception" not in st.session_state:
    st.session_state.eval_exception = None
if "active_experiment_id" not in st.session_state:
    st.session_state.active_experiment_id = None
if "compare_exp_a" not in st.session_state:
    st.session_state.compare_exp_a = None
if "compare_exp_b" not in st.session_state:
    st.session_state.compare_exp_b = None
if "show_comparison" not in st.session_state:
    st.session_state.show_comparison = False

# =============================================================================
# SIDEBAR (GLOBAL SETTINGS ONLY)
# =============================================================================

st.sidebar.markdown("# SupportSphere AI")

st.sidebar.markdown("---")

# Group 1: Inference Settings
st.sidebar.markdown("### Inference Settings")
provider = st.sidebar.selectbox(
    "Provider",
    options=["Google", "OpenAI", "Groq"],
    index=0,
    key="sb_provider"
)

# Dynamically change models based on provider selection
model_options = {
    "Google": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    "Groq": ["llama-3.3-70b-versatile"]
}

selected_model = st.sidebar.selectbox(
    "Model",
    options=model_options[provider],
    index=0,
    key="sb_model"
)

# Text inputs for API Keys (Default to environment variables if available)
google_key = st.sidebar.text_input(
    "Google API Key",
    value=os.getenv("GOOGLE_API_KEY", ""),
    type="password"
)
groq_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password"
)
openai_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password"
)
anthropic_key = ""


# Verify required keys are provided based on the active provider selection
keys_provided = True
if provider == "Google" and not google_key:
    keys_provided = False
elif provider == "OpenAI" and not openai_key:
    keys_provided = False
elif provider == "Groq" and not groq_key:
    keys_provided = False

st.sidebar.markdown("---")

# Group 2: Retrieval Settings
st.sidebar.markdown("### Retrieval Settings")
top_k = st.sidebar.slider(
    "Top-K Chunks",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    key="sb_top_k"
)
search_mode = st.sidebar.selectbox(
    "Search Mode",
    options=["Hybrid", "Vector", "BM25"],
    index=0,
    key="sb_search_mode"
)
reranker = st.sidebar.selectbox(
    "Reranker",
    options=["None", "FlashRank", "CrossEncoder", "LLM"],
    index=0,
    key="sb_reranker"
)

st.sidebar.markdown("---")

# Group 3: System Information (No action controls)
st.sidebar.markdown("### System Information")
st.sidebar.write(f"**Current Provider**: `{provider}`")
st.sidebar.write(f"**Current Model**: `{selected_model}`")
st.sidebar.write(f"**Current Search Mode**: `{search_mode}`")
st.sidebar.write(f"**Current Reranker**: `{reranker}`")
st.sidebar.write("**Application Version**: `v1.2.0`")

# =============================================================================
# MAIN NAVIGATION TABS
# =============================================================================

tab_home, tab_single, tab_batch, tab_eval = st.tabs(
    ["🏠 Dashboard", "🎫 Single Ticket", "📂 Batch Processing", "📊 Evaluation Analytics"]
)

# Helper function to patch environment keys dynamically before running SupportAgent
def _patch_agent_configs():
    src_config.LLM_PROVIDER = provider.lower()
    src_config.LLM_MODEL = selected_model
    src_config.GOOGLE_API_KEY = google_key
    src_config.OPENAI_API_KEY = openai_key
    src_config.ANTHROPIC_API_KEY = anthropic_key
    src_config.GROQ_API_KEY = groq_key
    src_config.FINAL_TOP_K = top_k
    src_config.MAX_CONTEXT_CHUNKS = top_k
    src_config.BM25_TOP_K = top_k * 2
    src_config.VECTOR_TOP_K = top_k * 2
    if hasattr(src_config, "SEARCH_MODE"):
        setattr(src_config, "SEARCH_MODE", search_mode.lower())

def show_active_retrieval_config(actual_retrieved: int, expected_top_k: int):
    """
    Renders active retrieval settings summary and validates retrieved vs configured top-k.
    """
    st.markdown("#### ⚙️ Active Retrieval Configuration")
    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
    rc1.write(f"**Provider**: `{provider}`")
    rc2.write(f"**Model**: `{selected_model}`")
    rc3.write(f"**Search Mode**: `{search_mode}`")
    rc4.write(f"**Reranker**: `{reranker}`")
    rc5.write(f"**Top-K (Configured)**: `{expected_top_k}`")
    rc6.write(f"**Retrieved Chunks**: `{actual_retrieved}`")

    # Display warning if they differ (Requirement 7)
    if actual_retrieved != expected_top_k:
        st.warning(f"⚠️ Retrieved chunk count ({actual_retrieved}) does not match configured Top-K ({expected_top_k}).")

    # Assert retrieved <= top-k (Requirement 8)
    if actual_retrieved > expected_top_k:
        st.warning("⚠️ Mismatch: Retrieved chunks count exceeds configured Top-K limit.")
        logger.warning(f"Runtime validation mismatch: actual_retrieved ({actual_retrieved}) > expected_top_k ({expected_top_k})")

# =============================================================================
# DASHBOARD TAB
# =============================================================================

with tab_home:
    st.title("SupportSphere AI Dashboard")
    st.write("Welcome to the SupportSphere AI enterprise multi-domain support triage and evaluation system.")
    
    st.markdown("### Active Configuration Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("LLM Provider", provider)
    with col2:
        st.metric("LLM Model", selected_model)
    with col3:
        st.metric("Retrieval Search Mode", search_mode)
    with col4:
        st.metric("Reranker", reranker)

    st.markdown("### System Architecture")
    st.info(
        "SupportSphere AI is designed on a modular pipeline structure using LangGraph. "
        "It includes hybrid search (dense embeddings + BM25 sparse keyword search) with Reranker nodes "
        "to feed contexts directly to LLMs under rule-based classification and output guardrail structures."
    )
    
    st.markdown("#### System Metrics Summary Status")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.write("**Supported Companies**: HackerRank, Claude, Visa")
        st.write("**Database Provider**: Pinecone Vector DB")
        st.write("**Embeddings Model**: gemini-embedding-001 (Google)")
    with sc2:
        st.write("**System Status**: Active & Standby")
        st.write("**Classification Engine**: Rule-based + LLM validation")
        st.write("**Framework Version**: v1.2.0")

# =============================================================================
# SINGLE TICKET TAB
# =============================================================================

with tab_single:
    st.title("Single Ticket Support Triage")
    st.write("Submit a single support inquiry to view real-time RAG response classification and context references.")

    with st.form("single_ticket_form"):
        issue = st.text_area("Customer Issue", value="I cannot access my test session, it keeps saying link expired.")
        subject = st.text_input("Subject", value="Test access expired")
        company = st.selectbox("Company / Client Domain", options=["HackerRank", "Claude", "Visa", "None"])
        
        submit_btn = st.form_submit_button("Triage Support Ticket")

    if submit_btn:
        if not issue.strip():
            st.error("Please enter a valid issue description.")
        else:
            with st.spinner("Processing inquiry through SupportSphere AI..."):
                _patch_agent_configs()
                
                agent = SupportAgent()
                ticket = SupportTicket(
                    issue=issue,
                    subject=subject,
                    company=company if company != "None" else "",
                    reranker=reranker.lower()
                )
                
                t_start = time.perf_counter()
                try:
                    res = agent.invoke(ticket)
                    t_elapsed = time.perf_counter() - t_start
                    
                    st.success("Triage completed successfully!")
                    
                    st.markdown("### Decision Gate & Classification Outputs")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Request Type", str(res.response.request_type))
                    c2.metric("Product Area", str(res.response.product_area))
                    c3.metric("Status Triage", str(res.response.status))
                    
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Retrieval Required", "True ✅" if res.retrieval_required else "False ❌")
                    c5.metric("Routing Confidence", f"{res.confidence:.2%}")
                    with c6:
                        st.write(f"**Routing Reason**:\n{res.routing_reason or 'None'}")
                    
                    st.markdown("### Performance")
                    p1, p2 = st.columns(2)
                    p1.metric("Response Latency", f"{t_elapsed:.2f}s")
                    p2.metric("Retrieved Chunks Count", res.num_chunks)
                    
                    show_active_retrieval_config(actual_retrieved=res.num_chunks, expected_top_k=top_k)
                    
                    st.markdown("### Generated Support Response")
                    st.write(res.response.response)
                    
                    st.markdown("#### Classification Justification")
                    st.info(res.response.justification)
                    
                    # Expandable chunks viewer
                    st.markdown("### Retrieved Context Reference Chunks")
                    for i, chunk in enumerate(res.retrieved_chunks):
                        with st.expander(f"Chunk #{i+1} (Source: {chunk.company} | Area: {chunk.product_area} | Score: {chunk.score:.4f})"):
                            st.write(f"**URL**: [{chunk.url}]({chunk.url})")
                            st.markdown(f"**Text Reference**:\n```\n{chunk.text}\n```")
                            
                except Exception as e:
                    logger.error(f"Failed to triage single ticket: {e}", exc_info=True)
                    st.error(f"Processing failed: {e}")

# =============================================================================
# BATCH PROCESSING TAB
# =============================================================================

with tab_batch:
    st.title("Batch Support Triage")
    st.write("Upload a CSV file containing batch support tickets to parse and class-label them.")

    uploaded_batch = st.file_uploader(
        "Upload Tickets CSV",
        type=["csv"],
        key="uploaded_batch_file"
    )

    if uploaded_batch is not None:
        try:
            df = pd.read_csv(uploaded_batch)
            st.session_state.batch_df = df
            st.success(f"Loaded batch file containing {len(df)} rows.")
            st.dataframe(df.head(5))
        except Exception as e:
            st.error(f"Failed to load batch file: {e}")

    # Process batch predictions
    if st.session_state.batch_df is not None:
        batch_btn = st.button("Run Batch Triage", key="run_batch_triage_btn")
        
        if batch_btn:
            df = st.session_state.batch_df
            _patch_agent_configs()
            
            agent = SupportAgent()
            outputs = []
            
            p_bar = st.progress(0)
            status_text = st.empty()
            
            t_start = time.time()
            for idx, row in df.iterrows():
                p_bar.progress((idx + 1) / len(df))
                status_text.text(f"Processing ticket {idx + 1}/{len(df)}...")
                
                issue_val = str(row.get("Issue", row.get("issue", ""))).strip()
                subj_val = str(row.get("Subject", row.get("subject", ""))).strip()
                comp_val = str(row.get("Company", row.get("company", ""))).strip()
                if comp_val.lower() == "nan" or comp_val.lower() == "none" or not comp_val:
                    comp_val = ""
                    
                ticket = SupportTicket(
                    issue=issue_val,
                    subject=subj_val,
                    company=comp_val,
                    reranker=reranker.lower()
                )
                
                try:
                    res = agent.invoke(ticket)
                    outputs.append({
                        "Issue": issue_val,
                        "Subject": subj_val,
                        "Company": comp_val if comp_val else "None",
                        "Predicted Request Type": res.response.request_type,
                        "Predicted Product Area": res.response.product_area,
                        "Predicted Status": res.response.status,
                        "Generated Response": res.response.response,
                        "Num Chunks": res.num_chunks
                    })
                except Exception as e:
                    logger.error(f"Failed prediction on batch item {idx}: {e}")
                    outputs.append({
                        "Issue": issue_val,
                        "Subject": subj_val,
                        "Company": comp_val if comp_val else "None",
                        "Predicted Request Type": "invalid",
                        "Predicted Product Area": "General",
                        "Predicted Status": "Replied",
                        "Generated Response": "Prediction failed.",
                        "Num Chunks": 0
                    })
            
            t_elapsed = time.time() - t_start
            p_bar.empty()
            status_text.empty()
            
            out_df = pd.DataFrame(outputs)
            st.session_state.batch_outputs = out_df
            
            st.markdown("### Batch Run Summary")
            col1, col2 = st.columns(2)
            col1.metric("Total Tickets Processed", len(out_df))
            col2.metric("Total Runtime (seconds)", f"{t_elapsed:.2f}s")
            
            # Show configuration details based on average chunks retrieved
            avg_chunks_retrieved = int(out_df["Num Chunks"].mean())
            show_active_retrieval_config(actual_retrieved=avg_chunks_retrieved, expected_top_k=top_k)
            
            st.markdown("### Interactive Results Table")
            st.dataframe(out_df, width=1200)
            
            csv_data = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions CSV",
                data=csv_data,
                file_name="triage_predictions.csv",
                mime="text/csv"
            )

# =============================================================================
# EVALUATION ANALYTICS TAB
# =============================================================================

def get_placeholder_charts():
    """Generates beautiful placeholder Plotly charts for the empty state."""
    # 1. Bar Chart
    acc_df = pd.DataFrame({
        "Category": ["Request Type", "Product Area", "Status"],
        "Accuracy": [0.0, 0.0, 0.0]
    })
    fig_bar = px.bar(
        acc_df, x="Category", y="Accuracy", color="Category",
        text_auto=".1%", range_y=[0, 1.05],
        title="Classification Accuracies (No Data Loaded)"
    )
    fig_bar.update_layout(showlegend=False)

    # 2. Radar Chart
    categories = ['Context Precision', 'Context Recall', 'Faithfulness', 'Answer Correctness', 'Answer Relevancy']
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=[0.0, 0.0, 0.0, 0.0, 0.0],
        theta=categories,
        fill='toself',
        line_color="#a0a0a0"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Ragas Performance Shape (No Data Loaded)"
    )

    # 3. Histogram
    fig_hist = px.histogram(
        x=[0.0], nbins=10,
        labels={'x': 'Faithfulness Score'},
        title="Faithfulness Score Distribution (No Data Loaded)"
    )

    # 4. Heatmap
    fig_cm = ff.create_annotated_heatmap(
        z=[[0, 0], [0, 0]],
        x=["Class A", "Class B"],
        y=["Class A", "Class B"],
        colorscale="Greys",
        showscale=True
    )
    fig_cm.update_layout(
        title="Confusion Matrix (No Data Loaded)",
        xaxis_title="Predicted",
        yaxis_title="Ground Truth"
    )

    return fig_bar, fig_radar, fig_hist, fig_cm


def get_actual_charts(summary, results):
    """Generates active Plotly charts based on summary and results data."""
    # 1. Bar Chart
    acc_df = pd.DataFrame({
        "Category": ["Request Type", "Product Area", "Status"],
        "Accuracy": [
            summary.request_type_metrics.accuracy,
            summary.product_area_metrics.accuracy,
            summary.status_metrics.accuracy
        ]
    })
    fig_bar = px.bar(
        acc_df, x="Category", y="Accuracy", color="Category",
        text_auto=".1%", range_y=[0, 1.05],
        title="Classification Accuracies"
    )
    fig_bar.update_layout(showlegend=False)

    # 2. Radar Chart
    categories = ['Context Precision', 'Context Recall', 'Faithfulness', 'Answer Correctness', 'Answer Relevancy']
    r_values = [
        summary.avg_context_precision if summary.avg_context_precision is not None else 0.0,
        summary.avg_context_recall if summary.avg_context_recall is not None else 0.0,
        summary.avg_faithfulness if summary.avg_faithfulness is not None else 0.0,
        summary.avg_answer_correctness if summary.avg_answer_correctness is not None else 0.0,
        summary.avg_answer_relevancy if summary.avg_answer_relevancy is not None else 0.0,
    ]
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=r_values,
        theta=categories,
        fill='toself',
        line_color="#4F8BF9"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Ragas Performance Shape"
    )

    # 3. Histogram
    hist_data = [r.ragas_metrics.faithfulness for r in results if r.ragas_metrics.faithfulness is not None]
    if not hist_data:
        hist_data = [0.0]
    fig_hist = px.histogram(
        x=hist_data, nbins=10,
        labels={'x': 'Faithfulness Score'},
        title="Faithfulness Score Distribution"
    )

    # 4. Heatmap
    y_true = [r.sample.expected_request_type for r in results]
    y_pred = [r.prediction.predicted_request_type for r in results]
    labels = sorted(list(set(y_true) | set(y_pred)))
    if labels:
        z = summary.request_type_metrics.confusion_matrix
        x = labels
        y = labels
    else:
        z = [[0, 0], [0, 0]]
        x = ["A", "B"]
        y = ["A", "B"]
        
    fig_cm = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=True
    )
    fig_cm.update_layout(
        title="Confusion Matrix (Request Type)",
        xaxis_title="Predicted",
        yaxis_title="Ground Truth"
    )

    return fig_bar, fig_radar, fig_hist, fig_cm


def load_experiment_from_paths(summary_json_path, report_csv_path):
    with open(summary_json_path, "r", encoding="utf-8") as f:
        exp_json = json.load(f)
        
    meta = exp_json.get("metadata", {})
    class_dict = exp_json.get("classification", {})
    routing_dict = exp_json.get("routing", {})
    ragas_dict = exp_json.get("ragas", {})
    
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
            per_label_metrics=d.get("per_label_metrics", {})
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
        top_k=meta.get("top_k", 0)
    )
    
    df_csv = pd.read_csv(report_csv_path)
    results = []
    for _, row in df_csv.iterrows():
        sample = EvaluationSample(
            issue=str(row.get("issue")),
            subject=str(row.get("subject", "")),
            company=str(row.get("company", "None")),
            expected_response=str(row.get("expected_response")),
            expected_request_type=str(row.get("expected_request_type")),
            expected_product_area=str(row.get("expected_product_area")),
            expected_status=str(row.get("expected_status"))
        )
        
        pred = ClassificationPrediction(
            predicted_request_type=str(row.get("predicted_request_type")),
            predicted_product_area=str(row.get("predicted_product_area")),
            predicted_status=str(row.get("predicted_status")),
            request_type_accuracy=int(row.get("request_type_accuracy", 0)),
            product_area_accuracy=int(row.get("product_area_accuracy", 0)),
            status_accuracy=int(row.get("status_accuracy", 0))
        )
        
        ragas = RagasMetrics(
            context_precision=float(row.get("ragas_context_precision")) if pd.notna(row.get("ragas_context_precision")) else None,
            context_recall=float(row.get("ragas_context_recall")) if pd.notna(row.get("ragas_context_recall")) else None,
            faithfulness=float(row.get("ragas_faithfulness")) if pd.notna(row.get("ragas_faithfulness")) else None,
            answer_correctness=float(row.get("ragas_answer_correctness")) if pd.notna(row.get("ragas_answer_correctness")) else None,
            answer_relevancy=float(row.get("ragas_answer_relevancy")) if pd.notna(row.get("ragas_answer_relevancy")) else None
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
                confidence=float(row.get("routing_confidence", 1.0))
            )
        )
        
    return summary, results


def get_comparison_charts(exp_a, exp_b):
    categories = ['Context Precision', 'Context Recall', 'Faithfulness', 'Answer Correctness', 'Answer Relevancy']
    a_vals = [
        exp_a.get("context_precision") or 0.0,
        exp_a.get("context_recall") or 0.0,
        exp_a.get("faithfulness") or 0.0,
        exp_a.get("answer_correctness") or 0.0,
        exp_a.get("answer_relevancy") or 0.0,
    ]
    b_vals = [
        exp_b.get("context_precision") or 0.0,
        exp_b.get("context_recall") or 0.0,
        exp_b.get("faithfulness") or 0.0,
        exp_b.get("answer_correctness") or 0.0,
        exp_b.get("answer_relevancy") or 0.0,
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=a_vals,
        theta=categories,
        fill='toself',
        name=f"Previous ({exp_a['id']})"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=b_vals,
        theta=categories,
        fill='toself',
        name=f"Current ({exp_b['id']})"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="RAGAS Performance Radar Comparison"
    )
    
    fig_bar = go.Figure(data=[
        go.Bar(name=f"Previous ({exp_a['id']})", x=categories, y=a_vals),
        go.Bar(name=f"Current ({exp_b['id']})", x=categories, y=b_vals)
    ])
    fig_bar.update_layout(
        barmode='group',
        title="Grouped RAGAS Scores",
        yaxis=dict(range=[0, 1.05])
    )
    return fig_radar, fig_bar


def get_improvement_row(metric_name, val_a, val_b):
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
        "Improvement": arrow
    }


with tab_eval:
    st.title("Evaluation Analytics Suite")
    st.write("Compare system triage predictions and RAGAS answers accuracy against ground-truth validation sets.")
    st.markdown("---")

    # Fetch comparative runs/experiments
    exp_mgr = ExperimentManager()
    experiments = exp_mgr.list_experiments()
    
    # ---------------------------------------------------------
    # 1. Dataset Upload & Configuration Settings
    # ---------------------------------------------------------
    st.markdown("### 📂 Evaluation Dataset & Configuration")
    
    col_u1, col_u2 = st.columns([2, 3])
    with col_u1:
        eval_gt_file = st.file_uploader(
            "Upload Ground Truth CSV (Evaluation)",
            type=["csv"],
            key="eval_gt_file_uploader",
            help="Upload a CSV with true classifications and expected answers."
        )
        
        eval_filepath: Optional[str] = None
        if eval_gt_file is not None:
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"eval_{eval_gt_file.name}"
            with open(temp_path, "wb") as f:
                f.write(eval_gt_file.getbuffer())
            eval_filepath = str(temp_path)
            
            # Load dataset to show details
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
        cc1.write(f"**Provider**: `{provider}`")
        cc2.write(f"**Model**: `{selected_model}`")
        cc3.write(f"**Search Mode**: `{search_mode}`")
        
        cc4, cc5 = st.columns(2)
        cc4.write(f"**Top-K Chunks**: `{top_k}`")
        cc5.write(f"**Reranker**: `{reranker}`")

    # ---------------------------------------------------------
    # 2. Run New Experiment
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚙️ Run New Experiment")
    run_btn_disabled = (eval_filepath is None) or (not keys_provided)
    run_eval_pipeline = st.button("🚀 Run Evaluation", disabled=run_btn_disabled, width='stretch')

    if run_eval_pipeline:
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_elapsed_text = st.empty()
        
        t_start_pipeline = time.time()

        def _update_runner_status(current: int, total: int, status_msg: str):
            ratio = current / total
            progress_bar.progress(ratio)
            elapsed = time.time() - t_start_pipeline
            eta = (elapsed / ratio) - elapsed if ratio > 0 else 0.0
            status_text.text(f"Stage: {status_msg} | Ticket {current}/{total}")
            time_elapsed_text.text(f"Elapsed Time: {elapsed:.1f}s | ETA: {eta:.1f}s")

        try:
            runner = EvaluationRunner()
            summary, results = runner.run(
                dataset_path=eval_filepath,  # type: ignore
                provider=provider,
                model_name=selected_model,
                search_mode=search_mode,
                reranker=reranker,
                top_k=top_k,
                google_key=google_key,
                openai_key=openai_key,
                anthropic_key=anthropic_key,
                groq_key=groq_key,
                progress_callback=_update_runner_status
            )
            st.session_state.eval_summary = summary
            st.session_state.eval_results = results
            st.session_state.eval_exception = None
            
            # Load the newly created run as the active viewed run
            refreshed_exps = exp_mgr.list_experiments()
            if refreshed_exps:
                latest_exp = refreshed_exps[-1]
                st.session_state.active_experiment_id = latest_exp["id"]
                st.session_state.show_comparison = False
                
            st.success(f"Evaluation completed successfully! Created and loaded: {st.session_state.active_experiment_id}")
            progress_bar.empty()
            status_text.empty()
            time_elapsed_text.empty()
            st.rerun()
        except Exception as e:
            logger.error(f"Failed to run pipeline evaluation: {e}", exc_info=True)
            st.session_state.eval_summary = None
            st.session_state.eval_results = []
            st.session_state.eval_exception = str(e)
            st.error(f"Evaluation failed: {e}")

    # ---------------------------------------------------------
    # 3. Chronological Timeline Trends (Optional)
    # ---------------------------------------------------------
    if experiments:
        st.markdown("---")
        st.markdown("### 📈 Chronological Metric Evolution Timeline")
        timeline_rows = []
        for e in experiments:
            timeline_rows.append({
                "Experiment ID": e.get("id"),
                "Timestamp": e.get("timestamp"),
                "Faithfulness": e.get("faithfulness") if e.get("faithfulness") is not None else 0.0,
                "Answer Correctness": e.get("answer_correctness") if e.get("answer_correctness") is not None else 0.0,
                "Context Precision": e.get("context_precision") if e.get("context_precision") is not None else 0.0,
            })
        
        timeline_df = pd.DataFrame(timeline_rows)
        fig_line = px.line(
            timeline_df,
            x="Experiment ID",
            y=["Faithfulness", "Answer Correctness", "Context Precision"],
            markers=True,
            title="Registry Performance Trend",
            labels={"value": "Score", "variable": "Metric"}
        )
        fig_line.update_layout(yaxis_range=[0, 1.05], hovermode="x unified")
        st.plotly_chart(fig_line, width='stretch')

    # ---------------------------------------------------------
    # 4. Experiment Registry & History
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🗂️ Experiment History & Registry")
    
    if experiments:
        # Render clean registry dataframe
        history_rows = []
        for e in experiments:
            history_rows.append({
                "Experiment ID": e.get("id"),
                "Timestamp": e.get("timestamp"),
                "Provider": e.get("provider"),
                "Model": e.get("model"),
                "Search Mode": e.get("search_mode"),
                "Reranker": e.get("reranker"),
                "Top-K": e.get("top_k"),
                "Tickets": e.get("total_tickets"),
                "Runtime": f"{e.get('elapsed_time', 0.0):.2f}s"
            })
        st.dataframe(pd.DataFrame(history_rows), width='stretch')
        
        # Action controls below the table
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            st.markdown("#### Registry Actions")
            selected_action_exp = st.selectbox(
                "Select Experiment for Action",
                options=[e["id"] for e in experiments],
                key="dashboard_action_exp_selectbox"
            )
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("👁️ View Run Details", width='stretch'):
                    exp = next(e for e in experiments if e["id"] == selected_action_exp)
                    st.session_state.active_experiment_id = exp["id"]
                    st.session_state.show_comparison = False
                    
                    summary, results = load_experiment_from_paths(exp["summary_json_path"], exp["report_csv_path"])
                    st.session_state.eval_summary = summary
                    st.session_state.eval_results = results
                    st.success(f"Successfully loaded {exp['id']}!")
                    st.rerun()
                    
            with btn_col2:
                if st.button("🗑️ Delete Run", width='stretch'):
                    if exp_mgr.delete_experiment(selected_action_exp):
                        st.success(f"Deleted experiment {selected_action_exp}!")
                        if st.session_state.active_experiment_id == selected_action_exp:
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
                key="dashboard_compare_a_selectbox"
            )
            exp_b_opt = st.selectbox(
                "Experiment B (Current / Optimized)",
                options=[e["id"] for e in experiments],
                key="dashboard_compare_b_selectbox"
            )
            
            if st.button("⚔️ Run Comparison", width='stretch'):
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

    # ---------------------------------------------------------
    # 3. Main Display logic (Comparison Mode vs View Mode)
    # ---------------------------------------------------------
    if st.session_state.show_comparison and experiments:
        exp_a_id = st.session_state.compare_exp_a
        exp_b_id = st.session_state.compare_exp_b
        
        exp_a = next((e for e in experiments if e["id"] == exp_a_id), None)
        exp_b = next((e for e in experiments if e["id"] == exp_b_id), None)
        
        if exp_a and exp_b:
            st.markdown("---")
            st.markdown(f"## ⚔️ Comparison Report: `{exp_a_id}` (Previous) vs `{exp_b_id}` (Current)")
            
            # Grouped charts: Radar and Grouped Bar
            st.markdown("### 📊 Performance Shape Comparison")
            fig_radar, fig_bar = get_comparison_charts(exp_a, exp_b)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(fig_radar, width='stretch')
            with chart_col2:
                st.plotly_chart(fig_bar, width='stretch')
                
            # Improvement Panel Table
            st.markdown("### 📈 Improvement Panel")
            metrics_list = [
                ("Faithfulness", exp_a.get("faithfulness"), exp_b.get("faithfulness")),
                ("Answer Correctness", exp_a.get("answer_correctness"), exp_b.get("answer_correctness")),
                ("Answer Relevancy", exp_a.get("answer_relevancy"), exp_b.get("answer_relevancy")),
                ("Context Precision", exp_a.get("context_precision"), exp_b.get("context_precision")),
                ("Context Recall", exp_a.get("context_recall"), exp_b.get("context_recall")),
            ]
            
            improvement_data = []
            for name, val_a, val_b in metrics_list:
                improvement_data.append(get_improvement_row(name, val_a, val_b))
                
            st.table(pd.DataFrame(improvement_data))
            
            # Side-by-side Ticket Inspector comparison
            st.markdown("---")
            st.markdown("### 🔍 Comparative Ticket Inspector")
            
            # Load results for both A and B to inspect
            summary_a, results_a = load_experiment_from_paths(exp_a["summary_json_path"], exp_a["report_csv_path"])
            summary_b, results_b = load_experiment_from_paths(exp_b["summary_json_path"], exp_b["report_csv_path"])
            
            common_len = min(len(results_a), len(results_b))
            if common_len > 0:
                selected_idx = st.selectbox(
                    "Select Ticket index to inspect side-by-side",
                    options=range(1, common_len + 1),
                    key="inspect_comparison_ticket_idx"
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
                
            # Report Export comparison
            st.markdown("#### Export Comparison Report")
            comparison_summary = {
                "experiment_a": exp_a_id,
                "experiment_b": exp_b_id,
                "metrics_comparison": improvement_data
            }
            comp_json_str = json.dumps(comparison_summary, indent=4)
            st.download_button(
                "Export Comparison Metrics JSON",
                data=comp_json_str.encode('utf-8'),
                file_name=f"comparison_{exp_a_id}_vs_{exp_b_id}.json",
                mime="application/json",
                width='stretch'
            )
            
    else:
        # SINGLE RUN VIEW MODE
        summary_obj = st.session_state.eval_summary
        results_list = st.session_state.eval_results
        active_id = st.session_state.active_experiment_id
        
        if active_id:
            st.markdown(f"### 👁️ Viewing Experiment Run: `{active_id}`")
        else:
            st.markdown("### 👁️ Viewing Active/Current Run")

        # KPI Summary Cards (3x3 grid)
        st.markdown("---")
        st.markdown("### 🏆 KPI Performance Metrics")
        
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col4, kpi_col5, kpi_col6 = st.columns(3)
        kpi_col7, kpi_col8, kpi_col9 = st.columns(3)
        
        if summary_obj is not None:
            error_msg = summary_obj.evaluation_error
            def fmt_ragas_metric(val):
                return f"{val:.4f}" if val is not None else "N/A"
                
            kpi_col1.metric("Overall Accuracy", f"{summary_obj.request_type_metrics.accuracy:.2%}")
            kpi_col2.metric("Avg Faithfulness", fmt_ragas_metric(summary_obj.avg_faithfulness))
            kpi_col3.metric("Avg Correctness", fmt_ragas_metric(summary_obj.avg_answer_correctness))
            
            kpi_col4.metric("Avg Context Precision", fmt_ragas_metric(summary_obj.avg_context_precision))
            kpi_col5.metric("Avg Context Recall", fmt_ragas_metric(summary_obj.avg_context_recall))
            kpi_col6.metric("Execution Runtime", f"{summary_obj.elapsed_time_seconds:.2f}s")
            
            kpi_col7.metric("Retrieved Chunks", f"{summary_obj.avg_chunks:.2f}")
            kpi_col8.metric("Routing Accuracy", f"{summary_obj.retrieval_decision_accuracy:.2%}")
            kpi_col9.metric("Knowledge Accuracy", fmt_ragas_metric(summary_obj.avg_answer_correctness))
        else:
            st.info("ℹ️ No active experiment metrics loaded. Select a historical experiment or run evaluation above.")
            for col in [kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7, kpi_col8, kpi_col9]:
                col.metric("Metric Value", "N/A ℹ️")

        if summary_obj is not None:
            # 5 tab metrics breakdowns
            st.markdown("---")
            st.markdown("### 🔍 Evaluation Metrics Breakdowns")
            
            sec_tab1, sec_tab2, sec_tab3, sec_tab4, sec_tab5 = st.tabs([
                "🎯 1. Classification Metrics",
                "🔀 2. Routing Metrics",
                "🧠 3. Knowledge (Ragas)",
                "📊 4. Generation Quality",
                "⚙️ 5. System Performance"
            ])
            
            fig_bar, fig_radar, fig_hist, fig_cm = get_actual_charts(summary_obj, results_list)
            
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
                    ("Status Triage", summary_obj.status_metrics)
                ]:
                    class_metrics_data.append({
                        "Category": name,
                        "Accuracy": f"{metrics.accuracy:.2%}",
                        "Precision (Macro)": f"{metrics.precision_macro:.4f}",
                        "Recall (Macro)": f"{metrics.recall_macro:.4f}",
                        "F1 Score (Macro)": f"{metrics.f1_macro:.4f}"
                    })
                st.dataframe(pd.DataFrame(class_metrics_data), width='stretch')
                st.plotly_chart(fig_bar, width='stretch')
                
            with sec_tab2:
                st.markdown("#### Routing Accuracies")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Retrieval Decision", f"{summary_obj.retrieval_decision_accuracy:.2%}")
                r2.metric("Escalation Accuracy", f"{summary_obj.escalation_accuracy:.2%}")
                r3.metric("Out-of-Scope Accuracy", f"{summary_obj.out_of_scope_accuracy:.2%}")
                r4.metric("Greeting Accuracy", f"{summary_obj.greeting_accuracy:.2%}")
                
            with sec_tab3:
                st.markdown("#### Ragas Scores")
                def fmt_ragas_metric(val):
                    return f"{val:.4f}" if val is not None else "N/A"
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
                    st.plotly_chart(fig_radar, width='stretch')
                with gc2:
                    st.plotly_chart(fig_hist, width='stretch')
                    
            with sec_tab5:
                st.markdown("#### System Performance Metrics")
                s1, s2, s3 = st.columns(3)
                s1.metric("Average Latency", f"{summary_obj.avg_latency:.2f}s")
                s2.metric("Average Chunks Retrieved", f"{summary_obj.avg_chunks:.2f}")
                s3.metric("Retrieval Skip Rate", f"{summary_obj.retrieval_skip_rate:.2%}")
                
            # Log & Ticket Inspector
            st.markdown("---")
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
                    "Latency": f"{r.latency:.2f}s"
                })
            st.dataframe(pd.DataFrame(table_rows), width='stretch')

            # Ticket Inspector
            st.markdown("### 🔍 Detailed Sample Inspector")
            selected_idx = st.selectbox(
                "Select Ticket index to inspect",
                options=range(1, len(results_list) + 1),
                key="inspect_evaluation_ticket_idx"
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

            # Export buttons
            st.markdown("---")
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
                    "Avg Latency": summary_obj.avg_latency
                }])
                st.download_button(
                    "Export Summary Metrics CSV",
                    data=csv_report.to_csv(index=False).encode('utf-8'),
                    file_name="evaluation_summary_metrics.csv",
                    mime="text/csv",
                    width='stretch'
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
                    "elapsed_time_seconds": summary_obj.elapsed_time_seconds
                }, indent=4)
                st.download_button(
                    "Export Summary Metrics JSON",
                    data=summary_json_str.encode('utf-8'),
                    file_name="evaluation_summary_metrics.json",
                    mime="application/json",
                    width='stretch'
                )
