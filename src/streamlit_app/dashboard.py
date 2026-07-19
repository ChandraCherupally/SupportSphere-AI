"""
Dashboard, Single Ticket, and Batch Processing tabs for SupportSphere AI.

Implements a modern, professional, high-end SaaS user experience with clean layouts,
KPI cards, Mermaid pipeline flows, and progressive disclosures.
"""

from __future__ import annotations

import logging
import time
import pandas as pd
import streamlit as st

import src.config as src_config
from src.ai.models import SupportTicket
from src.graph.support_agent import SupportAgent
from src.streamlit_app.utils import AppConfig, patch_agent_configs, fmt_cost

logger = logging.getLogger(__name__)


def show_active_retrieval_config(cfg: AppConfig, actual_retrieved: int, retrieval_trace: dict | None = None) -> None:
    """Renders the active retrieval configuration summary panel.

    Args:
        cfg: Current AppConfig.
        actual_retrieved: Number of chunks actually retrieved.
        retrieval_trace: Optional trace metadata of retrieval steps.
    """
    st.markdown("---")
    st.markdown("#### ⚙️ Active Retrieval Configuration")
    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
    rc1.write(f"**Provider**: `{cfg.provider}`")
    rc2.write(f"**Model**: `{cfg.selected_model}`")
    rc3.write(f"**Search Mode**: `{cfg.search_mode}`")
    rc4.write(f"**Reranker**: `{cfg.reranker}`")
    rc5.write(f"**Top-K (Configured)**: `{cfg.top_k}`")
    rc6.write(f"**Retrieved Chunks**: `{actual_retrieved}`")

    if actual_retrieved != cfg.top_k:
        st.warning(f"⚠️ Retrieved chunk count ({actual_retrieved}) does not match configured Top-K ({cfg.top_k}).")


def render_dashboard(cfg: AppConfig) -> None:
    """Render the Dashboard overview tab in SaaS style.

    Args:
        cfg: Current AppConfig.
    """
    # Header card
    st.markdown(
        """
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 25px; border-radius: 12px; margin-bottom: 25px;">
            <h1 style="margin: 0; font-size: 2.2em; color: #0f172a;">🤖 SupportSphere AI Platform</h1>
            <p style="margin: 5px 0 0 0; font-size: 1.1em; color: #64748b;">
                Production-quality multi-domain support triage, hybrid semantic context search, and guardrailed generation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Executive Summary & System Status side-by-side
    col_summary, col_status = st.columns([1.6, 1])

    with col_summary:
        st.markdown("### 🏢 Core Architecture Summary")
        st.markdown(
            """
            - ✓ **LangGraph Workflow**: Orchestrates step-by-step stateful ticket routing, validation, and execution.
            - ✓ **Hybrid Retrieval**: Integrates sparse BM25 and dense Vector search for context coverage.
            - ✓ **Stage-Based Model Routing**: Dynamically directs traffic to optimal models per workflow stage.
            - ✓ **Guardrails**: Performs automated input safety validation and output validation checks.
            - ✓ **RAGAS Evaluation**: Automates metrics computation for precision, recall, faithfulness, correctness.
            - ✓ **Experiment Tracking**: Snapshots configuration, results, and pricing per run for full reproducibility.
            """
        )

    with col_status:
        with st.container(border=True):
            st.markdown("### ⚡ System Status")
            st.markdown("🟢 **Status**: `Active`")
            st.markdown(f"🗄 **Vector Database**: `{src_config.VECTOR_DB.title() if hasattr(src_config, 'VECTOR_DB') else 'Pinecone'}`")
            st.markdown(f"🧠 **Embeddings**: `{src_config.EMBEDDING_MODEL}`")
            st.markdown("🏢 **Supported Companies**: `Visa, Claude, HackerRank`")

    st.markdown("---")

    # 2. Engine Selections in a Grid of 4 Bordered Cards
    st.markdown("### ⚙️ Current Engine Selection")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        with st.container(border=True):
            st.markdown("##### 🔀 Decision Engine")
            st.write(f"**Provider**: `{cfg.decision_provider}`")
            st.write(f"**Model**: `{cfg.decision_model}`")
    with col_e2:
        with st.container(border=True):
            st.markdown("##### ✍️ Generation Engine")
            st.write(f"**Provider**: `{cfg.generation_provider}`")
            st.write(f"**Model**: `{cfg.generation_model}`")
    with col_e3:
        with st.container(border=True):
            st.markdown("##### 🧠 Embedding Engine")
            st.write(f"**Provider**: `{src_config.EMBEDDING_PROVIDER.title()}`")
            st.write(f"**Model**: `{src_config.EMBEDDING_MODEL}`")
    with col_e4:
        with st.container(border=True):
            st.markdown("##### 🔍 Retrieval & Rank")
            st.write(f"**Mode**: `{cfg.search_mode}`")
            st.write(f"**Reranker**: `{cfg.reranker}`")
            st.write(f"**Top-K**: `{cfg.top_k}`")

    st.markdown("---")

    # Resolve dynamic retrieval labels based on search mode
    search_mode_lower = cfg.search_mode.lower()
    if search_mode_lower == "hybrid":
        retrieval_text = "Hybrid Retrieval\\n(BM25 + Dense Vector)"
    elif search_mode_lower == "bm25":
        retrieval_text = "BM25 Retrieval\\n(Sparse Keyword Search)"
    else:
        retrieval_text = "Dense Vector Retrieval\\n(Pinecone Semantic Search)"

    # Append reranker if enabled
    reranker_lower = cfg.reranker.lower()
    if reranker_lower != "none":
        retrieval_text += f"\\n{cfg.reranker}"

    # Resolve dynamic model labels
    decision_model_label = f"({cfg.decision_provider} • {cfg.decision_model})"
    generation_model_label = f"({cfg.generation_provider} • {cfg.generation_model})"

    # 3. Centered Workflow Diagram
    st.markdown("<h3 style='text-align: center;'>🔄 Agent State Workflow Diagram</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Visual representation of the active multi-stage state machine and routing execution path.</p>", unsafe_allow_html=True)
    
    col_w_left, col_w_mid, col_w_right = st.columns([1, 8, 1])
    with col_w_mid:
        st.markdown(
            f"""
            ```mermaid
            graph TD
                ST[📥 SupportTicket] --> IV[🛡️ Input Validation]
                IV --> DG["🔀 Decision Gate\\n{decision_model_label}"]
                DG --> DR{{🔍 Retrieval Required?}}
                
                DR -->|No| GEN["✍️ Generation\\n{generation_model_label}"]
                DR -->|Yes| HR["{retrieval_text}"]
                
                HR --> RV[🛡️ Retrieval Validation]
                RV --> GEN
                
                GEN --> OG[🛡️ Output Guardrails]
                OG --> AR[📤 AgentResponse]
                
                style ST fill:none,stroke:#94a3b8,stroke-width:2px
                style IV fill:none,stroke:#3b82f6,stroke-width:2px
                style DG fill:none,stroke:#0ea5e9,stroke-width:2px
                style DR fill:none,stroke:#0ea5e9,stroke-width:2px
                style HR fill:none,stroke:#10b981,stroke-width:2px
                style RV fill:none,stroke:#047857,stroke-width:2px
                style GEN fill:none,stroke:#f97316,stroke-width:2px
                style OG fill:none,stroke:#ef4444,stroke-width:2px
                style AR fill:none,stroke:#8b5cf6,stroke-width:2px
            ```
            """
        )


def validate_configuration(cfg: AppConfig) -> None:
    """Validate configuration settings before running inference."""
    from src.billing.catalog import get_supported_models
    supported = get_supported_models()
    
    # 1. Validate Decision Provider & Model
    dp_key = cfg.decision_provider.title() if cfg.decision_provider.lower() != "openai" else "OpenAI"
    if dp_key not in supported:
        raise ValueError(f"Decision Provider '{cfg.decision_provider}' is not supported.")
    if cfg.decision_model not in supported[dp_key]:
        raise ValueError(
            f"Selected Decision Model '{cfg.decision_model}' is not available for {dp_key}.\n"
            f"Please select one of the supported models: {', '.join(supported[dp_key])}"
        )
        
    # 2. Validate Generation Provider & Model
    gp_key = cfg.generation_provider.title() if cfg.generation_provider.lower() != "openai" else "OpenAI"
    if gp_key not in supported:
        raise ValueError(f"Generation Provider '{cfg.generation_provider}' is not supported.")
    if cfg.generation_model not in supported[gp_key]:
        raise ValueError(
            f"Selected Generation Model '{cfg.generation_model}' is not available for {gp_key}.\n"
            f"Please select one of the supported models: {', '.join(supported[gp_key])}"
        )

    # 3. Validate API Keys for selected providers
    # Check Decision Provider Key
    dp_lower = cfg.decision_provider.lower()
    if dp_lower == "google" and not cfg.google_key.strip():
        raise ValueError("Google API Key is required for the Decision stage. Please set it in the sidebar.")
    if dp_lower == "openai" and not cfg.openai_key.strip():
        raise ValueError("OpenAI API Key is required for the Decision stage. Please set it in the sidebar.")
    if dp_lower == "groq" and not cfg.groq_key.strip():
        raise ValueError("Groq API Key is required for the Decision stage. Please set it in the sidebar.")
        
    # Check Generation Provider Key
    gp_lower = cfg.generation_provider.lower()
    if gp_lower == "google" and not cfg.google_key.strip():
        raise ValueError("Google API Key is required for the Generation stage. Please set it in the sidebar.")
    if gp_lower == "openai" and not cfg.openai_key.strip():
        raise ValueError("OpenAI API Key is required for the Generation stage. Please set it in the sidebar.")
    if gp_lower == "groq" and not cfg.groq_key.strip():
        raise ValueError("Groq API Key is required for the Generation stage. Please set it in the sidebar.")


def render_single_ticket(cfg: AppConfig) -> None:
    """Render the Single Ticket triage tab using a polished 2-column SaaS layout.

    Args:
        cfg: Current AppConfig.
    """
    st.markdown("## 🎫 Single Ticket Triage")
    st.caption("Test the pipeline end-to-end. Process custom user inquiries and inspect routing, billing, and contexts.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("### 📥 Ticket Input")
        with st.form("single_ticket_form"):
            issue = st.text_area(
                "Customer Issue Inquiry", 
                value="Hi, please pause our subscription. We have stopped all hiring efforts for now.",
                height=160
            )
            subject = st.text_input("Subject Line", value="Subscription pause")
            company = st.selectbox("Client Company", options=["HackerRank", "Claude", "Visa", "None"])
            submit_btn = st.form_submit_button("Run Support Triage Pipeline", type="primary", use_container_width=True)

    with col_right:
        st.markdown("### 📊 Prediction Summary")
        
        # We store run outputs in session state to maintain views across options toggles
        if submit_btn:
            if not issue.strip():
                st.error("Please enter a valid customer inquiry text.")
                return

            # Perform validation before execution
            try:
                validate_configuration(cfg)
            except ValueError as val_err:
                st.error(f"❌ Configuration Validation Error:\n\n{val_err}")
                return

            with st.spinner("Executing support agent agentic nodes..."):
                patch_agent_configs(cfg)
                agent = SupportAgent()
                ticket = SupportTicket(
                    issue=issue,
                    subject=subject,
                    company=company if company != "None" else "",
                    reranker=cfg.reranker.lower(),
                    search_mode=cfg.search_mode.lower(),
                )

                t_start = time.perf_counter()
                try:
                    res = agent.invoke(ticket)
                    t_elapsed = time.perf_counter() - t_start
                    
                    # Compute costs
                    from src.billing.calculator import BillingCalculator
                    from src.billing.models import TokenUsage
                    
                    billing_calc = BillingCalculator(
                        decision_provider=cfg.decision_provider.lower(),
                        decision_model=cfg.decision_model,
                        generation_provider=cfg.generation_provider.lower(),
                        generation_model=cfg.generation_model,
                        embedding_provider=getattr(src_config, "EMBEDDING_PROVIDER", "google"),
                        embedding_model=getattr(src_config, "EMBEDDING_MODEL", "gemini-embedding-001"),
                        reranker=cfg.reranker.lower(),
                    )
                    
                    decision_usage = TokenUsage(
                        input_tokens=res.decision_input_tokens,
                        output_tokens=res.decision_output_tokens,
                    )
                    generation_usage = TokenUsage(
                        input_tokens=res.generation_input_tokens,
                        output_tokens=res.generation_output_tokens,
                    )
                    
                    ticket_cost = billing_calc.calculate_ticket_cost(
                        decision_usage=decision_usage,
                        generation_usage=generation_usage,
                        embedding_tokens=0,
                        retrieval_required=res.retrieval_required,
                    )
                    
                    # Persist run state
                    st.session_state.last_run_result = res
                    st.session_state.last_run_latency = t_elapsed
                    st.session_state.last_run_cost = ticket_cost
                    st.success("Triage processed successfully!")
                except Exception as e:
                    logger.error("Single ticket pipeline failed: %s", e, exc_info=True)
                    st.error(f"Execution failed: {e}")

        # Draw Prediction Metrics KPI cards if results are loaded
        if "last_run_result" in st.session_state:
            res = st.session_state.last_run_result
            t_elapsed = st.session_state.last_run_latency
            ticket_cost = st.session_state.last_run_cost

            # Check company match and display warnings/alerts
            if not getattr(res, "company_match", True):
                st.warning(
                    f"⚠️ **Company Mismatch Detected**\n\n"
                    f"• **Selected**: `{getattr(res, 'selected_company', '')}`\n"
                    f"• **Detected**: `{getattr(res, 'detected_company', '')}`\n"
                    f"• **Retrieval will use**: **{getattr(res, 'verified_company', '')}**\n"
                    f"• **Confidence**: {getattr(res, 'company_confidence', 1.0):.1%}"
                )
            elif getattr(res, "detected_company", "").lower() == "unknown":
                st.warning(
                    f"⚠️ **Company Detection Warning**\n\n"
                    f"Unable to confidently identify the company. Defaulting to selected company: **{getattr(res, 'selected_company', '')}**."
                )

            # Render theme-aware status colors and metric cards
            status_val = res.response.status
            if status_val.lower() == "resolved":
                status_display = f"🟢 **{status_val.title()}**"
            elif status_val.lower() == "escalated":
                status_display = f"🔴 **{status_val.title()}**"
            else:
                status_display = f"🟡 **{status_val.title()}**"

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                with st.container(border=True):
                    st.caption("🏷️ Request Type")
                    st.markdown(f"**{res.response.request_type.title()}**")
            with col_b:
                with st.container(border=True):
                    st.caption("📂 Product Area")
                    st.markdown(f"**{res.response.product_area.title()}**")
            with col_c:
                with st.container(border=True):
                    st.caption("🎯 Status")
                    st.markdown(status_display)

            col_d, col_e, col_f = st.columns(3)
            with col_d:
                with st.container(border=True):
                    st.caption("🛡️ Confidence")
                    st.markdown(f"**{res.confidence:.1%}**")
            with col_e:
                with st.container(border=True):
                    st.caption("⏱️ Latency")
                    st.markdown(f"**{t_elapsed:.2f}s**")
            with col_f:
                with st.container(border=True):
                    st.caption("💳 Total Cost")
                    st.markdown(f"**{fmt_cost(ticket_cost.total_cost)}**")
        else:
            st.info("ℹ️ Fill out the ticket inquiry form on the left and submit to view prediction summaries.")

    # -----------------------------------------------------------------------
    # Tabbed Interface Below the Columns
    # -----------------------------------------------------------------------
    if "last_run_result" in st.session_state:
        res = st.session_state.last_run_result
        t_elapsed = st.session_state.last_run_latency
        ticket_cost = st.session_state.last_run_cost

        st.markdown("---")
        tab_resp, tab_perf, tab_retrieval = st.tabs([
            "✍️ Generated Response",
            "📊 Token Billing Details",
            "🔍 Retrieval Pipeline"
        ])

        with tab_resp:
            st.markdown("### 🤖 Final Output Response")
            
            # CSS for SaaS generated response box
            st.markdown(
                """
                <style>
                .saas-response-box {
                    border-left: 4px solid #10b981;
                    padding-left: 20px;
                    margin-bottom: 20px;
                    font-size: 1.1em;
                    line-height: 1.6;
                }
                .saas-response-box p {
                    margin-bottom: 0.8em;
                }
                .saas-response-box ul, .saas-response-box ol {
                    margin-bottom: 0.8em;
                    padding-left: 24px;
                }
                .saas-response-box li {
                    margin-bottom: 0.4em;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <div class="saas-response-box">

{res.response.response}

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### Classification Justification")
            st.info(res.response.justification)

        with tab_perf:
            st.markdown("### 💳 Token Usage Billing Breakdown")
            
            # Decision metrics vs Generation metrics
            col_b1, col_b2, col_b3 = st.columns(3)
            
            with col_b1:
                with st.container(border=True):
                    st.markdown("##### 🔀 Decision Triage")
                    st.write(f"**Input Tokens**: `{ticket_cost.decision_input_tokens}`")
                    st.write(f"**Output Tokens**: `{ticket_cost.decision_output_tokens}`")
                    st.write(f"**Stage Cost**: `{fmt_cost(ticket_cost.decision_cost)}`")
                    
            with col_b2:
                with st.container(border=True):
                    st.markdown("##### ✍️ Generation Stage")
                    st.write(f"**Input Tokens**: `{ticket_cost.generation_input_tokens}`")
                    st.write(f"**Output Tokens**: `{ticket_cost.generation_output_tokens}`")
                    st.write(f"**Stage Cost**: `{fmt_cost(ticket_cost.generation_cost)}`")
                    
            with col_b3:
                total_in = ticket_cost.decision_input_tokens + ticket_cost.generation_input_tokens
                total_out = ticket_cost.decision_output_tokens + ticket_cost.generation_output_tokens
                with st.container(border=True):
                    st.markdown("##### 📈 Aggregated Totals")
                    st.write(f"**Total Tokens**: `{total_in + total_out}`")
                    st.write(f"**Ratio (In/Out)**: `{total_in}:{total_out}`")
                    st.write(f"**Combined Cost**: **{fmt_cost(ticket_cost.total_cost)}**")

        with tab_retrieval:
            st.markdown("### ⚙️ Retrieval Step Analysis")
            
            if not res.retrieval_required:
                st.info("ℹ️ Retrieval stage was bypassed. The Decision Gate routed this inquiry directly without matching documents.")
            else:
                trace = res.retrieval_trace
                
                # Render Company Detection info if available
                st.markdown("#### 🕵️ Company Detection & Routing")
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    with st.container(border=True):
                        st.caption("Selected Company")
                        st.write(getattr(res, "selected_company", "N/A") or "N/A")
                with col_c2:
                    with st.container(border=True):
                        st.caption("Detected Company")
                        st.write(getattr(res, "detected_company", "N/A") or "N/A")
                with col_c3:
                    with st.container(border=True):
                        st.caption("Verified Company")
                        st.write(getattr(res, "verified_company", "N/A") or "N/A")
                with col_c4:
                    with st.container(border=True):
                        st.caption("Confidence")
                        st.write(f"{getattr(res, 'company_confidence', 1.0):.1%}")
                
                if getattr(res, "routing_reason", ""):
                    st.info(f"**Routing Reason**: {getattr(res, 'routing_reason', '')}")

                st.markdown("#### ⚙️ Active Retrieval Configuration")
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    with st.container(border=True):
                        st.caption("Search Mode")
                        st.write(trace.get("search_mode", "N/A").upper())
                with col_t2:
                    with st.container(border=True):
                        st.caption("Collection Used")
                        st.write(trace.get("collection_used", "N/A"))
                with col_t3:
                    with st.container(border=True):
                        st.caption("Top-K")
                        st.write(str(trace.get("top_k", "N/A")))

                # Flow chart rendering
                st.markdown("#### 🔄 Pipeline Selection & Deduplication Flow")
                st.markdown(
                    f"""
                    ```mermaid
                    graph LR
                        Mode["🔍 Mode: {str(trace.get('search_mode', 'N/A')).upper()}"] --> Cand["📥 Candidates: {trace.get('merged_count', 0)}"]
                        Cand --> Dedup["🧹 Unique: {trace.get('unique_count', 0)}"]
                        Dedup --> TopK["🎯 Reranked: {trace.get('reranked_count', 0)}"]
                        TopK --> Final["📦 Top-K: {trace.get('final_returned_count', 0)}"]
                        
                        style Mode fill:none,stroke:#94a3b8,stroke-width:1px
                        style Cand fill:none,stroke:#10b981,stroke-width:2px
                        style Dedup fill:none,stroke:#f59e0b,stroke-width:2px
                        style TopK fill:none,stroke:#ec4899,stroke-width:2px
                        style Final fill:none,stroke:#0ea5e9,stroke-width:2px
                    ```
                    """
                )

                st.markdown("#### 📄 Matching Document Cards")
                for i, chunk in enumerate(res.retrieved_chunks):
                    with st.container(border=True):
                        st.markdown(f"##### 📄 {chunk.title}")
                        st.markdown(
                            f"**Source URL**: {chunk.url} | "
                            f"**Domain**: {chunk.company} | "
                            f"**Product Area**: {chunk.product_area} | "
                            f"**Sim Score**: `{chunk.score:.4f}`"
                        )
                        with st.expander("Expand Source Text Reference", expanded=False):
                            st.markdown(f"```\n{chunk.text}\n```")


def render_batch(cfg: AppConfig) -> None:
    """Render the Batch Processing tab using structured dashboard summary cards.

    Args:
        cfg: Current AppConfig.
    """
    st.markdown("## 📦 Batch Support Triage")
    st.caption("Process datasets containing multiple inquiries to benchmark accuracy, speeds, and total costs.")
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_batch = st.file_uploader(
        "Upload Inquiries CSV (must contain 'Issue' or 'Subject' columns)", 
        type=["csv"], 
        key="uploaded_batch_file"
    )

    if uploaded_batch is not None:
        try:
            df = pd.read_csv(uploaded_batch)
            st.session_state.batch_df = df
            st.success(f"Successfully loaded CSV containing {len(df)} inquiries.")
            st.dataframe(df.head(3), width='stretch')
        except Exception as e:
            st.error(f"Failed to parse upload CSV: {e}")

    if st.session_state.get("batch_df") is not None:
        batch_btn = st.button("Execute Batch Processing Job", key="run_batch_triage_btn", type="primary", use_container_width=True)

        if batch_btn:
            # Perform validation before execution
            try:
                validate_configuration(cfg)
            except ValueError as val_err:
                st.error(f"❌ Configuration Validation Error:\n\n{val_err}")
                return
            df = st.session_state.batch_df
            patch_agent_configs(cfg)

            agent = SupportAgent()
            outputs = []

            from src.billing.calculator import BillingCalculator
            from src.billing.models import TokenUsage
            
            billing_calc = BillingCalculator(
                decision_provider=cfg.decision_provider.lower(),
                decision_model=cfg.decision_model,
                generation_provider=cfg.generation_provider.lower(),
                generation_model=cfg.generation_model,
                embedding_provider=getattr(src_config, "EMBEDDING_PROVIDER", "google"),
                embedding_model=getattr(src_config, "EMBEDDING_MODEL", "gemini-embedding-001"),
                reranker=cfg.reranker.lower(),
            )

            p_bar = st.progress(0)
            status_text = st.empty()
            t_start = time.time()

            for idx, row in df.iterrows():
                p_bar.progress((idx + 1) / len(df))
                status_text.text(f"Processing ticket {idx + 1}/{len(df)}...")

                issue_val = str(row.get("Issue", row.get("issue", ""))).strip()
                subj_val = str(row.get("Subject", row.get("subject", ""))).strip()
                comp_val = str(row.get("Company", row.get("company", ""))).strip()
                if comp_val.lower() in {"nan", "none"} or not comp_val:
                    comp_val = ""

                ticket = SupportTicket(
                    issue=issue_val, subject=subj_val,
                    company=comp_val, reranker=cfg.reranker.lower(),
                    search_mode=cfg.search_mode.lower()
                )

                try:
                    res = agent.invoke(ticket)

                    decision_usage = TokenUsage(
                        input_tokens=res.decision_input_tokens,
                        output_tokens=res.decision_output_tokens,
                    )
                    generation_usage = TokenUsage(
                        input_tokens=res.generation_input_tokens,
                        output_tokens=res.generation_output_tokens,
                    )
                    ticket_cost = billing_calc.calculate_ticket_cost(
                        decision_usage=decision_usage,
                        generation_usage=generation_usage,
                        embedding_tokens=0,
                        retrieval_required=res.retrieval_required,
                    )

                    outputs.append({
                        "Issue": issue_val, "Subject": subj_val,
                        "Company": comp_val or "None",
                        "Predicted Request Type": res.response.request_type,
                        "Predicted Product Area": res.response.product_area,
                        "Predicted Status": res.response.status,
                        "Generated Response": res.response.response,
                        "Num Chunks": res.num_chunks,
                        "Input Tokens": ticket_cost.total_input_tokens,
                        "Output Tokens": ticket_cost.total_output_tokens,
                        "Total Tokens": ticket_cost.total_tokens,
                        "Cost ($)": ticket_cost.total_cost,
                    })
                except Exception as e:
                    logger.error("Failed prediction on batch item %d: %s", idx, e)
                    outputs.append({
                        "Issue": issue_val, "Subject": subj_val, "Company": comp_val or "None",
                        "Predicted Request Type": "invalid", "Predicted Product Area": "General",
                        "Predicted Status": "Replied", "Generated Response": "Prediction failed.",
                        "Num Chunks": 0,
                        "Input Tokens": 0,
                        "Output Tokens": 0,
                        "Total Tokens": 0,
                        "Cost ($)": 0.0,
                    })

            t_elapsed = time.time() - t_start
            p_bar.empty()
            status_text.empty()

            out_df = pd.DataFrame(outputs)
            st.session_state.batch_outputs = out_df
            st.session_state.batch_runtime = t_elapsed

        # Display Summary Dashboard Card and Table below if outputs exist
        if st.session_state.get("batch_outputs") is not None:
            out_df = st.session_state.batch_outputs
            t_elapsed = st.session_state.get("batch_runtime", 0.0)

            st.markdown("### 🏆 Batch Job Run Dashboard")
            
            total_cost_val = out_df["Cost ($)"].sum()
            avg_cost_val = out_df["Cost ($)"].mean()
            avg_tokens_val = out_df["Total Tokens"].mean()
            avg_chunks = int(out_df["Num Chunks"].mean()) if len(out_df) > 0 else 0

            # Beautiful summary metric dashboard grid
            st.markdown(
                f"""
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 25px;">
                    <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; text-align: center;">
                        <span style="font-size: 0.85em; color: #64748b;">🏷️ Total Tickets</span><br/>
                        <strong style="font-size: 1.3em; color: #0f172a;">{len(out_df)}</strong>
                    </div>
                    <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; text-align: center;">
                        <span style="font-size: 0.85em; color: #64748b;">⏱️ Runtime</span><br/>
                        <strong style="font-size: 1.3em; color: #0f172a;">{t_elapsed:.2f}s</strong>
                    </div>
                    <div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; border: 1px solid #bbf7d0; text-align: center;">
                        <span style="font-size: 0.85em; color: #166534;">💳 Total Cost</span><br/>
                        <strong style="font-size: 1.3em; color: #15803d;">{fmt_cost(total_cost_val)}</strong>
                    </div>
                    <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; text-align: center;">
                        <span style="font-size: 0.85em; color: #64748b;">💸 Avg Cost / Ticket</span><br/>
                        <strong style="font-size: 1.3em; color: #0f172a;">{fmt_cost(avg_cost_val)}</strong>
                    </div>
                    <div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; text-align: center;">
                        <span style="font-size: 0.85em; color: #64748b;">🧬 Avg Tokens / Ticket</span><br/>
                        <strong style="font-size: 1.3em; color: #0f172a;">{avg_tokens_val:.1f}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            show_active_retrieval_config(cfg, actual_retrieved=avg_chunks)

            st.markdown("### 📊 Interactive Results Table")
            st.dataframe(out_df, width='stretch')

            csv_data = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Predictions CSV Report", data=csv_data,
                file_name="triage_predictions.csv", mime="text/csv",
                width='stretch'
            )
