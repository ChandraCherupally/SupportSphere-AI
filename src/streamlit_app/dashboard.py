"""
Dashboard, Single Ticket, and Batch Processing tabs for SupportSphere AI.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import streamlit as st

from src.ai.models import SupportTicket
from src.graph.support_agent import SupportAgent
from src.streamlit_app.utils import AppConfig, patch_agent_configs

logger = logging.getLogger(__name__)


def show_active_retrieval_config(cfg: AppConfig, actual_retrieved: int) -> None:
    """Renders the active retrieval configuration summary panel.

    Args:
        cfg: Current AppConfig.
        actual_retrieved: Number of chunks actually retrieved.
    """
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
    if actual_retrieved > cfg.top_k:
        logger.warning("Runtime validation mismatch: actual_retrieved (%d) > top_k (%d)", actual_retrieved, cfg.top_k)


def render_dashboard(cfg: AppConfig) -> None:
    """Render the Dashboard overview tab.

    Args:
        cfg: Current AppConfig.
    """
    st.title("SupportSphere AI Dashboard")
    st.write("Welcome to the SupportSphere AI enterprise multi-domain support triage and evaluation system.")

    st.markdown("### Active Configuration Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM Provider", cfg.provider)
    col2.metric("LLM Model", cfg.selected_model)
    col3.metric("Retrieval Search Mode", cfg.search_mode)
    col4.metric("Reranker", cfg.reranker)

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
        st.write("**Framework Version**: v1.3.0")


def render_single_ticket(cfg: AppConfig) -> None:
    """Render the Single Ticket triage tab.

    Args:
        cfg: Current AppConfig.
    """
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
            return

        with st.spinner("Processing inquiry through SupportSphere AI..."):
            patch_agent_configs(cfg)
            agent = SupportAgent()
            ticket = SupportTicket(
                issue=issue,
                subject=subject,
                company=company if company != "None" else "",
                reranker=cfg.reranker.lower(),
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

                show_active_retrieval_config(cfg, actual_retrieved=res.num_chunks)

                st.markdown("### Generated Support Response")
                st.write(res.response.response)

                st.markdown("#### Classification Justification")
                st.info(res.response.justification)

                st.markdown("### Retrieved Context Reference Chunks")
                for i, chunk in enumerate(res.retrieved_chunks):
                    with st.expander(f"Chunk #{i+1} (Source: {chunk.company} | Area: {chunk.product_area} | Score: {chunk.score:.4f})"):
                        st.write(f"**URL**: [{chunk.url}]({chunk.url})")
                        st.markdown(f"**Text Reference**:\n```\n{chunk.text}\n```")

            except Exception as e:
                logger.error("Failed to triage single ticket: %s", e, exc_info=True)
                st.error(f"Processing failed: {e}")


def render_batch(cfg: AppConfig) -> None:
    """Render the Batch Processing tab.

    Args:
        cfg: Current AppConfig.
    """
    st.title("Batch Support Triage")
    st.write("Upload a CSV file containing batch support tickets to parse and class-label them.")

    uploaded_batch = st.file_uploader("Upload Tickets CSV", type=["csv"], key="uploaded_batch_file")

    if uploaded_batch is not None:
        try:
            df = pd.read_csv(uploaded_batch)
            st.session_state.batch_df = df
            st.success(f"Loaded batch file containing {len(df)} rows.")
            st.dataframe(df.head(5))
        except Exception as e:
            st.error(f"Failed to load batch file: {e}")

    if st.session_state.get("batch_df") is not None:
        batch_btn = st.button("Run Batch Triage", key="run_batch_triage_btn")

        if batch_btn:
            df = st.session_state.batch_df
            patch_agent_configs(cfg)

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
                if comp_val.lower() in {"nan", "none"} or not comp_val:
                    comp_val = ""

                ticket = SupportTicket(
                    issue=issue_val, subject=subj_val,
                    company=comp_val, reranker=cfg.reranker.lower()
                )

                try:
                    res = agent.invoke(ticket)
                    outputs.append({
                        "Issue": issue_val, "Subject": subj_val,
                        "Company": comp_val or "None",
                        "Predicted Request Type": res.response.request_type,
                        "Predicted Product Area": res.response.product_area,
                        "Predicted Status": res.response.status,
                        "Generated Response": res.response.response,
                        "Num Chunks": res.num_chunks,
                    })
                except Exception as e:
                    logger.error("Failed prediction on batch item %d: %s", idx, e)
                    outputs.append({
                        "Issue": issue_val, "Subject": subj_val, "Company": comp_val or "None",
                        "Predicted Request Type": "invalid", "Predicted Product Area": "General",
                        "Predicted Status": "Replied", "Generated Response": "Prediction failed.",
                        "Num Chunks": 0,
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

            avg_chunks = int(out_df["Num Chunks"].mean()) if len(out_df) > 0 else 0
            show_active_retrieval_config(cfg, actual_retrieved=avg_chunks)

            st.markdown("### Interactive Results Table")
            st.dataframe(out_df, width="stretch")

            csv_data = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV", data=csv_data,
                file_name="triage_predictions.csv", mime="text/csv"
            )
