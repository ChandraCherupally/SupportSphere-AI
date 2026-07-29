"""
Batch Processing UI module.
Executes a batch of support tickets from CSV/JSON, providing live metrics, cost, and latency tracking.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai.models import AgentResponse, SupportTicket
from src.graph.support_agent import SupportAgent
from src.streamlit.utils import AppConfig, patch_agent_configs

logger = logging.getLogger("SupportSphere")

DEMO_DATASET_PATH = Path("data/support_tickets/support_tickets5.csv")


def _get_active_dataset(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None) -> tuple[pd.DataFrame | None, str | None, bool]:
    """
    Resolve active dataset based on priority:
    1. Uploaded dataset
    2. Demo dataset (data/support_tickets/support_tickets5.csv)
    3. None (silently fallback)
    """
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            return df, uploaded_file.name, False
        except Exception as e:
            logger.error("Failed to parse uploaded file: %s", e)
            st.error(f"Error loading uploaded dataset: {e}")
            return None, None, False

    if DEMO_DATASET_PATH.exists():
        try:
            df = pd.read_csv(DEMO_DATASET_PATH)
            return df, DEMO_DATASET_PATH.name, True
        except Exception as e:
            logger.warning("Silent fallback: Failed to load demo dataset: %s", e)
            return None, None, False

    return None, None, False


def render_batch_processing(cfg: AppConfig) -> None:
    """Render the Batch Ticket Processing tab."""
    st.header("⚡ Batch Ticket Processing Engine")
    st.caption("Upload a dataset CSV or JSON file to process multiple tickets through SupportSphere AI.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Ticket Dataset", type=["csv", "json"])
    df, dataset_name, is_demo = _get_active_dataset(uploaded_file)

    if df is not None:
        try:
            demo_tag = " (Demo Dataset)" if is_demo else ""
            st.success(f"Loaded {len(df)} tickets from `{dataset_name}`{demo_tag}.")
            st.dataframe(df.head(5), width="stretch")

            if st.button("🚀 Start Batch Triage Run", width="stretch"):
                patch_agent_configs(cfg)
                agent = SupportAgent()

                results = []
                p_bar = st.progress(0.0)
                status_text = st.empty()
                t_start = time.time()

                for idx, row in df.iterrows():
                    p_bar.progress((idx + 1) / len(df))
                    issue_val = str(row.get("issue", row.get("Issue", ""))).strip()
                    subject_val = str(row.get("subject", row.get("Subject", ""))).strip()
                    company_val = str(row.get("company", row.get("Company", ""))).strip()
                    if company_val.lower() in ("nan", "none", "null", "unsupported co"):
                        company_val = ""

                    status_text.markdown(f"**Processing ticket {idx+1}/{len(df)}**: `{subject_val or issue_val[:30]}`")

                    ticket = SupportTicket(
                        issue=issue_val,
                        subject=subject_val,
                        company=company_val,
                        search_mode=cfg.search_mode.lower(),
                        reranker=cfg.reranker.lower(),
                        top_k=cfg.top_k,
                    )

                    try:
                        res: AgentResponse = agent.invoke(ticket)
                        raw_trace = getattr(res, "execution_trace", []) or []
                        cost = sum(r.get("cost", 0.0) for r in raw_trace if isinstance(r, dict))
                        ms = sum(r.get("latency_ms", r.get("latency", 0.0)) for r in raw_trace if isinstance(r, dict))
                        tokens = (
                            getattr(res, "decision_input_tokens", 0)
                            + getattr(res, "decision_output_tokens", 0)
                            + getattr(res, "generation_input_tokens", 0)
                            + getattr(res, "generation_output_tokens", 0)
                        )
                        warnings_str = ", ".join(res.warnings) if getattr(res, "warnings", None) else "None"

                        results.append(
                            {
                                # --- Business Output (Canonical Ground Truth Schema) ---
                                "Issue": issue_val,
                                "Subject": subject_val,
                                "Company": res.selected_company or company_val,
                                "Response": res.response.response,
                                "Product Area": res.response.product_area,
                                "Status": res.response.status,
                                "Request Type": res.response.request_type,
                                # --- Technical Metadata ---
                                "Intent": res.intent,
                                "Routing Strategy": res.routing_strategy,
                                "Verified Company": res.verified_company or res.selected_company,
                                "Retrieval Used": res.retriever_used or res.retrieval_required,
                                "Latency (ms)": round(ms, 1),
                                "Tokens": tokens,
                                "Cost ($)": round(cost, 6),
                                "Warnings": warnings_str,
                            }
                        )
                    except Exception as e:
                        logger.error("Batch ticket %d failed: %s", idx + 1, e)
                        results.append(
                            {
                                "Issue": issue_val,
                                "Subject": subject_val,
                                "Company": company_val,
                                "Response": f"Error: {e}",
                                "Product Area": "ERROR",
                                "Status": "Failed",
                                "Request Type": "invalid",
                                "Intent": "ERROR",
                                "Routing Strategy": "ERROR",
                                "Verified Company": "ERROR",
                                "Retrieval Used": False,
                                "Latency (ms)": 0.0,
                                "Tokens": 0,
                                "Cost ($)": 0.0,
                                "Warnings": f"Exception: {e}",
                            }
                        )

                total_time = time.time() - t_start
                status_text.success(f"Finished processing {len(df)} tickets in {total_time:.2f} seconds.")

                res_df = pd.DataFrame(results)

                st.markdown("### 📊 Batch Execution Results")
                st.caption("🏢 **Business Output** (Cols 1–7: Ground Truth Schema) &nbsp;│&nbsp; ⚙️ **Technical Metadata** (Cols 8–15: Observability)")

                # Configure columns with subtle visual grouping via tooltips and formatting
                col_config = {
                    "Issue": st.column_config.TextColumn("Issue", help="🏢 Business Output: Customer ticket issue text", width="medium"),
                    "Subject": st.column_config.TextColumn("Subject", help="🏢 Business Output: Ticket subject line", width="small"),
                    "Company": st.column_config.TextColumn("Company", help="🏢 Business Output: Target company domain", width="small"),
                    "Response": st.column_config.TextColumn("Response", help="🏢 Business Output: AI-generated customer response", width="large"),
                    "Product Area": st.column_config.TextColumn("Product Area", help="🏢 Business Output: Documented product area", width="small"),
                    "Status": st.column_config.TextColumn("Status", help="🏢 Business Output: Replied vs Escalated", width="small"),
                    "Request Type": st.column_config.TextColumn("Request Type", help="🏢 Business Output: Ticket request classification", width="small"),
                    "Intent": st.column_config.TextColumn("Intent", help="⚙️ Technical Metadata: Router intent classification", width="small"),
                    "Routing Strategy": st.column_config.TextColumn("Routing Strategy", help="⚙️ Technical Metadata: LangGraph execution strategy", width="small"),
                    "Verified Company": st.column_config.TextColumn("Verified Company", help="⚙️ Technical Metadata: NeMo guardrails verified company", width="small"),
                    "Retrieval Used": st.column_config.CheckboxColumn("Retrieval Used", help="⚙️ Technical Metadata: Whether vector search was triggered", width="small"),
                    "Latency (ms)": st.column_config.NumberColumn("Latency (ms)", help="⚙️ Technical Metadata: Total pipeline latency in ms", format="%.1f ms"),
                    "Tokens": st.column_config.NumberColumn("Tokens", help="⚙️ Technical Metadata: Total token usage", format="%d"),
                    "Cost ($)": st.column_config.NumberColumn("Cost ($)", help="⚙️ Technical Metadata: Inference cost in USD", format="$%.6f"),
                    "Warnings": st.column_config.TextColumn("Warnings", help="⚙️ Technical Metadata: System fallbacks or warnings", width="small"),
                }

                st.dataframe(res_df, column_config=col_config, width="stretch")

                # Metrics summary row
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Batch Cost", f"${res_df['Cost ($)'].sum():.6f}")
                with col_m2:
                    st.metric("Avg Ticket Latency", f"{res_df['Latency (ms)'].mean():.1f} ms")
                with col_m3:
                    st.metric("Success Rate", f"{(res_df['Status'] != 'Failed').mean() * 100:.1f}%")

                st.markdown("---")

                # Export CSV button preserving canonical schema ordering
                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results CSV (Canonical Schema)",
                    data=csv_bytes,
                    file_name="support_sphere_batch_results.csv",
                    mime="text/csv",
                    width="stretch",
                )

        except Exception as e:
            st.error(f"Error loading dataset: {e}")
