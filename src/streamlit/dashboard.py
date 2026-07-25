"""
Streamlit Dashboard Module.
Displays current architecture (Decision Engine + Generation Engine) and live system status.
"""

from __future__ import annotations

import streamlit as st
import src.config as src_config
from src.streamlit.utils import AppConfig


def render_dashboard(cfg: AppConfig) -> None:
    """Render the Architecture & System Overview tab."""
    st.title("SupportSphere AI")
    st.subheader("Production Multi-Domain Support Triage Platform")
    st.markdown(
        "Stateful LangGraph orchestration with a NeMo-powered Decision Engine, "
        "Hybrid Retrieval, and grounded AI response generation."
    )
    st.markdown("---")

    # Overview Cards
    col_arch, col_status = st.columns([2, 1])

    with col_arch:
        with st.container(border=True):
            st.markdown("### 📄 Current Architecture Overview")
            st.markdown("""
            - **LLM #1: Decision Engine**: NeMo Guardrails performs safety, intent, company detection, routing strategy, and retrieval decision.
            - **LLM #2: Generation Engine**: Generates grounded support responses.
            - **Retrieval Engine**: Hybrid BM25 + Dense Vector retrieval with configurable reranking.
            - **Validation**: Deterministic retrieval validation and response validation without additional LLM calls.
            - **LangGraph**: Stateful orchestration with complete LangSmith observability.
            """)

    with col_status:
        with st.container(border=True):
            st.markdown("### ⚡ System Status")
            st.markdown("🟢 **Status**: `Active`")
            st.markdown(f"🟣 **Vector Database**: `{getattr(src_config, 'VECTOR_DB', 'Pinecone').title()}`")
            st.markdown(f"🧠 **Decision Engine**: `{cfg.decision_provider} / {cfg.decision_model}`")
            st.markdown(f"✨ **Generation Engine**: `{cfg.generation_provider} / {cfg.generation_model}`")
            st.markdown(f"🔍 **Search Mode**: `{cfg.search_mode}` (Top-{cfg.top_k})")

    st.markdown("---")

    # Production LangGraph Workflow Diagram
    st.markdown("### 🌐 Production Workflow")
    st.markdown("""
    ```mermaid
    graph TD
        Ticket["📥 Ticket Input"] --> DecisionEngine["🧠 Decision Engine (LLM #1 + NeMo)"]
        
        DecisionEngine -->|Blocked / Unsafe| SafeRefusal["🛡️ Safe Refusal Template"]
        DecisionEngine -->|Safe & Support Request| Retrieval["🔍 Retrieval Engine<br/>(BM25 + Dense Vector + Optional Reranker)"]
        DecisionEngine -->|Safe & Direct Template| TemplateResp["📋 Static Template Response"]
        
        Retrieval --> RetrievalValidation["🧹 Retrieval Validation (Rules)"]
        RetrievalValidation --> Generation["✨ Generation Engine (LLM #2)"]
        Generation --> ResponseValidation["🛡️ Response Validation (Rules)"]
        
        SafeRefusal --> END["🏁 END"]
        ResponseValidation --> END
        TemplateResp --> END

        style DecisionEngine fill:#dbeafe,stroke:#2563eb,stroke-width:2px
        style Generation fill:#dcfce7,stroke:#16a34a,stroke-width:2px
        style RetrievalValidation fill:#fef3c7,stroke:#d97706,stroke-width:2px
        style ResponseValidation fill:#fef3c7,stroke:#d97706,stroke-width:2px
    ```
    """)
