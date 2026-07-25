"""
Streamlit Sidebar UI module.
Configures active Decision Engine (LLM #1), Generation Engine (LLM #2), Retrieval Configuration, and API Credentials.
"""

from __future__ import annotations

import os
import streamlit as st

import src.config as src_config
from src.billing.catalog import get_supported_models
from src.streamlit.utils import AppConfig


def render_sidebar() -> AppConfig:
    """Render the sidebar and return the active AppConfig."""
    with st.sidebar:
        st.title("SupportSphere AI")
        st.caption("⚙️ Runtime Configuration")
        st.markdown("---")

        supported_models = get_supported_models()
        providers = list(supported_models.keys())

        def _find_idx(items: list[str], target: str) -> int:
            t = target.lower()
            for i, item in enumerate(items):
                if item.lower() == t:
                    return i
            return 0

        # ---------------------------------------------------------------------
        # 1. LLM #1: Decision Engine
        # ---------------------------------------------------------------------
        def_dec_prov = getattr(src_config, "DECISION_PROVIDER", "openai")
        def_dec_model = getattr(src_config, "DECISION_MODEL", "gpt-5-mini")

        st.markdown("### 🧠 Decision Engine")
        dec_provider = st.selectbox(
            "Provider",
            options=providers,
            index=_find_idx(providers, def_dec_prov),
            key="sb_dec_provider",
        )
        dec_models = supported_models.get(dec_provider, [])
        dec_model = st.selectbox(
            "Model",
            options=dec_models,
            index=_find_idx(dec_models, def_dec_model),
            key="sb_dec_model",
        )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 2. LLM #2: Generation Engine
        # ---------------------------------------------------------------------
        def_gen_prov = getattr(src_config, "GENERATION_PROVIDER", "openai")
        def_gen_model = getattr(src_config, "GENERATION_MODEL", "gpt-5.4-mini")

        st.markdown("### ✨ Generation Engine")
        gen_provider = st.selectbox(
            "Provider",
            options=providers,
            index=_find_idx(providers, def_gen_prov),
            key="sb_gen_provider",
        )
        gen_models = supported_models.get(gen_provider, [])
        gen_model = st.selectbox(
            "Model",
            options=gen_models,
            index=_find_idx(gen_models, def_gen_model),
            key="sb_gen_model",
        )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 3. Retrieval Configuration (Above API Credentials)
        # ---------------------------------------------------------------------
        st.markdown("### 🔍 Retrieval Configuration")
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5, key="sb_top_k")
        search_mode = st.selectbox("Search Mode", options=["Hybrid", "Vector", "BM25"], index=0, key="sb_search_mode")
        reranker = st.selectbox("Reranker", options=["None", "Cross-Encoder", "FlashRank", "LLM"], index=0, key="sb_reranker")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 4. API Credentials (Collapsible, collapsed by default)
        # ---------------------------------------------------------------------
        with st.expander("🔑 API Credentials", expanded=False):
            google_key = st.text_input(
                "Google API Key",
                value=os.getenv("GOOGLE_API_KEY", getattr(src_config, "GOOGLE_API_KEY", "")),
                type="password",
                key="sb_google_key",
            )
            openai_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", getattr(src_config, "OPENAI_API_KEY", "")),
                type="password",
                key="sb_openai_key",
            )
            groq_key = st.text_input(
                "Groq API Key",
                value=os.getenv("GROQ_API_KEY", getattr(src_config, "GROQ_API_KEY", "")),
                type="password",
                key="sb_groq_key",
            )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 5. System Information (Collapsible)
        # ---------------------------------------------------------------------
        with st.expander("ℹ️ System Information", expanded=False):
            st.markdown("**Version**: `1.0.0 (Production Frozen)`")
            st.markdown(f"**Vector DB**: `{getattr(src_config, 'VECTOR_DB', 'Pinecone').title()}`")
            st.markdown("**Orchestration**: `LangGraph Stateful Graph`")
            st.markdown("**Observability**: `ExecutionTrace + LangSmith`")

        return AppConfig(
            decision_provider=dec_provider,
            decision_model=dec_model,
            generation_provider=gen_provider,
            generation_model=gen_model,
            google_key=google_key,
            openai_key=openai_key,
            groq_key=groq_key,
            top_k=top_k,
            search_mode=search_mode,
            reranker=reranker,
        )

