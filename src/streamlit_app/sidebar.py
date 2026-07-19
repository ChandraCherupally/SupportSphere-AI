"""
Sidebar rendering for SupportSphere AI Streamlit application.

Renders the reorganized sidebar grouping settings into clean sections
and expanders to minimize visual clutter for recruiter demos.
"""

from __future__ import annotations

import os
import streamlit as st

from src.streamlit_app.utils import AppConfig


def render_sidebar() -> AppConfig:
    """Render the full Streamlit sidebar and return an AppConfig.

    Returns:
        AppConfig populated with all current sidebar widget values.
    """
    st.sidebar.markdown("# ⚙️ SupportSphere Platform")
    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # Group 1: Inference Settings (Visible by default)
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### 🤖 AI Models")
    
    from src.billing.catalog import get_supported_models
    model_options = get_supported_models()

    # 1.1 Decision Model Config
    st.sidebar.markdown("**🔀 Decision Model (Routing)**")
    decision_provider = st.sidebar.selectbox(
        "Decision Provider",
        options=["Google", "OpenAI", "Groq"],
        index=0,
        key="sb_decision_provider",
        label_visibility="collapsed"
    )
    decision_model = st.sidebar.selectbox(
        "Decision Model",
        options=model_options[decision_provider],
        index=0,
        key="sb_decision_model",
        label_visibility="visible"
    )

    # 1.2 Generation Model Config
    st.sidebar.markdown("**✍️ Generation Model (Response)**")
    generation_provider = st.sidebar.selectbox(
        "Generation Provider",
        options=["Google", "OpenAI", "Groq"],
        index=0,
        key="sb_generation_provider",
        label_visibility="collapsed"
    )
    
    gen_opts = model_options[generation_provider]
    gen_default_idx = 0
    if generation_provider == "Google":
        if "gemini-2.5-flash" in gen_opts:
            gen_default_idx = gen_opts.index("gemini-2.5-flash")
        elif len(gen_opts) > 1:
            gen_default_idx = 1
            
    generation_model = st.sidebar.selectbox(
        "Generation Model",
        options=gen_opts,
        index=gen_default_idx,
        key="sb_generation_model",
        label_visibility="visible"
    )

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # Group 2: Retrieval Settings (Visible by default)
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### 🔍 Retrieval Configuration")

    top_k = st.sidebar.slider(
        "Top-K Chunks", min_value=1, max_value=15, value=5, step=1, key="sb_top_k"
    )
    search_mode = st.sidebar.selectbox(
        "Search Mode", options=["Hybrid", "Vector", "BM25"], index=0, key="sb_search_mode"
    )
    reranker = st.sidebar.selectbox(
        "Reranker", options=["None", "FlashRank", "CrossEncoder", "LLM"], index=0, key="sb_reranker"
    )

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # Group 3: Collapsible Advanced Settings (Expanders)
    # -----------------------------------------------------------------------
    
    # 3.1 API Keys Expander
    with st.sidebar.expander("🔑 API Credentials", expanded=False):
        google_key = st.text_input(
            "Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password"
        )
        groq_key = st.text_input(
            "Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password"
        )
        openai_key = st.text_input(
            "OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password"
        )

    # 3.2 System Information Expander
    with st.sidebar.expander("ℹ️ System Information", expanded=False):
        st.write(f"**Decision**: `{decision_provider}` / `{decision_model}`")
        st.write(f"**Generation**: `{generation_provider}` / `{generation_model}`")
        st.write(f"**Search Mode**: `{search_mode}`")
        st.write(f"**Reranker**: `{reranker}`")

    # Key validation based on active providers
    active_providers = {decision_provider, generation_provider}
    keys_provided = True
    if "Google" in active_providers and not google_key:
        keys_provided = False
    if "OpenAI" in active_providers and not openai_key:
        keys_provided = False
    if "Groq" in active_providers and not groq_key:
        keys_provided = False

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.8em; margin-top: 15px;'>Platform Version: v1.5.0</div>", unsafe_allow_html=True)

    return AppConfig(
        decision_provider=decision_provider,
        decision_model=decision_model,
        generation_provider=generation_provider,
        generation_model=generation_model,
        google_key=google_key,
        openai_key=openai_key,
        groq_key=groq_key,
        anthropic_key="",
        top_k=top_k,
        search_mode=search_mode,
        reranker=reranker,
        keys_provided=keys_provided,
    )
