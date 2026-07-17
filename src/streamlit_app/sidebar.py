"""
Sidebar rendering for SupportSphere AI Streamlit application.

Renders the full sidebar (provider, model, API keys, retrieval settings,
system info) and returns an AppConfig dataclass with all user selections.
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
    st.sidebar.markdown("# SupportSphere AI")
    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # Group 1: Inference Settings
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### Inference Settings")

    provider = st.sidebar.selectbox(
        "Provider",
        options=["Google", "OpenAI", "Groq"],
        index=0,
        key="sb_provider",
    )

    model_options = {
        "Google": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5"],
        "Groq": ["llama-3.3-70b-versatile"],
    }

    selected_model = st.sidebar.selectbox(
        "Model",
        options=model_options[provider],
        index=0,
        key="sb_model",
    )

    # -----------------------------------------------------------------------
    # API Keys
    # -----------------------------------------------------------------------
    google_key = st.sidebar.text_input(
        "Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password"
    )
    groq_key = st.sidebar.text_input(
        "Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password"
    )
    openai_key = st.sidebar.text_input(
        "OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password"
    )

    # Key validation
    keys_provided = True
    if provider == "Google" and not google_key:
        keys_provided = False
    elif provider == "OpenAI" and not openai_key:
        keys_provided = False
    elif provider == "Groq" and not groq_key:
        keys_provided = False

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # Group 2: Retrieval Settings
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### Retrieval Settings")

    top_k = st.sidebar.slider("Top-K Chunks", min_value=1, max_value=15, value=5, step=1, key="sb_top_k")
    search_mode = st.sidebar.selectbox(
        "Search Mode", options=["Hybrid", "Vector", "BM25"], index=0, key="sb_search_mode"
    )
    reranker = st.sidebar.selectbox(
        "Reranker", options=["None", "FlashRank", "CrossEncoder", "LLM"], index=0, key="sb_reranker"
    )

    st.sidebar.markdown("---")

    # -----------------------------------------------------------------------
    # Group 3: System Information
    # -----------------------------------------------------------------------
    st.sidebar.markdown("### System Information")
    st.sidebar.write(f"**Provider**: `{provider}`")
    st.sidebar.write(f"**Model**: `{selected_model}`")
    st.sidebar.write(f"**Search Mode**: `{search_mode}`")
    st.sidebar.write(f"**Reranker**: `{reranker}`")
    st.sidebar.write("**Version**: `v1.3.0`")

    return AppConfig(
        provider=provider,
        selected_model=selected_model,
        google_key=google_key,
        openai_key=openai_key,
        groq_key=groq_key,
        anthropic_key="",
        top_k=top_k,
        search_mode=search_mode,
        reranker=reranker,
        keys_provided=keys_provided,
    )
