"""
SupportSphere AI — Single Entry Point.

This file is intentionally minimal. All rendering logic lives in
src/streamlit/. Do NOT add business logic here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Patch Windows asyncio Proactor Loop WinError 10054 noise
if sys.platform == "win32":
    try:
        import asyncio.proactor_events as _pe

        _orig_ccl = _pe._ProactorBasePipeTransport._call_connection_lost

        def _ccl_patched(self, exc):
            try:
                _orig_ccl(self, exc)
            except (ConnectionResetError, ConnectionAbortedError):
                pass

        _pe._ProactorBasePipeTransport._call_connection_lost = _ccl_patched
    except Exception:
        pass

# Ensure project root is on PYTHONPATH when launching via `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.logging import setup_logging
from src.streamlit.batch_processing import render_batch_processing
from src.streamlit.dashboard import render_dashboard
from src.streamlit.evaluation import render_evaluation
from src.streamlit.sidebar import render_sidebar
from src.streamlit.single_ticket import render_single_ticket

setup_logging()
logger = logging.getLogger("SupportSphere")

# ---------------------------------------------------------------------------
# Application Startup Warmup (Initialize once at startup)
# ---------------------------------------------------------------------------
try:
    logger.info("Application Startup")
    from src.guardrails.service import NeMoGuardrailsService

    NeMoGuardrailsService()
    logger.info("Guardrails Initialized")

    from vectordb_ingestion_pipeline.ingestion.pinecone_client import PineconeClient

    PineconeClient()
    logger.info("Pinecone Connected")

    from src.retrieval.vector_search import VectorSearch

    VectorSearch()
    logger.info("Embedding Model Loaded")

    from src.graph.nodes.retrieval import get_retriever

    get_retriever("none", "hybrid")
    logger.info("Retriever Ready")
except Exception as warmup_err:
    logger.error("Warmup initialization failed: %s", warmup_err)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SupportSphere AI — Triage & Support System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar (returns AppConfig)
# ---------------------------------------------------------------------------
cfg = render_sidebar()

# ---------------------------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------------------------
tab_home, tab_single, tab_batch, tab_eval = st.tabs(
    ["🏠 Dashboard", "🎫 Single Ticket", "📦 Batch Processing", "📊 Evaluation"]
)

with tab_home:
    render_dashboard(cfg)

with tab_single:
    render_single_ticket(cfg)

with tab_batch:
    render_batch_processing(cfg)

with tab_eval:
    render_evaluation(cfg)
