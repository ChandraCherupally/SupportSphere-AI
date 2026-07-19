"""
SupportSphere AI — Single Entry Point.

This file is intentionally minimal. All rendering logic lives in
src/streamlit_app/. Do NOT add business logic here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when launching via `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.streamlit_app.sidebar import render_sidebar
from src.streamlit_app.dashboard import render_dashboard, render_single_ticket, render_batch
from src.streamlit_app.evaluation import render_evaluation

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s| %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SupportSphere AI - Dashboard & Evaluation Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
_SESSION_DEFAULTS = {
    "eval_summary": None,
    "eval_results": [],
    "batch_df": None,
    "batch_outputs": None,
    "eval_exception": None,
    "active_experiment_id": None,
    "compare_exp_a": None,
    "compare_exp_b": None,
    "show_comparison": False,
}
for key, default in _SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

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
    render_batch(cfg)

with tab_eval:
    render_evaluation(cfg)
