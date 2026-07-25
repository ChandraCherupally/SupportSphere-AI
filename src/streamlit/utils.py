"""
Shared Streamlit UI utilities and formatting helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import src.config as src_config

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Sidebar-configured runtime settings."""

    decision_provider: str = "OpenAI"
    decision_model: str = "gpt-5-mini"
    generation_provider: str = "OpenAI"
    generation_model: str = "gpt-5.4-mini"
    google_key: str = ""
    openai_key: str = ""
    groq_key: str = ""
    top_k: int = 5
    search_mode: str = "Hybrid"
    reranker: str = "None"
    keys_provided: bool = True

    @property
    def provider(self) -> str:
        return self.generation_provider

    @property
    def selected_model(self) -> str:
        return self.generation_model

    def to_runtime_config(self):
        """Return a typed RuntimeConfig built from this AppConfig."""
        from src.runtime_config.runtime import RuntimeConfig

        return RuntimeConfig(
            decision_provider=self.decision_provider.lower(),
            decision_model=self.decision_model,
            generation_provider=self.generation_provider.lower(),
            generation_model=self.generation_model,
            google_api_key=self.google_key,
            openai_api_key=self.openai_key,
            groq_api_key=self.groq_key,
            top_k=self.top_k,
            search_mode=self.search_mode.lower(),
            reranker=self.reranker.lower(),
        )


def patch_agent_configs(cfg: AppConfig) -> None:
    """Propagate Streamlit sidebar UI settings to system config modules."""
    src_config.DECISION_PROVIDER = cfg.decision_provider.lower()
    src_config.DECISION_MODEL = cfg.decision_model
    src_config.GENERATION_PROVIDER = cfg.generation_provider.lower()
    src_config.GENERATION_MODEL = cfg.generation_model

    src_config.GOOGLE_API_KEY = cfg.google_key or getattr(src_config, "GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or ""
    src_config.OPENAI_API_KEY = cfg.openai_key or getattr(src_config, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or ""
    src_config.GROQ_API_KEY = cfg.groq_key or getattr(src_config, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "") or ""

    src_config.FINAL_TOP_K = cfg.top_k
    src_config.SEARCH_MODE = cfg.search_mode.lower()
    src_config.RERANKER = cfg.reranker.lower()


def get_html_badge(text: str, badge_type: str = "info") -> str:
    """Render a styled HTML badge."""
    colors = {
        "success": ("#dcfce7", "#15803d", "#86efac"),
        "danger": ("#fee2e2", "#b91c1c", "#fca5a5"),
        "warning": ("#fef3c7", "#b45309", "#fde047"),
        "info": ("#e0f2fe", "#0369a1", "#7dd3fc"),
        "neutral": ("#f1f5f9", "#475569", "#cbd5e1"),
    }
    bg, fg, border = colors.get(badge_type, colors["info"])
    return (
        f'<span style="background-color: {bg}; color: {fg}; border: 1px solid {border}; '
        f'padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.82em;">'
        f"{text}</span>"
    )
