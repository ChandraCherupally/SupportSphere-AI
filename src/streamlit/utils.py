"""
Shared Streamlit UI utilities and formatting helpers.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import src.config as src_config

logger = logging.getLogger("SupportSphere")


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

    src_config.GOOGLE_API_KEY = cfg.google_key.strip()
    src_config.OPENAI_API_KEY = cfg.openai_key.strip()
    src_config.GROQ_API_KEY = cfg.groq_key.strip()

    src_config.FINAL_TOP_K = cfg.top_k
    src_config.SEARCH_MODE = cfg.search_mode.lower()
    src_config.RERANKER = cfg.reranker.lower()


def validate_provider_key(provider: str, cfg: AppConfig) -> str | None:
    """Validate that the required API Key for a given provider is entered in API Credentials."""
    p = provider.lower()
    if p == "google" and not cfg.google_key.strip():
        return "Google API Key is missing. Please enter your Google API Key under 'API Credentials' in the sidebar."
    if p == "openai" and not cfg.openai_key.strip():
        return "OpenAI API Key is missing. Please enter your OpenAI API Key under 'API Credentials' in the sidebar."
    if p == "groq" and not cfg.groq_key.strip():
        return "Groq API Key is missing. Please enter your Groq API Key under 'API Credentials' in the sidebar."
    return None


def validate_runtime_keys(cfg: AppConfig) -> list[str]:
    """Validate required API keys for Decision and Generation engines."""
    errors = []
    dec_err = validate_provider_key(cfg.decision_provider, cfg)
    if dec_err:
        errors.append(f"🧠 **Decision Engine ({cfg.decision_provider})**: {dec_err}")

    gen_err = validate_provider_key(cfg.generation_provider, cfg)
    if gen_err:
        # Avoid duplicate error message if Decision and Generation use same provider
        if cfg.generation_provider.lower() != cfg.decision_provider.lower() or not dec_err:
            errors.append(f"✨ **Generation Engine ({cfg.generation_provider})**: {gen_err}")

    return errors


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
