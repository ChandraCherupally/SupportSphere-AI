"""
RuntimeConfig — single source of truth for active provider, model, and execution settings.

Created from Streamlit sidebar selections or global config and injected into LangGraph state.
Every node reads from state["runtime_config"].
"""

from __future__ import annotations

from dataclasses import dataclass
import src.config as src_config


@dataclass
class RuntimeConfig:
    """
    Immutable runtime configuration for a single ticket execution.
    Propagated through LangGraph state so every node sees the same settings.
    """

    decision_provider: str = "openai"
    decision_model: str = "gpt-5-mini"
    generation_provider: str = "openai"
    generation_model: str = "gpt-5.4-mini"
    google_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    top_k: int = 5
    search_mode: str = "hybrid"
    reranker: str = "none"

    # -------------------------------------------------------------------------
    # Stage-level accessors
    # -------------------------------------------------------------------------

    def get_stage_provider(self, stage: str) -> str:
        """Return the configured provider for the given stage name."""
        if stage == "decision":
            return self.decision_provider
        if stage == "generation":
            return self.generation_provider
        raise ValueError(f"Unknown stage: '{stage}'")

    def get_stage_model(self, stage: str) -> str:
        """Return the configured model for the given stage name."""
        if stage == "decision":
            return self.decision_model
        if stage == "generation":
            return self.generation_model
        raise ValueError(f"Unknown stage: '{stage}'")

    def get_api_key(self, provider: str) -> str:
        """Return the API key for the given provider, falling back to global config."""
        p = provider.lower()
        key = ""
        if p == "google":
            key = self.google_api_key
        elif p == "openai":
            key = self.openai_api_key
        elif p == "groq":
            key = self.groq_api_key
        return key or src_config.get_api_key(p)
