"""
Centralized model registry for SupportSphere AI LLM stages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import src.config as config

logger = logging.getLogger(__name__)


class LLMRegistry:
    """
    Centralized registry for stage-specific LLM providers, models, and API keys.
    """

    @staticmethod
    def get_stage_config(stage: str) -> Dict[str, str]:
        """
        Retrieve provider and model config for a given workflow stage.

        Args:
            stage: Name of the stage (e.g., 'decision', 'generation').

        Returns:
            Dict containing 'provider' and 'model'.
        """
        stage_key = stage.lower()
        
        # Check LLM_CONFIG from central config first
        if hasattr(config, "LLM_CONFIG") and stage_key in config.LLM_CONFIG:
            cfg = config.LLM_CONFIG[stage_key]
            provider = cfg.get("provider")
            model = cfg.get("model")
            
            if not provider or not model:
                raise ValueError(f"Stage '{stage}' configuration is incomplete. Provider or model is missing.")
            
            # Validation against centralized catalog (single source of truth)
            from src.billing.catalog import get_supported_models
            supported = get_supported_models()
            prov_key = provider.title() if provider.lower() != "openai" else "OpenAI"
            
            if prov_key not in supported:
                raise ValueError(f"Selected provider '{provider}' is not supported.")
            if model not in supported[prov_key]:
                raise ValueError(f"Selected model '{model}' is not available for {prov_key}. Please select one of the supported models: {', '.join(supported[prov_key])}")
            
            return {
                "provider": provider.lower(),
                "model": model,
            }

        raise ValueError(f"Stage '{stage}' is not configured in LLM_CONFIG.")

    @staticmethod
    def get_api_key(provider: str) -> str:
        """
        Get the API credentials for a specific provider.

        Args:
            provider: LLM provider name.

        Returns:
            API key string.
        """
        prov = provider.lower()
        if prov == "google":
            return getattr(config, "GOOGLE_API_KEY", "")
        elif prov == "openai":
            return getattr(config, "OPENAI_API_KEY", "")
        elif prov == "anthropic":
            return getattr(config, "ANTHROPIC_API_KEY", "")
        elif prov == "groq":
            return getattr(config, "GROQ_API_KEY", "")
        return ""
