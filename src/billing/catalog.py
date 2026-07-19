"""
Billing catalog for SupportSphere AI.

Defines provider and model pricing as of the PRICING_VERSION date.
Every experiment saves a snapshot of this catalog to pricing_snapshot.json
so historical cost calculations remain reproducible even if prices change.
"""

from __future__ import annotations

from typing import Dict, Optional

from src.billing.models import EmbeddingPrice, ModelPrice

# =============================================================================
# Catalog Version
# =============================================================================

PRICING_VERSION: str = "2025-07-18"

# =============================================================================
# Pinecone Query Cost (configurable)
# =============================================================================

PINECONE_QUERY_COST: float = 0.00002  # USD per single vector query

# =============================================================================
# LLM Model Pricing Catalog
# Provider → Model → ModelPrice(input_per_million, output_per_million)
# All prices are in USD per 1 million tokens.
# =============================================================================

_LLM_CATALOG: Dict[str, Dict[str, ModelPrice]] = {
    "google": {
        "gemini-2.5-flash-lite": ModelPrice(input_per_million=0.10, output_per_million=0.40),
        "gemini-2.5-flash": ModelPrice(input_per_million=0.30, output_per_million=2.50),
        "gemini-2.5-pro": ModelPrice(input_per_million=1.25, output_per_million=10.00),
        "gemini-3.1-flash-lite": ModelPrice(input_per_million=0.10, output_per_million=0.40),
        "gemini-3.5-flash": ModelPrice(input_per_million=1.50, output_per_million=9.00),
    },
    "openai": {
        "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60),
        "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00),
        "gpt-5-mini": ModelPrice(input_per_million=0.45, output_per_million=3.60),
        "gpt-5": ModelPrice(input_per_million=2.50, output_per_million=20.00),
        "gpt-5.4-mini": ModelPrice(input_per_million=0.75, output_per_million=4.50),
        "gpt-5.4": ModelPrice(input_per_million=2.50, output_per_million=15.00),
    },
    "groq": {
        "llama-3.3-70b-versatile": ModelPrice(input_per_million=0.59, output_per_million=0.79),
        "openai/gpt-oss-120b": ModelPrice(input_per_million=0.15, output_per_million=0.60),
        "openai/gpt-oss-20b": ModelPrice(input_per_million=0.075, output_per_million=0.30),
    },
}

def get_supported_models() -> Dict[str, list[str]]:
    """Return a mapping of provider to list of model names from _LLM_CATALOG."""
    res = {}
    for provider, models in _LLM_CATALOG.items():
        prov_key = "Google" if provider == "google" else ("OpenAI" if provider == "openai" else provider.title())
        res[prov_key] = list(models.keys())
    return res

# =============================================================================
# Embedding Pricing Catalog
# Provider → Model → EmbeddingPrice(input_per_million)
# =============================================================================

_EMBEDDING_CATALOG: Dict[str, Dict[str, EmbeddingPrice]] = {
    "google": {
        "gemini-embedding-001": EmbeddingPrice(input_per_million=0.15),
    },
    "openai": {
        "text-embedding-3-small": EmbeddingPrice(input_per_million=0.02),
        "text-embedding-3-large": EmbeddingPrice(input_per_million=0.13),
    },
}


# =============================================================================
# Public Accessors
# =============================================================================


def get_model_price(provider: str, model: str) -> Optional[ModelPrice]:
    """Return the ModelPrice for a provider/model combination, or None if unknown.

    Args:
        provider: LLM provider name (e.g. "google", "openai", "groq").
        model: Model identifier string (e.g. "gemini-2.5-flash-lite").

    Returns:
        ModelPrice if found, else None.
    """
    return _LLM_CATALOG.get(provider.lower(), {}).get(model.lower())


def get_embedding_price(provider: str, model: str) -> Optional[EmbeddingPrice]:
    """Return the EmbeddingPrice for a provider/model combination, or None if unknown.

    Args:
        provider: Embedding provider name (e.g. "google").
        model: Embedding model identifier string (e.g. "gemini-embedding-001").

    Returns:
        EmbeddingPrice if found, else None.
    """
    return _EMBEDDING_CATALOG.get(provider.lower(), {}).get(model.lower())


def get_catalog_snapshot() -> dict:
    """Return a serializable snapshot of the full billing catalog.

    Used to persist pricing_snapshot.json alongside each experiment run so
    historical cost calculations remain reproducible even when prices change.

    Returns:
        Dict with pricing_version, pinecone_query_cost, llm_catalog, and
        embedding_catalog suitable for JSON serialization.
    """
    llm_snap: Dict[str, Dict[str, dict]] = {}
    for provider, models in _LLM_CATALOG.items():
        llm_snap[provider] = {
            model: {
                "input_per_million_usd": price.input_per_million,
                "output_per_million_usd": price.output_per_million,
            }
            for model, price in models.items()
        }

    emb_snap: Dict[str, Dict[str, dict]] = {}
    for provider, models in _EMBEDDING_CATALOG.items():
        emb_snap[provider] = {
            model: {"input_per_million_usd": price.input_per_million}
            for model, price in models.items()
        }

    return {
        "pricing_version": PRICING_VERSION,
        "pinecone_query_cost_usd": PINECONE_QUERY_COST,
        "llm_catalog": llm_snap,
        "embedding_catalog": emb_snap,
    }
