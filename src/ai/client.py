"""
Provider-agnostic LLM client supporting stage-based routing, caching, and token usage capture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from src.ai.registry import LLMRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMUsage:
    """
    Immutable token usage count.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    """
    Immutable response wrapper returned by LLMClient.
    """
    response: Any  # Parsed BaseModel (if response_schema passed) or str (if raw output)
    usage: LLMUsage
    latency: float
    provider: str
    model: str
    raw_response: Any


class LLMClient:
    """
    Orchestrates connection and invocation of multiple LLM providers based on workflow stages.
    """

    def __init__(self) -> None:
        self._cached_models: Dict[tuple[str, str, str], BaseChatModel] = {}

    def _get_model(self, provider: str, model_name: str) -> BaseChatModel:
        """
        Retrieves or instantiates a cached LangChain Chat client.
        """
        api_key = LLMRegistry.get_api_key(provider)
        key = (provider.lower(), model_name.lower(), api_key)
        if key in self._cached_models:
            return self._cached_models[key]

        import src.config as config
        temp = getattr(config, "LLM_TEMPERATURE", 0.0)

        logger.info("Instantiating Chat client for provider=%s model=%s", provider, model_name)

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            kwargs: Dict[str, Any] = {"model": model_name, "temperature": temp}
            if api_key:
                kwargs["google_api_key"] = api_key
            model = ChatGoogleGenerativeAI(**kwargs)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            kwargs: Dict[str, Any] = {"model": model_name, "temperature": temp}
            if api_key:
                kwargs["api_key"] = api_key
            model = ChatOpenAI(**kwargs)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            kwargs: Dict[str, Any] = {"model": model_name, "temperature": temp}
            if api_key:
                kwargs["api_key"] = api_key
            model = ChatAnthropic(**kwargs)
        elif provider == "groq":
            from langchain_groq import ChatGroq
            kwargs: Dict[str, Any] = {"model": model_name, "temperature": temp}
            if api_key:
                kwargs["api_key"] = api_key
            model = ChatGroq(**kwargs)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            model = ChatOllama(
                model=model_name,
                temperature=temp,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        self._cached_models[key] = model
        return model

    def generate(
        self,
        messages: Any,
        response_schema: Optional[Type[BaseModel]] = None,
        stage: str = "generation",
    ) -> LLMResult:
        """
        Execute generation for a specific workflow stage.

        Args:
            messages: Prompts/messages list.
            response_schema: Optional Pydantic model for structured output validation.
            stage: Configured stage ('decision', 'generation', etc.)

        Returns:
            Immutable LLMResult container containing output + token usage.
        """
        stage_cfg = LLMRegistry.get_stage_config(stage)
        provider = stage_cfg["provider"]
        model_name = stage_cfg["model"]

        import src.config as config
        config_source = "UI Selection" if hasattr(config, "LLM_CONFIG") else "Environment Config"

        logger.info(
            "Executing Stage\n%s\nProvider\n%s\nModel\n%s\nSource\n%s",
            stage, provider.title() if provider.lower() != "openai" else "OpenAI", model_name, config_source
        )

        llm = self._get_model(provider, model_name)

        t_start = time.perf_counter()
        
        try:
            if response_schema is not None:
                structured_llm = llm.with_structured_output(response_schema, include_raw=True)
                res = structured_llm.invoke(messages)
                parsed = res["parsed"]
                raw = res["raw"]
            else:
                raw = llm.invoke(messages)
                parsed = raw.content
        except Exception as e:
            logger.error(
                "LLM call failed for stage=%s using provider=%s model=%s: %s",
                stage, provider, model_name, e, exc_info=True
            )
            raise e

        t_elapsed = time.perf_counter() - t_start

        input_tokens = 0
        output_tokens = 0
        if raw and hasattr(raw, "usage_metadata") and raw.usage_metadata:
            input_tokens = raw.usage_metadata.get("input_tokens", 0)
            output_tokens = raw.usage_metadata.get("output_tokens", 0)

        usage = LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        logger.info(
            "Stage '%s' execution complete (Provider: %s, Model: %s). "
            "Latency: %.2fs, Input: %d, Output: %d.",
            stage, provider, model_name, t_elapsed, input_tokens, output_tokens
        )

        return LLMResult(
            response=parsed,
            usage=usage,
            latency=t_elapsed,
            provider=provider,
            model=model_name,
            raw_response=raw,
        )