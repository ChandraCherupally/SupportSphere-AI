"""
Provider-agnostic LLM client supporting stage-based routing, caching, and token usage capture.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import src.config as config

logger = logging.getLogger("SupportSphere")


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


class SafeChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that strips unsupported temperature from API payloads for GPT-5/o-series models."""
    def _get_request_payload(self, input_, *args, **kwargs):
        payload = super()._get_request_payload(input_, *args, **kwargs)
        if "temperature" in payload:
            model_name = str(getattr(self, "model_name", None) or getattr(self, "model", "") or "").lower()
            if any(sub in model_name for sub in ("o1", "o3", "o-1", "o-3", "reason", "r1", "gpt-5", "gpt-4.5")) or payload["temperature"] in (0.3, 0.01, 0.0, 0.7):
                del payload["temperature"]
        return payload


class SafeChatGroq(ChatGroq):
    """ChatGroq subclass that strips unsupported temperature from invocation params for reasoning/versatile models."""
    def _get_invocation_params(self, stop=None, **kwargs):
        params = super()._get_invocation_params(stop=stop, **kwargs)
        if "temperature" in params:
            model_name = str(getattr(self, "model", "") or "").lower()
            if any(sub in model_name for sub in ("o1", "o3", "o-1", "o-3", "reason", "r1", "versatile")) or params["temperature"] in (0.3, 0.01, 0.0, 0.7):
                params["temperature"] = None
        return params


class LLMClient:
    """
    Orchestrates connection and invocation of multiple LLM providers based on workflow stages.
    """

    def __init__(self) -> None:
        self._cached_models: Dict[tuple[str, str, str], BaseChatModel] = {}

    def _get_model(
        self, provider: str, model_name: str, api_key: Optional[str] = None, stage: str = "generation"
    ) -> BaseChatModel:
        """
        Retrieves or instantiates a cached LangChain Chat client.
        """
        prov_clean = provider.lower()
        key_str = api_key or config.get_api_key(prov_clean)
        cache_key = (prov_clean, model_name.lower(), key_str)
        if cache_key in self._cached_models:
            return self._cached_models[cache_key]

        temp = getattr(config, "LLM_TEMPERATURE", 0.0)
        node_prefix = "[Node 1: Decision Engine]" if stage.lower() == "decision" else "[Node 4: Generation Engine]"

        logger.info("%s Instantiating Chat client for provider=%s model=%s", node_prefix, provider, model_name)

        is_temp_restricted = any(sub in model_name.lower() for sub in ("o1", "o3", "o-1", "o-3", "reason", "r1", "gpt-5", "gpt-4.5", "versatile"))

        if prov_clean == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs: Dict[str, Any] = {
                "model": model_name,
                "google_api_key": key_str or "",
            }
            if not is_temp_restricted:
                kwargs["temperature"] = temp
            model = ChatGoogleGenerativeAI(**kwargs)
        elif prov_clean == "openai":
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "api_key": key_str or "",
            }
            if not is_temp_restricted:
                kwargs["temperature"] = temp
            model = SafeChatOpenAI(**kwargs)
        elif prov_clean == "groq":
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "api_key": key_str or "",
            }
            if not is_temp_restricted:
                kwargs["temperature"] = temp
            model = SafeChatGroq(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if is_temp_restricted and hasattr(model, "temperature"):
            try:
                model.temperature = None
            except Exception:
                pass

        self._cached_models[cache_key] = model
        return model

    def _fallback_parse_schema(self, text: str, response_schema: Type[BaseModel]) -> BaseModel:
        """
        Robust fallback parser when structured output function calling fails or outputs text.
        Extracts JSON, YAML, or key-value fields from text and instantiates response_schema.
        """
        if not isinstance(text, str):
            text = str(text)

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json|yaml)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # 1. Direct JSON parsing
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return response_schema.model_validate(data)
        except Exception:
            pass

        # 2. Extract key-value pairs using regex
        extracted_data: Dict[str, Any] = {}

        pattern = re.compile(r"^(?:[-*]\s*)?\**([a-zA-Z0-9_]+)\**\s*:\s*(.+)$", re.MULTILINE)

        for match in pattern.finditer(text):
            k, v = match.group(1).strip().lower(), match.group(2).strip()
            v_clean = v.strip("`'\"")
            extracted_data[k] = v_clean

        schema_fields = getattr(response_schema, "model_fields", {})
        init_kwargs: Dict[str, Any] = {}
        for field_name, field_info in schema_fields.items():
            if field_name in extracted_data:
                val = extracted_data[field_name]
                target_type = field_info.annotation
                if target_type == bool:
                    init_kwargs[field_name] = str(val).lower() in ("true", "1", "yes")
                elif target_type == float:
                    try:
                        init_kwargs[field_name] = float(val)
                    except ValueError:
                        init_kwargs[field_name] = 1.0
                elif target_type == int:
                    try:
                        init_kwargs[field_name] = int(val)
                    except ValueError:
                        init_kwargs[field_name] = 0
                else:
                    init_kwargs[field_name] = str(val)

        # Supply smart defaults for missing required fields
        if "request_type" in schema_fields and "request_type" not in init_kwargs:
            init_kwargs["request_type"] = "product_issue"
        if "status" in schema_fields and "status" not in init_kwargs:
            init_kwargs["status"] = "Replied"
        if "product_area" in schema_fields and "product_area" not in init_kwargs:
            init_kwargs["product_area"] = "Conversation_Management"
        if "response" in schema_fields and "response" not in init_kwargs:
            init_kwargs["response"] = text[:1000] if text else "Support assistance."
        if "justification" in schema_fields and "justification" not in init_kwargs:
            init_kwargs["justification"] = "Fallback parsed response."

        try:
            return response_schema.model_validate(init_kwargs)
        except Exception as parse_err:
            logger.warning("Fallback schema validation failed: %s. Constructing default.", parse_err)
            return response_schema.model_construct(**init_kwargs)

    def generate(
        self,
        messages: Any,
        response_schema: Optional[Type[BaseModel]] = None,
        stage: str = "generation",
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> LLMResult:
        """
        Execute generation for a specific workflow stage.
        """
        if not provider or not model_name:
            if stage.lower() == "decision":
                provider = provider or config.DECISION_PROVIDER
                model_name = model_name or config.DECISION_MODEL
            else:
                provider = provider or config.GENERATION_PROVIDER
                model_name = model_name or config.GENERATION_MODEL

        key_str = api_key or config.get_api_key(provider)
        node_prefix = "[Node 1: Decision Engine]" if stage.lower() == "decision" else "[Node 4: Generation Engine]"

        logger.info(
            "%s Executing Stage: %s | Provider: %s | Model: %s",
            node_prefix,
            stage,
            provider,
            model_name,
        )

        llm = self._get_model(provider, model_name, key_str, stage=stage)

        t_start = time.perf_counter()

        try:
            if response_schema is not None:
                so_kwargs = {}
                # Pass method="function_calling" ONLY for OpenAI
                if provider.lower() == "openai":
                    so_kwargs["method"] = "function_calling"

                try:
                    structured_llm = llm.with_structured_output(
                        response_schema, include_raw=True, **so_kwargs
                    )
                    res = structured_llm.invoke(messages)
                    parsed = res.get("parsed") if isinstance(res, dict) else None
                    raw = res.get("raw") if isinstance(res, dict) else res
                    if parsed is None and raw and hasattr(raw, "content"):
                        parsed = self._fallback_parse_schema(raw.content, response_schema)
                except Exception as inner_err:
                    logger.warning(
                        "Structured output failed for %s/%s (%s). Attempting fallback recovery.",
                        provider,
                        model_name,
                        inner_err,
                    )
                    # Try raw LLM invoke without structured output constraint
                    try:
                        raw = llm.invoke(messages)
                        content_str = raw.content if hasattr(raw, "content") else str(raw)
                        parsed = self._fallback_parse_schema(content_str, response_schema)
                    except Exception as raw_err:
                        # Extract failed_generation from error payload (Groq 400 error)
                        failed_gen = getattr(inner_err, "failed_generation", None)
                        if not failed_gen and hasattr(inner_err, "body") and isinstance(inner_err.body, dict):
                            err_dict = inner_err.body.get("error", {})
                            failed_gen = err_dict.get("failed_generation")
                        if not failed_gen:
                            err_str = str(inner_err)
                            if "failed_generation" in err_str:
                                match = re.search(r"['\"]failed_generation['\"]\s*:\s*['\"](.*?)['\"]\s*\}", err_str, re.DOTALL)
                                if match:
                                    failed_gen = match.group(1)
                        if failed_gen:
                            parsed = self._fallback_parse_schema(failed_gen, response_schema)
                            raw = None
                        else:
                            raise inner_err
            else:
                raw = llm.invoke(messages)
                parsed = raw.content

        except Exception as e:
            logger.error(
                "%s LLM call failed for stage=%s using provider=%s model=%s: %s",
                node_prefix,
                stage,
                provider,
                model_name,
                e,
                exc_info=True,
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
            "%s Stage '%s' execution complete (Provider: %s, Model: %s). "
            "Latency: %.2fs, Input: %d, Output: %d.",
            node_prefix,
            stage,
            provider,
            model_name,
            t_elapsed,
            input_tokens,
            output_tokens,
        )

        return LLMResult(
            response=parsed,
            usage=usage,
            latency=t_elapsed,
            provider=provider,
            model=model_name,
            raw_response=raw,
        )
