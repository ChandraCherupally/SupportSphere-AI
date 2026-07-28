"""
NVIDIA NeMo Guardrails Service wrapper.
Routes all LLM calls through the Decision Model via LLMClient and SupportSphereChatModel.
Exposes pure NeMo policy execution (check_input, check_retrieval, check_output).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.providers import register_llm_provider

import src.config as src_config
from src.ai.client import LLMClient
from src.billing.calculator import BillingCalculator

logger = logging.getLogger("SupportSphere")


class GuardrailUsageAccumulator:
    """
    Thread-local storage for accumulating token counts and billing costs
    across multiple guardrail LLM runs for a single ticket.
    """

    _local = threading.local()

    @classmethod
    def reset(cls) -> None:
        """Reset ticket-level accumulated totals at ticket start."""
        cls._local.input_tokens = 0
        cls._local.output_tokens = 0
        cls._local.cost = 0.0
        cls._local.stage_input_tokens = 0
        cls._local.stage_output_tokens = 0
        cls._local.stage_cost = 0.0

    @classmethod
    def start_stage_tracking(cls) -> None:
        """Reset stage-level tracking totals at the start of a guardrail stage."""
        if not hasattr(cls._local, "input_tokens"):
            cls.reset()
        cls._local.stage_input_tokens = 0
        cls._local.stage_output_tokens = 0
        cls._local.stage_cost = 0.0

    @classmethod
    def add(cls, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Add tokens and cost to both stage and ticket accumulators."""
        if not hasattr(cls._local, "input_tokens"):
            cls.reset()
        cls._local.input_tokens += input_tokens
        cls._local.output_tokens += output_tokens
        cls._local.cost += cost
        cls._local.stage_input_tokens += input_tokens
        cls._local.stage_output_tokens += output_tokens
        cls._local.stage_cost += cost

    @classmethod
    def get_stage_totals(cls) -> tuple[int, int, float]:
        """Get accumulated totals for the current guardrail stage."""
        if not hasattr(cls._local, "stage_input_tokens"):
            return 0, 0, 0.0
        return (
            cls._local.stage_input_tokens,
            cls._local.stage_output_tokens,
            cls._local.stage_cost,
        )

    @classmethod
    def get_ticket_totals(cls) -> tuple[int, int, float]:
        """Get cumulative totals for the entire ticket."""
        if not hasattr(cls._local, "input_tokens"):
            return 0, 0, 0.0
        return (
            cls._local.input_tokens,
            cls._local.output_tokens,
            cls._local.cost,
        )


from typing import AsyncIterator, List, Optional, Union
from nemoguardrails.types import ChatMessage, LLMResponse, LLMResponseChunk, UsageInfo
from nemoguardrails.llm.providers import register_provider


class SupportSphereLLMModel:
    """
    Custom NeMo LLMModel engine registered with NeMo Guardrails.
    Routes all NeMo LLM calls directly to the configured Decision Model.
    """

    def __init__(self, model: str = "decision_model", **kwargs):
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> Optional[str]:
        return "supportsphere_llm"

    @property
    def provider_url(self) -> Optional[str]:
        return None

    async def generate_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        if isinstance(prompt, str):
            messages = [("human", prompt)]
        else:
            messages = []
            for msg in prompt:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", str(msg))
                if role in ("system", "context"):
                    messages.append(("system", content))
                elif role in ("assistant", "bot"):
                    messages.append(("assistant", content))
                else:
                    messages.append(("human", content))

        client = LLMClient()
        res = client.generate(messages, stage="decision")

        import src.config as src_config

        calc = BillingCalculator(
            embedding_provider=getattr(src_config, "EMBEDDING_PROVIDER", "google"),
            embedding_model=getattr(src_config, "EMBEDDING_MODEL", "gemini-embedding-001"),
            reranker="none",
            decision_provider=src_config.DECISION_PROVIDER,
            decision_model=src_config.DECISION_MODEL,
            generation_provider=src_config.GENERATION_PROVIDER,
            generation_model=src_config.GENERATION_MODEL,
        )
        cost = calc._llm_cost(res.usage, calc._decision_price)

        GuardrailUsageAccumulator.add(res.usage.input_tokens, res.usage.output_tokens, cost)

        return LLMResponse(
            content=res.response,
            usage=UsageInfo(
                input_tokens=res.usage.input_tokens,
                output_tokens=res.usage.output_tokens,
                total_tokens=res.usage.input_tokens + res.usage.output_tokens,
            ),
        )

    async def stream_async(
        self,
        prompt: Union[str, List[ChatMessage]],
        *,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]:
        resp = await self.generate_async(prompt, stop=stop, **kwargs)
        yield LLMResponseChunk(text=resp.content)


# Register the custom LLM engine in NeMo Guardrails
register_provider("supportsphere_llm", SupportSphereLLMModel)



class NeMoGuardrailsService:
    """
    Orchestration layer for NVIDIA NeMo Guardrails policy execution.
    Derives its model configuration from the currently selected Decision Model.
    """

    _instance: Optional["NeMoGuardrailsService"] = None
    _evaluation_mode: bool = False

    @classmethod
    def set_evaluation_mode(cls, enabled: bool) -> None:
        """Enable or disable evaluation bypass mode."""
        cls._evaluation_mode = enabled

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NeMoGuardrailsService, cls).__new__(cls)
            cls._instance._initialize()
        else:
            provider = getattr(src_config, "DECISION_PROVIDER", "google").lower()
            model = getattr(src_config, "DECISION_MODEL", "gemini-2.5-flash-lite")
            if provider != cls._instance.provider or model != cls._instance.model:
                cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.provider = getattr(src_config, "DECISION_PROVIDER", "google").lower()
        self.model = getattr(src_config, "DECISION_MODEL", "gemini-2.5-flash-lite")

        # Enabled if API Key is configured for the decision provider
        self.enabled = bool(src_config.get_api_key(self.provider))
        self.rails = None

        if self.enabled:
            try:
                config_dir = os.path.join(os.path.dirname(__file__), "config")
                if os.path.exists(config_dir):
                    config = RailsConfig.from_path(config_dir)

                    # Replace model configuration in-place to use SupportSphere LLM engine
                    for model_cfg in config.models:
                        if model_cfg.type == "main":
                            model_cfg.engine = "supportsphere_llm"
                            model_cfg.model = "decision_model"

                    self.rails = LLMRails(config)
                else:
                    logger.error("NeMo config directory not found: %s", config_dir)
                    self.enabled = False
            except Exception as e:
                logger.error("Failed to initialize NeMo Guardrails: %s", e)
                self.enabled = False

    def is_available(self) -> bool:
        return (
            self.enabled
            and self.rails is not None
            and not self.__class__._evaluation_mode
        )

    def _generate_with_timeout(self, messages: List[Dict[str, Any]], timeout: float = 6.0) -> Dict[str, Any]:
        """
        Executes NeMo rails.generate inside a thread with a dedicated event loop.
        Drains pending tasks before event loop closure to prevent un-awaited coroutine warnings.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _run_with_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except Exception:
                pass

            try:
                return self.rails.generate(messages=messages)
            finally:
                try:
                    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                    if pending:
                        for t in pending:
                            t.cancel()
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_with_loop)
            return future.result(timeout=timeout)

    def check_input(self, subject: str, issue: str) -> Dict[str, Any]:
        """
        Executes NeMo input safety rails checks.
        """
        if self.__class__._evaluation_mode:
            return {
                "decision": "ALLOW",
                "reason": "Evaluation mode: guardrail bypassed",
            }
        if not self.is_available():
            return {"decision": "ALLOW", "reason": "Guardrail service not available"}

        try:
            res = self._generate_with_timeout(
                messages=[{"role": "user", "content": f"{subject}\n{issue}"}],
                timeout=6.0,
            )
            content = res.get("content", "").strip() if isinstance(res, dict) else ""

            refusal = ("I am sorry, but I cannot fulfill this request due to safety policies.")
            if refusal in content:
                return {
                    "decision": "BLOCK",
                    "reason": "Input safety violation detected by NeMo rails.",
                    "rule": "check_input_safety",
                }
            return {"decision": "ALLOW", "reason": "Input is safe."}
        except Exception as e:
            logger.error("NeMo check_input failed or timed out: %s", e)
            return {"decision": "ALLOW", "reason": f"Guardrail fallback: {str(e)}"}


    def check_retrieval(self, company: str, issue: str, context: List[Any]) -> Dict[str, Any]:
        """
        Executes NeMo retrieval context rails checks.
        """
        if self.__class__._evaluation_mode:
            return {
                "decision": "ALLOW",
                "reason": "Evaluation mode: guardrail bypassed",
                "filtered_context_indices": list(range(len(context))),
            }
        if not self.is_available() or not context:
            return {
                "decision": "ALLOW",
                "reason": "Guardrail service not available or context empty",
                "filtered_context_indices": list(range(len(context))),
            }

        try:
            from src.guardrails.config.actions import (filter_duplicate_chunks, verify_company)

            context_dicts = []
            for chunk in context:
                text = getattr(chunk, "text", "")
                if not text and isinstance(chunk, dict):
                    text = chunk.get("text", "")
                chunk_company = getattr(chunk, "company", "")
                if not chunk_company and isinstance(chunk, dict):
                    chunk_company = chunk.get("company", "")
                context_dicts.append({"text": text, "company": chunk_company})

            # 1. Company verification
            ver_res = verify_company(company, context_dicts)
            if ver_res.get("decision") == "BLOCK":
                return {
                    "decision": "BLOCK",
                    "reason": ver_res.get("reason", "Company mismatch detected."),
                    "rule": "verify_company",
                    "filtered_context_indices": list(range(len(context))),
                }

            # 2. Duplicate chunk filtering
            dup_res = filter_duplicate_chunks(context_dicts)
            filtered_indices = dup_res.get("filtered_context_indices", list(range(len(context))))

            return {
                "decision": "ALLOW",
                "reason": "Retrieval is safe.",
                "filtered_context_indices": filtered_indices,
            }
        except Exception as e:
            logger.error("NeMo check_retrieval failed or timed out: %s", e)
            return {
                "decision": "ALLOW",
                "reason": f"Guardrail fallback: {str(e)}",
                "filtered_context_indices": list(range(len(context))),
            }

    def check_output(self, issue: str, context: List[Any], response_text: str) -> Dict[str, Any]:
        """
        Executes NeMo output generated response safety rails checks.
        """
        if self.__class__._evaluation_mode:
            return {
                "decision": "ALLOW",
                "reason": "Evaluation mode: guardrail bypassed",
            }
        if not self.is_available():
            return {"decision": "ALLOW", "reason": "Guardrail service not available"}

        try:
            from src.guardrails.config.actions import calculate_grounding_score

            context_dicts = []
            for chunk in context:
                text = getattr(chunk, "text", "")
                if not text and isinstance(chunk, dict):
                    text = chunk.get("text", "")
                context_dicts.append({"text": text})

            ground_res = calculate_grounding_score(response_text, context_dicts)
            score = ground_res.get("grounding_score", 1.0)

            if score < 0.15:
                return {
                    "decision": "BLOCK",
                    "reason": f"Output grounding score ({score:.2f}) below threshold (0.15).",
                    "rule": "check_output_safety",
                }

            return {"decision": "ALLOW", "reason": "Output is safe."}
        except Exception as e:
            logger.error("NeMo check_output failed or timed out: %s", e)
            return {"decision": "ALLOW", "reason": f"Guardrail fallback: {str(e)}"}

