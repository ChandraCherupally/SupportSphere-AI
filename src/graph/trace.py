"""
ExecutionTrace — single source of truth for per-ticket pipeline observability.

Each LangGraph node appends exactly one StageRecord.
The Streamlit timeline renders directly from ExecutionTrace.to_timeline().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageRecord:
    """
    Immutable execution record for one pipeline stage.
    Appended by each LangGraph node after it completes.
    """

    stage: str
    """Stage key: input_guardrails, intent_routing, retrieval, retrieval_guardrails, generation, output_guardrails"""

    execution_mode: str
    """Human-readable mode: Safety Engine, Rule Engine, LLM, Deterministic, Hybrid Search, Template"""

    provider: str
    """LLM provider (google, openai, groq) or N/A / system for non-LLM stages."""

    model: str
    """Model name or N/A / template for non-LLM stages."""

    llm_called: bool
    """True when a real LLM API call was made during this stage."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    decision: str = "PASS"
    """Stage outcome: ALLOW, BLOCK, WARN, PASS, REGENERATE"""

    warnings: List[str] = field(default_factory=list)

    # Optional stage-specific metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict for Pydantic models and JSON output."""
        d = {
            "stage": self.stage,
            "execution_mode": self.execution_mode,
            "provider": self.provider,
            "model": self.model,
            "llm_called": self.llm_called,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "decision": self.decision,
            "warnings": self.warnings,
        }
        d.update(self.extra)
        return d


class ExecutionTrace:
    """
    Ordered collection of StageRecords for one ticket execution.
    Single source of truth for observability — no other dicts needed.
    """

    def __init__(self) -> None:
        self._records: List[StageRecord] = []

    def append(self, record: StageRecord) -> None:
        """Append a completed stage record."""
        self._records.append(record)

    def get(self, stage: str) -> Optional[StageRecord]:
        """Return the most recent record for the given stage key, or None."""
        for r in reversed(self._records):
            if r.stage == stage:
                return r
        return None

    @property
    def records(self) -> List[StageRecord]:
        return list(self._records)

    def to_timeline(self) -> List[Dict[str, Any]]:
        """Return all records as plain dicts — consumed directly by the Streamlit timeline."""
        return [r.to_dict() for r in self._records]

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return backward-compatible dict keyed by stage name (for logger.py)."""
        result: Dict[str, Dict[str, Any]] = {}
        for r in self._records:
            result[r.stage] = r.to_dict()
        return result

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialise to a JSON-safe list for storage in Pydantic AgentResponse."""
        return self.to_timeline()

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "ExecutionTrace":
        """Reconstruct from a serialised list (e.g., loaded from AgentResponse)."""
        trace = cls()
        for d in (data or []):
            trace.append(
                StageRecord(
                    stage=d.get("stage", ""),
                    execution_mode=d.get("execution_mode", ""),
                    provider=d.get("provider", "N/A"),
                    model=d.get("model", "N/A"),
                    llm_called=d.get("llm_called", False),
                    input_tokens=d.get("input_tokens", 0),
                    output_tokens=d.get("output_tokens", 0),
                    latency_ms=d.get("latency_ms", 0.0),
                    cost=d.get("cost", 0.0),
                    decision=d.get("decision", "PASS"),
                    warnings=d.get("warnings", []),
                    extra={
                        k: v
                        for k, v in d.items()
                        if k
                        not in {
                            "stage",
                            "execution_mode",
                            "provider",
                            "model",
                            "llm_called",
                            "input_tokens",
                            "output_tokens",
                            "latency_ms",
                            "cost",
                            "decision",
                            "warnings",
                        }
                    },
                )
            )
        return trace
