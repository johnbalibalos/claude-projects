"""
Data models for the modular hypothesis pipeline.

These models define the configuration and results for hypothesis testing experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar


class ReasoningType(Enum):
    """Types of reasoning strategies."""
    DIRECT = "direct"
    COT = "cot"  # Chain of Thought
    WOT = "wot"  # Web of Thought / Tree of Thought
    FEW_SHOT = "few_shot"
    SELF_CONSISTENCY = "self_consistency"
    REACT = "react"  # Reasoning + Acting


class ContextLevel(Enum):
    """Levels of context richness."""
    NONE = "none"
    MINIMAL = "minimal"
    STANDARD = "standard"
    RICH = "rich"
    ORACLE = "oracle"  # Perfect context (for upper-bound testing)


class RAGMode(Enum):
    """RAG retrieval modes."""
    NONE = "none"
    VECTOR = "vector"
    HYBRID = "hybrid"  # Vector + keyword
    ORACLE = "oracle"  # Perfect retrieval (for upper-bound testing)
    NEGATIVE = "negative"  # Wrong documents (for lower-bound testing)


class DataSource(Enum):
    """Data source types for experiment organization."""
    SYNTHETIC = "synthetic"  # Generated/artificial test cases
    REAL = "real"  # Real-world dataset
    MIXED = "mixed"  # Combination of synthetic and real


@dataclass
class ToolConfig:
    """Configuration for a single tool/MCP."""

    name: str
    description: str
    input_schema: dict[str, Any]
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class HypothesisCondition:
    """
    A single experimental condition combining all dimensions.

    This is the "treatment" in the hypothesis test - a specific
    combination of reasoning strategy, context level, RAG mode, and tools.
    """

    name: str

    # Reasoning dimension
    reasoning_type: ReasoningType = ReasoningType.DIRECT
    reasoning_config: dict[str, Any] = field(default_factory=dict)

    # Context dimension
    context_level: ContextLevel = ContextLevel.STANDARD
    context_config: dict[str, Any] = field(default_factory=dict)

    # RAG dimension
    rag_mode: RAGMode = RAGMode.NONE
    rag_config: dict[str, Any] = field(default_factory=dict)

    # Tools dimension
    tools_enabled: bool = False
    tool_names: list[str] = field(default_factory=list)  # Which tools to enable

    # Model configuration
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Additional system prompt additions
    system_prompt_additions: str = ""

    # Metadata for analysis
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def condition_id(self) -> str:
        """Unique identifier for this condition."""
        parts = [
            self.model.split("-")[1] if "-" in self.model else self.model,
            self.reasoning_type.value,
            self.context_level.value,
            self.rag_mode.value,
        ]
        if self.tools_enabled:
            parts.append("tools")
        return "_".join(parts)


@dataclass
class TrialInput:
    """Input for a single trial."""

    id: str
    raw_input: Any  # The original test case data
    prompt: str  # Base prompt before strategy/context applied
    ground_truth: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_raw: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Args:
            include_raw: Whether to include raw_input (may be large/complex)

        Returns:
            Serializable dictionary
        """
        result = {
            "id": self.id,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }

        # Handle raw_input - try to serialize, fall back to repr
        if include_raw:
            try:
                # Try JSON serialization first
                json.dumps(self.raw_input, default=str)
                result["raw_input"] = self.raw_input
            except (TypeError, ValueError):
                # Fall back to string representation
                result["raw_input_repr"] = repr(self.raw_input)[:1000]
                result["raw_input_type"] = type(self.raw_input).__name__

        # Handle ground_truth similarly
        try:
            json.dumps(self.ground_truth, default=str)
            result["ground_truth"] = self.ground_truth
        except (TypeError, ValueError):
            result["ground_truth_repr"] = repr(self.ground_truth)[:1000]
            result["ground_truth_type"] = type(self.ground_truth).__name__

        return result


@dataclass
class TrialResult:
    """Result of a single trial."""

    trial_id: str
    condition_name: str

    # Timing
    start_time: datetime
    end_time: datetime
    latency_seconds: float

    # Model interaction
    raw_response: str = ""
    extracted_output: Any = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Evaluation scores
    scores: dict[str, float] = field(default_factory=dict)

    # Error tracking
    error: str | None = None

    # Context used (for debugging)
    context_used: str = ""
    rag_documents: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "condition_name": self.condition_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "latency_seconds": self.latency_seconds,
            "raw_response": self.raw_response,
            "extracted_output": self.extracted_output,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "scores": self.scores,
            "error": self.error,
            "context_used": self.context_used[:500] if self.context_used else "",  # Truncate
            "rag_documents": self.rag_documents,
        }


@dataclass
class ExperimentResults:
    """Complete results from an experiment."""

    experiment_name: str
    started_at: datetime
    completed_at: datetime | None = None

    conditions: list[HypothesisCondition] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)

    # Aggregated metrics
    metrics_by_condition: dict[str, dict[str, float]] = field(default_factory=dict)

    def filter_by_condition(self, condition_name: str) -> list[TrialResult]:
        """Get trials for a specific condition."""
        return [t for t in self.trials if t.condition_name == condition_name]

    def success_rate(self, condition_name: str | None = None) -> float:
        """Calculate success rate overall or for a condition."""
        trials = self.filter_by_condition(condition_name) if condition_name else self.trials
        if not trials:
            return 0.0
        return sum(1 for t in trials if t.success) / len(trials)

    def avg_score(self, metric: str, condition_name: str | None = None) -> float:
        """Get average score for a metric."""
        trials = self.filter_by_condition(condition_name) if condition_name else self.trials
        scores = [t.scores.get(metric, 0) for t in trials if t.success and metric in t.scores]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "n_conditions": len(self.conditions),
            "n_trials": len(self.trials),
            "trials": [t.to_dict() for t in self.trials],
            "metrics_by_condition": self.metrics_by_condition,
        }


# Type variable for generic test cases
T = TypeVar("T")
