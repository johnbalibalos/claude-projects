"""
Data models for MCP testing framework.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime
from enum import Enum


class TestCaseType(Enum):
    """Classification of test case difficulty."""
    IN_DISTRIBUTION = "in_distribution"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    ADVERSARIAL = "adversarial"
    EDGE_CASE = "edge_case"


@dataclass
class TestCase:
    """
    A single test case for evaluation.

    Attributes:
        id: Unique identifier
        prompt: The prompt to send to the model
        ground_truth: Expected output (format depends on evaluator)
        case_type: Classification for stratified analysis
        metadata: Additional info (e.g., source, difficulty)
    """
    id: str
    prompt: str
    ground_truth: Any
    case_type: TestCaseType = TestCaseType.IN_DISTRIBUTION
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "case_type": self.case_type.value,
            "metadata": self.metadata
        }


@dataclass
class Condition:
    """
    An experimental condition to test.

    Attributes:
        name: Condition name (e.g., "baseline", "mcp_enabled")
        tools_enabled: Whether MCP tools are available
        tools: List of tool definitions (Anthropic format)
        context: Additional context to prepend to prompts
        weight: For importance weighting in analysis
    """
    name: str
    tools_enabled: bool = False
    tools: list[dict] = field(default_factory=list)
    context: str = ""
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tools_enabled": self.tools_enabled,
            "tools_count": len(self.tools),
            "has_context": bool(self.context),
            "weight": self.weight,
            "metadata": self.metadata
        }


@dataclass
class TrialResult:
    """
    Result of a single trial (one test case + one condition).

    Attributes:
        condition_name: Which condition was tested
        test_case_id: Which test case was used
        test_case_type: Classification of test case
        raw_response: Full model response text
        extracted_output: Parsed output from response
        tool_calls: List of tool calls made
        scores: Dict of metric scores from evaluator
        latency_seconds: Time taken for API call(s)
        input_tokens: Tokens in input
        output_tokens: Tokens in output
        error: Error message if failed
        timestamp: When trial was run
    """
    condition_name: str
    test_case_id: str
    test_case_type: str
    raw_response: str = ""
    extracted_output: Any = None
    tool_calls: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def tool_calls_count(self) -> int:
        return len(self.tool_calls)


@dataclass
class StudyResults:
    """
    Complete results from an ablation study.
    """
    study_name: str
    model: str
    started_at: str
    completed_at: str
    trials: list[TrialResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "study_name": self.study_name,
            "model": self.model,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "trials": [t.to_dict() for t in self.trials],
            "metadata": self.metadata
        }

    def filter_by_condition(self, condition_name: str) -> list[TrialResult]:
        return [t for t in self.trials if t.condition_name == condition_name]

    def filter_by_case_type(self, case_type: str) -> list[TrialResult]:
        return [t for t in self.trials if t.test_case_type == case_type]

    def success_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.success) / len(self.trials)

    def avg_score(self, metric: str) -> float:
        scores = [t.scores.get(metric, 0) for t in self.trials if t.success]
        return sum(scores) / len(scores) if scores else 0.0
