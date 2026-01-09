"""
Schemas for reasoning benchmark test cases.

These test cases evaluate genuine biological reasoning, not pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReasoningTestType(Enum):
    """Types of reasoning tests."""

    LINEAGE_NEGATIVE = "lineage_negative"  # Dump channel / exclusion gating
    BIOLOGICAL_IMPOSSIBILITY = "biological_impossibility"  # Doublet/artifact detection
    CONTEXT_SWITCH = "context_switch"  # Tissue-specific adaptations
    FMO_LOGIC = "fmo_logic"  # Gating boundary determination
    PANEL_SUBSET = "panel_subset"  # Instrument constraint optimization


class ReasoningQuality(Enum):
    """Quality levels for reasoning responses."""

    FAIL = "fail"  # Pattern matching response, misses key insight
    PARTIAL = "partial"  # Some reasoning but incomplete
    PASS = "pass"  # Demonstrates required biological reasoning


@dataclass
class ExpectedBehavior:
    """
    Defines expected model behavior for a reasoning test.

    Separates pattern-matching failures from reasoning successes.
    """

    # What a pattern matcher would incorrectly produce
    fail_patterns: list[str] = field(default_factory=list)
    fail_description: str = ""

    # What demonstrates partial understanding
    partial_patterns: list[str] = field(default_factory=list)
    partial_description: str = ""

    # What demonstrates genuine reasoning
    pass_requirements: list[str] = field(default_factory=list)
    pass_description: str = ""

    # Key concepts that MUST be mentioned for full credit
    required_concepts: list[str] = field(default_factory=list)

    # Concepts that indicate pattern matching (negative indicators)
    pattern_matching_indicators: list[str] = field(default_factory=list)


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating a reasoning response."""

    # Required elements in the gating hierarchy (order matters)
    required_gates_in_order: list[str] = field(default_factory=list)

    # Gates that must come BEFORE others (prerequisite relationships)
    gate_prerequisites: dict[str, list[str]] = field(default_factory=dict)

    # Gates that must be EXCLUDED or mentioned as exclusion
    exclusion_gates: list[str] = field(default_factory=list)

    # Concepts that must be mentioned in reasoning
    required_reasoning_concepts: list[str] = field(default_factory=list)

    # Red flags that indicate pattern matching failure
    failure_indicators: list[str] = field(default_factory=list)

    # Bonus concepts that indicate deep understanding
    bonus_concepts: list[str] = field(default_factory=list)


@dataclass
class ReasoningTestCase:
    """
    A test case for the reasoning benchmark.

    Unlike pattern-matching tests, these have:
    - Clear biological rationale for correct answers
    - Predictable failure modes for pattern matchers
    - Evaluation based on reasoning quality, not exact matches
    """

    # Required fields (no defaults)
    test_id: str
    test_type: ReasoningTestType
    prompt: str

    # Optional fields with defaults
    difficulty: str = "medium"  # easy, medium, hard
    markers: list[str] = field(default_factory=list)
    target_population: str = ""
    tissue_context: str = "PBMC"  # Default: peripheral blood

    # Additional constraints or context
    constraints: dict[str, Any] = field(default_factory=dict)

    # Expected behaviors
    expected: ExpectedBehavior = field(default_factory=ExpectedBehavior)

    # Evaluation criteria
    criteria: EvaluationCriteria = field(default_factory=EvaluationCriteria)

    # Explanation for test designers
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "test_type": self.test_type.value,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "markers": self.markers,
            "target_population": self.target_population,
            "tissue_context": self.tissue_context,
            "constraints": self.constraints,
            "expected": {
                "fail_patterns": self.expected.fail_patterns,
                "fail_description": self.expected.fail_description,
                "pass_requirements": self.expected.pass_requirements,
                "pass_description": self.expected.pass_description,
                "required_concepts": self.expected.required_concepts,
            },
            "criteria": {
                "required_gates_in_order": self.criteria.required_gates_in_order,
                "gate_prerequisites": self.criteria.gate_prerequisites,
                "exclusion_gates": self.criteria.exclusion_gates,
                "required_reasoning_concepts": self.criteria.required_reasoning_concepts,
                "failure_indicators": self.criteria.failure_indicators,
            },
            "rationale": self.rationale,
        }


@dataclass
class ReasoningResult:
    """Result of evaluating a reasoning test."""

    test_id: str
    test_type: ReasoningTestType
    quality: ReasoningQuality

    # Scoring details
    gates_correct: bool = False
    order_correct: bool = False
    exclusions_present: bool = False
    reasoning_concepts_found: list[str] = field(default_factory=list)
    reasoning_concepts_missing: list[str] = field(default_factory=list)
    failure_indicators_found: list[str] = field(default_factory=list)
    bonus_concepts_found: list[str] = field(default_factory=list)

    # Raw data
    model_response: str = ""
    parsed_hierarchy: dict | None = None

    # Scores
    reasoning_score: float = 0.0  # 0-1, based on concept coverage
    structure_score: float = 0.0  # 0-1, based on gate order/prerequisites

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "test_type": self.test_type.value,
            "quality": self.quality.value,
            "gates_correct": self.gates_correct,
            "order_correct": self.order_correct,
            "exclusions_present": self.exclusions_present,
            "reasoning_concepts_found": self.reasoning_concepts_found,
            "reasoning_concepts_missing": self.reasoning_concepts_missing,
            "failure_indicators_found": self.failure_indicators_found,
            "bonus_concepts_found": self.bonus_concepts_found,
            "reasoning_score": self.reasoning_score,
            "structure_score": self.structure_score,
        }
