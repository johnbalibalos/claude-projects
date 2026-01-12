"""
Task failure detection for gating predictions.

Detects when models produce meta-commentary or questions instead of
actual gating hierarchies. This represents a fundamental task failure
rather than a scoring issue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TaskFailureType(Enum):
    """Types of task failures."""
    NONE = "none"
    META_QUESTIONS = "meta_questions"  # "What markers are being used?"
    REFUSAL = "refusal"  # "I cannot predict without more information"
    INSTRUCTIONS = "instructions"  # "Here's how you would create a hierarchy..."
    EMPTY = "empty"  # No meaningful content
    MALFORMED = "malformed"  # Syntactically broken


@dataclass
class TaskFailureResult:
    """Result of task failure detection."""
    is_failure: bool
    failure_type: TaskFailureType
    confidence: float  # 0-1, how confident we are this is a failure
    evidence: list[str]  # Specific phrases that triggered detection
    gate_count: int  # Number of valid-looking gates found


# Patterns that indicate meta-questions (asking for more info)
META_QUESTION_PATTERNS = [
    r"which\s+(fluorochromes?|channels?|markers?)\s+(are|is|being)",
    r"what\s+(is|are)\s+(the|your)\s+(research|experimental)",
    r"what\s+(cell\s+)?populations?\s+(of\s+interest|are\s+you|do\s+you)",
    r"what\s+specific\s+markers",
    r"what\s+is\s+the\s+(sample|tissue)\s+type",
    r"(need|require)s?\s+(more\s+)?(information|context|details)",
    r"please\s+(provide|specify|clarify)",
    r"could\s+you\s+(specify|clarify|tell|provide)",
    r"without\s+(knowing|more\s+information)",
    r"what\s+(markers?|cells?|gates?)\s+(do\s+you|should\s+i|would\s+you)",
    r"interested\s+in\s+(\w+ing|gate|identify|target)",
]

# Patterns that indicate refusal to complete task
REFUSAL_PATTERNS = [
    r"(cannot|can't|unable\s+to)\s+(predict|determine|create|provide)",
    r"(impossible|not\s+possible)\s+to\s+(predict|determine)",
    r"insufficient\s+(information|data|context)",
    r"would\s+need\s+(to\s+know|more)",
    r"this\s+is\s+(difficult|challenging)\s+without",
]

# Patterns that indicate instructional/meta content
INSTRUCTION_PATTERNS = [
    r"here('s|\s+is)\s+how\s+(you|one)\s+(would|could|might)",
    r"the\s+typical\s+approach\s+(would|is)",
    r"generally,?\s+(you|one)\s+would",
    r"a\s+common\s+strategy\s+(is|would\s+be)",
    r"depends\s+on\s+(the|your)",
]

# Patterns that look like actual gate names (positive signals)
GATE_NAME_PATTERNS = [
    r"\bsinglets?\b",
    r"\blive\s*(cells?)?\b",
    r"\blymphocytes?\b",
    r"\bcd\d+[+-]?\b",
    r"\bt\s*cells?\b",
    r"\bb\s*cells?\b",
    r"\bnk\s*cells?\b",
    r"\bmonocytes?\b",
    r"\bdendritic\b",
    r"\bleukocytes?\b",
    r"\bgranulocytes?\b",
]


def detect_task_failure(
    response: str,
    parsed_hierarchy: dict | None = None,
) -> TaskFailureResult:
    """
    Detect if an LLM response represents a task failure.

    Args:
        response: Raw LLM response text
        parsed_hierarchy: Optionally, the parsed hierarchy if available

    Returns:
        TaskFailureResult with failure type and evidence
    """
    if not response or not response.strip():
        return TaskFailureResult(
            is_failure=True,
            failure_type=TaskFailureType.EMPTY,
            confidence=1.0,
            evidence=["Empty response"],
            gate_count=0,
        )

    response_lower = response.lower()
    evidence = []

    # Check for meta-questions
    meta_score = 0
    for pattern in META_QUESTION_PATTERNS:
        matches = re.findall(pattern, response_lower)
        if matches:
            meta_score += len(matches)
            evidence.append(f"Meta-question: '{matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])}'")

    # Check for refusals
    refusal_score = 0
    for pattern in REFUSAL_PATTERNS:
        matches = re.findall(pattern, response_lower)
        if matches:
            refusal_score += len(matches)
            evidence.append(f"Refusal: '{matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])}'")

    # Check for instructions
    instruction_score = 0
    for pattern in INSTRUCTION_PATTERNS:
        matches = re.findall(pattern, response_lower)
        if matches:
            instruction_score += len(matches)
            evidence.append(f"Instructional: '{matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])}'")

    # Count valid gate names as positive signal
    gate_count = 0
    for pattern in GATE_NAME_PATTERNS:
        matches = re.findall(pattern, response_lower)
        gate_count += len(matches)

    # Check parsed hierarchy for meta-content in gate names
    if parsed_hierarchy:
        hierarchy_evidence = _check_hierarchy_for_meta(parsed_hierarchy)
        evidence.extend(hierarchy_evidence)
        if hierarchy_evidence:
            meta_score += len(hierarchy_evidence) * 2  # Weight hierarchy issues higher

    # Determine failure type and confidence
    total_failure_signals = meta_score + refusal_score + instruction_score

    # Strong positive signal from gates reduces failure confidence
    # If we have enough valid gates, meta-commentary is likely just preamble
    if gate_count >= 5 and total_failure_signals <= 1:
        return TaskFailureResult(
            is_failure=False,
            failure_type=TaskFailureType.NONE,
            confidence=0.0,
            evidence=[],
            gate_count=gate_count,
        )

    # Even stronger: many gates override moderate meta-commentary
    # A response with 10+ gates and meta-phrases is likely a valid hierarchy
    # with explanatory text, not a task failure
    if gate_count >= 10:
        return TaskFailureResult(
            is_failure=False,
            failure_type=TaskFailureType.NONE,
            confidence=0.0,
            evidence=[],
            gate_count=gate_count,
        )

    # Determine primary failure type
    # Require BOTH high meta_score AND low gate count to classify as failure
    # Previously: meta_score >= 2 alone triggered failure, ignoring valid content
    if meta_score >= 2 and gate_count < 5:
        confidence = min(1.0, meta_score / 3 + (0.3 if gate_count < 3 else 0))
        return TaskFailureResult(
            is_failure=True,
            failure_type=TaskFailureType.META_QUESTIONS,
            confidence=confidence,
            evidence=evidence[:5],  # Limit evidence
            gate_count=gate_count,
        )

    # Low gate count with any meta signal is suspicious
    if meta_score >= 1 and gate_count < 3:
        confidence = min(1.0, meta_score / 3 + 0.3)
        return TaskFailureResult(
            is_failure=True,
            failure_type=TaskFailureType.META_QUESTIONS,
            confidence=confidence,
            evidence=evidence[:5],
            gate_count=gate_count,
        )

    if refusal_score >= 1:
        confidence = min(1.0, refusal_score / 2 + 0.3)
        return TaskFailureResult(
            is_failure=True,
            failure_type=TaskFailureType.REFUSAL,
            confidence=confidence,
            evidence=evidence[:5],
            gate_count=gate_count,
        )

    if instruction_score >= 2 and gate_count < 5:
        confidence = min(1.0, instruction_score / 3)
        return TaskFailureResult(
            is_failure=True,
            failure_type=TaskFailureType.INSTRUCTIONS,
            confidence=confidence,
            evidence=evidence[:5],
            gate_count=gate_count,
        )

    # No failure detected
    return TaskFailureResult(
        is_failure=False,
        failure_type=TaskFailureType.NONE,
        confidence=0.0,
        evidence=[],
        gate_count=gate_count,
    )


def _check_hierarchy_for_meta(hierarchy: dict, path: str = "") -> list[str]:
    """Check hierarchy nodes for meta-content in gate names."""
    evidence = []

    name = hierarchy.get("name", "")
    name_lower = name.lower()

    # Check if gate name looks like a question or meta-content
    meta_in_name = [
        ("which", "Question word in gate name"),
        ("what", "Question word in gate name"),
        ("how to", "Instructional in gate name"),
        ("need to know", "Meta-content in gate name"),
        ("depends on", "Meta-content in gate name"),
        ("research question", "Meta-content in gate name"),
        ("populations of interest", "Meta-content in gate name"),
    ]

    for trigger, description in meta_in_name:
        if trigger in name_lower:
            evidence.append(f"{description}: '{name[:50]}...' at {path or 'root'}")

    # Recurse to children
    for i, child in enumerate(hierarchy.get("children", [])):
        child_path = f"{path}.children[{i}]" if path else f"children[{i}]"
        evidence.extend(_check_hierarchy_for_meta(child, child_path))

    return evidence


def compute_task_failure_rate(
    results: list[dict],
    include_parse_failures: bool = True,
) -> dict[str, float | int]:
    """
    Compute task failure metrics across a set of results.

    Args:
        results: List of result dicts with 'raw_response' and optionally 'parsed_hierarchy'
        include_parse_failures: Whether to count parse failures as task failures

    Returns:
        Dict with failure rates by type
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "task_failure_rate": 0.0}

    failures_by_type = dict.fromkeys(TaskFailureType, 0)

    for result in results:
        response = result.get("raw_response", "")
        hierarchy = result.get("parsed_hierarchy")
        parse_success = result.get("parse_success", True)

        if not parse_success and include_parse_failures:
            failures_by_type[TaskFailureType.MALFORMED] += 1
            continue

        detection = detect_task_failure(response, hierarchy)
        if detection.is_failure:
            failures_by_type[detection.failure_type] += 1

    # Compute rates
    total_failures = sum(v for k, v in failures_by_type.items() if k != TaskFailureType.NONE)

    return {
        "total": total,
        "task_failure_rate": total_failures / total,
        "task_failure_count": total_failures,
        "meta_questions": failures_by_type[TaskFailureType.META_QUESTIONS],
        "refusals": failures_by_type[TaskFailureType.REFUSAL],
        "instructions": failures_by_type[TaskFailureType.INSTRUCTIONS],
        "empty": failures_by_type[TaskFailureType.EMPTY],
        "malformed": failures_by_type[TaskFailureType.MALFORMED],
    }
