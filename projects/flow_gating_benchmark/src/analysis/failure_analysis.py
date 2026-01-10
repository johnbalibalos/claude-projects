"""
Failure mode analysis for gating predictions.

Categorizes and analyzes prediction failures to understand
systematic LLM limitations in flow cytometry gating.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from evaluation.scorer import ScoringResult


class FailureCategory(str, Enum):
    """Categories of prediction failures."""

    PARSE_FAILURE = "parse_failure"
    HALLUCINATED_MARKERS = "hallucinated_markers"
    MISSING_CRITICAL_GATES = "missing_critical_gates"
    WRONG_HIERARCHY_STRUCTURE = "wrong_hierarchy_structure"
    MISSING_QC_GATES = "missing_qc_gates"
    WRONG_GATING_ORDER = "wrong_gating_order"
    IMMUNOLOGICALLY_IMPLAUSIBLE = "immunologically_implausible"
    INCOMPLETE_HIERARCHY = "incomplete_hierarchy"
    OVERLY_COMPLEX = "overly_complex"


# QC gates that should typically be present
QC_GATES = {"time", "singlets", "live", "live/dead", "doublets"}

# Critical lineage gates
LINEAGE_GATES = {"cd45+", "lymphocytes", "t cells", "b cells", "nk cells", "monocytes"}


@dataclass
class FailureInstance:
    """A single failure instance."""

    category: FailureCategory
    test_case_id: str
    model: str
    condition: str
    details: str
    severity: int = 1  # 1-3, with 3 being most severe


@dataclass
class FailureAnalysis:
    """Complete failure analysis results."""

    total_predictions: int = 0
    total_failures: int = 0
    failures_by_category: dict[FailureCategory, int] = field(default_factory=dict)
    failure_instances: list[FailureInstance] = field(default_factory=list)
    failures_by_model: dict[str, dict[FailureCategory, int]] = field(default_factory=dict)
    failures_by_condition: dict[str, dict[FailureCategory, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_predictions": self.total_predictions,
            "total_failures": self.total_failures,
            "failure_rate": self.total_failures / self.total_predictions if self.total_predictions else 0,
            "failures_by_category": {
                cat.value: count for cat, count in self.failures_by_category.items()
            },
            "failures_by_model": {
                model: {cat.value: count for cat, count in cats.items()}
                for model, cats in self.failures_by_model.items()
            },
            "failures_by_condition": {
                cond: {cat.value: count for cat, count in cats.items()}
                for cond, cats in self.failures_by_condition.items()
            },
        }


def categorize_failure(result: ScoringResult) -> list[FailureInstance]:
    """
    Categorize failures in a single prediction.

    Args:
        result: ScoringResult to analyze

    Returns:
        List of FailureInstance for each failure found
    """
    failures = []
    base_info = {
        "test_case_id": result.test_case_id,
        "model": result.model,
        "condition": result.condition,
    }

    # Check for parse failure
    if not result.parse_success:
        failures.append(FailureInstance(
            category=FailureCategory.PARSE_FAILURE,
            details=result.parse_error or "Unknown parse error",
            severity=3,
            **base_info,
        ))
        return failures  # Can't analyze further

    if result.evaluation is None:
        return failures

    eval_result = result.evaluation

    # Check for hallucinated markers
    if eval_result.hallucinated_gates:
        failures.append(FailureInstance(
            category=FailureCategory.HALLUCINATED_MARKERS,
            details=f"Hallucinated gates: {', '.join(eval_result.hallucinated_gates[:5])}",
            severity=2,
            **base_info,
        ))

    # Check for missing critical gates
    if eval_result.missing_critical:
        failures.append(FailureInstance(
            category=FailureCategory.MISSING_CRITICAL_GATES,
            details=f"Missing critical: {', '.join(eval_result.missing_critical)}",
            severity=3,
            **base_info,
        ))

    # Check for structure errors
    if eval_result.structure_accuracy < 0.5:
        failures.append(FailureInstance(
            category=FailureCategory.WRONG_HIERARCHY_STRUCTURE,
            details=f"Structure accuracy: {eval_result.structure_accuracy:.1%}",
            severity=2,
            **base_info,
        ))

    # Check for missing QC gates
    predicted_lower = {g.lower() for g in eval_result.predicted_gates}
    missing_qc = QC_GATES - predicted_lower
    if missing_qc:
        failures.append(FailureInstance(
            category=FailureCategory.MISSING_QC_GATES,
            details=f"Missing QC gates: {', '.join(missing_qc)}",
            severity=2,
            **base_info,
        ))

    # Check for incomplete hierarchy (low recall)
    if eval_result.hierarchy_recall < 0.5:
        failures.append(FailureInstance(
            category=FailureCategory.INCOMPLETE_HIERARCHY,
            details=f"Only found {eval_result.hierarchy_recall:.1%} of expected gates",
            severity=2,
            **base_info,
        ))

    # Check for overly complex (low precision, high extra gates)
    if eval_result.hierarchy_precision < 0.5 and len(eval_result.extra_gates) > 5:
        failures.append(FailureInstance(
            category=FailureCategory.OVERLY_COMPLEX,
            details=f"Added {len(eval_result.extra_gates)} extra gates",
            severity=1,
            **base_info,
        ))

    return failures


def analyze_failures(results: list[ScoringResult]) -> FailureAnalysis:
    """
    Analyze failures across all results.

    Args:
        results: List of ScoringResults to analyze

    Returns:
        Complete FailureAnalysis
    """
    analysis = FailureAnalysis(total_predictions=len(results))

    for result in results:
        failures = categorize_failure(result)

        if failures:
            analysis.total_failures += 1

        for failure in failures:
            # Count by category
            if failure.category not in analysis.failures_by_category:
                analysis.failures_by_category[failure.category] = 0
            analysis.failures_by_category[failure.category] += 1

            # Count by model
            if failure.model not in analysis.failures_by_model:
                analysis.failures_by_model[failure.model] = {}
            if failure.category not in analysis.failures_by_model[failure.model]:
                analysis.failures_by_model[failure.model][failure.category] = 0
            analysis.failures_by_model[failure.model][failure.category] += 1

            # Count by condition
            if failure.condition not in analysis.failures_by_condition:
                analysis.failures_by_condition[failure.condition] = {}
            if failure.category not in analysis.failures_by_condition[failure.condition]:
                analysis.failures_by_condition[failure.condition][failure.category] = 0
            analysis.failures_by_condition[failure.condition][failure.category] += 1

            # Store instance
            analysis.failure_instances.append(failure)

    return analysis


def generate_failure_report(analysis: FailureAnalysis) -> str:
    """
    Generate a text report of failure analysis.

    Args:
        analysis: FailureAnalysis results

    Returns:
        Formatted text report
    """
    lines = [
        "=" * 60,
        "FAILURE MODE ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Total predictions: {analysis.total_predictions}",
        f"Predictions with failures: {analysis.total_failures}",
        f"Failure rate: {analysis.total_failures / analysis.total_predictions:.1%}"
        if analysis.total_predictions else "N/A",
        "",
        "-" * 60,
        "FAILURES BY CATEGORY",
        "-" * 60,
    ]

    # Sort by frequency
    sorted_categories = sorted(
        analysis.failures_by_category.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for category, count in sorted_categories:
        pct = count / analysis.total_predictions * 100 if analysis.total_predictions else 0
        lines.append(f"  {category.value}: {count} ({pct:.1f}%)")

    lines.extend([
        "",
        "-" * 60,
        "FAILURES BY MODEL",
        "-" * 60,
    ])

    for model, categories in analysis.failures_by_model.items():
        lines.append(f"\n  {model}:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"    {cat.value}: {count}")

    lines.extend([
        "",
        "-" * 60,
        "TOP FAILURE EXAMPLES",
        "-" * 60,
    ])

    # Show most severe failures
    severe = sorted(
        analysis.failure_instances,
        key=lambda x: x.severity,
        reverse=True,
    )[:10]

    for failure in severe:
        lines.append(
            f"\n  [{failure.category.value}] {failure.test_case_id} ({failure.model})"
        )
        lines.append(f"    {failure.details}")

    return "\n".join(lines)


def get_failure_patterns(analysis: FailureAnalysis) -> dict[str, Any]:
    """
    Identify patterns in failures.

    Args:
        analysis: FailureAnalysis results

    Returns:
        Dictionary of identified patterns
    """
    patterns = {}

    # Pattern 1: Model-specific failures
    model_patterns = {}
    for model, categories in analysis.failures_by_model.items():
        total = sum(categories.values())
        if total > 0:
            dominant = max(categories.items(), key=lambda x: x[1])
            if dominant[1] / total > 0.4:  # >40% of failures are this type
                model_patterns[model] = {
                    "dominant_failure": dominant[0].value,
                    "percentage": dominant[1] / total,
                }
    patterns["model_specific"] = model_patterns

    # Pattern 2: Context-level effects
    context_patterns = {}
    for condition, categories in analysis.failures_by_condition.items():
        if "minimal" in condition.lower():
            context_patterns["minimal"] = sum(categories.values())
        elif "standard" in condition.lower():
            context_patterns["standard"] = sum(categories.values())
        elif "rich" in condition.lower():
            context_patterns["rich"] = sum(categories.values())
    patterns["context_effects"] = context_patterns

    # Pattern 3: Common failure combinations
    failure_combos: Counter = Counter()
    instance_groups: dict[tuple[str, str], list[FailureCategory]] = {}

    for instance in analysis.failure_instances:
        key = (instance.test_case_id, instance.condition)
        if key not in instance_groups:
            instance_groups[key] = []
        instance_groups[key].append(instance.category)

    for categories in instance_groups.values():
        if len(categories) > 1:
            combo = tuple(sorted(c.value for c in categories))
            failure_combos[combo] += 1

    patterns["common_combinations"] = dict(failure_combos.most_common(5))

    return patterns
