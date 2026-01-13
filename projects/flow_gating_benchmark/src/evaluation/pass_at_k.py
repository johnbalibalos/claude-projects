"""
Pass@k metrics for evaluating LLM success rates across multiple attempts.

This module implements:
- pass@k: Probability that at least one of k samples passes
- pass^k: Probability that all k samples pass
- Statistical confidence intervals via bootstrapping
- Model comparison utilities

The pass@k metric is particularly useful for:
- Measuring reliability of code generation
- Evaluating multi-attempt success rates
- Comparing consistency across models

Usage:
    from evaluation.pass_at_k import (
        compute_pass_at_k,
        compute_pass_k_power,
        PassAtKResult,
    )

    # For single test case with multiple samples
    results = [True, False, True, True, False]  # 5 attempts
    pass_1 = compute_pass_at_k(results, k=1)  # ~0.6
    pass_3 = compute_pass_at_k(results, k=3)  # Higher

    # For experiment results
    result = compute_pass_at_k_from_results(scoring_results, k=3)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from .scorer import ScoringResult


def _comb(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def compute_pass_at_k(
    results: Sequence[bool],
    k: int,
) -> float:
    """
    Compute pass@k metric for a single test case.

    pass@k estimates the probability that at least one of k
    randomly selected samples passes. It's computed as:

        pass@k = 1 - C(n-c, k) / C(n, k)

    where n = total samples, c = correct samples.

    This is an unbiased estimator that avoids the high variance
    of simply sampling k items and checking if any pass.

    Args:
        results: Sequence of boolean pass/fail results
        k: Number of samples to consider

    Returns:
        pass@k probability (0-1)

    Example:
        >>> compute_pass_at_k([True, False, True, False, False], k=1)
        0.4  # 2 out of 5 pass
        >>> compute_pass_at_k([True, False, True, False, False], k=3)
        0.9  # Much more likely at least one of 3 passes
    """
    n = len(results)
    c = sum(1 for r in results if r)

    if n < k:
        # Not enough samples; use what we have
        if c > 0:
            return 1.0 - (1.0 - c / n) ** k  # Approximate
        return 0.0

    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    # Use exact formula: 1 - C(n-c, k) / C(n, k)
    # Computed more stably as:
    # C(n-c, k) / C(n, k) = product((n-c-i) / (n-i) for i in range(k))
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)

    return 1.0 - prob_all_fail


def compute_pass_k_power(
    results: Sequence[bool],
    k: int,
) -> float:
    """
    Compute pass^k metric (probability all k samples pass).

    pass^k = C(c, k) / C(n, k)

    where n = total samples, c = correct samples.

    This metric measures consistency/reliability - the probability
    that ALL randomly selected k samples are correct.

    Args:
        results: Sequence of boolean pass/fail results
        k: Number of samples to consider

    Returns:
        pass^k probability (0-1)

    Example:
        >>> compute_pass_k_power([True, True, True, False, False], k=2)
        0.3  # C(3,2)/C(5,2) = 3/10
    """
    n = len(results)
    c = sum(1 for r in results if r)

    if n < k:
        # Not enough samples
        if c == n and c >= k:
            return 1.0
        return (c / n) ** k if n > 0 else 0.0

    if c < k:
        return 0.0

    # C(c, k) / C(n, k)
    prob_all_pass = 1.0
    for i in range(k):
        prob_all_pass *= (c - i) / (n - i)

    return prob_all_pass


def compute_pass_at_k_threshold(
    scores: Sequence[float],
    k: int,
    threshold: float = 0.8,
) -> float:
    """
    Compute pass@k using a score threshold to determine pass/fail.

    Args:
        scores: Sequence of scores (e.g., F1 scores)
        k: Number of samples to consider
        threshold: Score threshold for passing

    Returns:
        pass@k probability
    """
    results = [s >= threshold for s in scores]
    return compute_pass_at_k(results, k)


@dataclass
class PassAtKResult:
    """Complete pass@k analysis result."""

    # Core metrics
    pass_at_1: float
    pass_at_k: float
    pass_k_power: float
    k: int

    # Sample statistics
    n_samples: int
    n_passes: int
    pass_rate: float

    # Confidence intervals (from bootstrapping)
    pass_at_1_ci: tuple[float, float] | None = None
    pass_at_k_ci: tuple[float, float] | None = None
    pass_k_power_ci: tuple[float, float] | None = None

    # Additional metrics for different k values
    pass_at_k_curve: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "pass_k_power": self.pass_k_power,
            "k": self.k,
            "n_samples": self.n_samples,
            "n_passes": self.n_passes,
            "pass_rate": self.pass_rate,
            "pass_at_1_ci": self.pass_at_1_ci,
            "pass_at_k_ci": self.pass_at_k_ci,
            "pass_k_power_ci": self.pass_k_power_ci,
            "pass_at_k_curve": self.pass_at_k_curve,
        }


def analyze_pass_at_k(
    results: Sequence[bool],
    k: int = 5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    compute_curve: bool = True,
) -> PassAtKResult:
    """
    Comprehensive pass@k analysis with confidence intervals.

    Args:
        results: Sequence of boolean pass/fail results
        k: Target k value for pass@k
        n_bootstrap: Number of bootstrap iterations for CI
        confidence_level: Confidence level for intervals
        compute_curve: Whether to compute pass@k for multiple k values

    Returns:
        PassAtKResult with all metrics and CIs
    """
    n = len(results)
    c = sum(1 for r in results if r)

    pass_at_1 = compute_pass_at_k(results, 1)
    pass_at_k_val = compute_pass_at_k(results, k)
    pass_k_power_val = compute_pass_k_power(results, k)

    # Bootstrap confidence intervals
    pass_at_1_ci = None
    pass_at_k_ci = None
    pass_k_power_ci = None

    if n_bootstrap > 0 and n > 0:
        pass_at_1_samples = []
        pass_at_k_samples = []
        pass_k_power_samples = []

        results_array = np.array(results, dtype=bool)

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n, size=n)
            resampled = results_array[indices]

            pass_at_1_samples.append(compute_pass_at_k(resampled, 1))
            pass_at_k_samples.append(compute_pass_at_k(resampled, k))
            pass_k_power_samples.append(compute_pass_k_power(resampled, k))

        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        pass_at_1_ci = (
            float(np.percentile(pass_at_1_samples, lower_percentile)),
            float(np.percentile(pass_at_1_samples, upper_percentile)),
        )
        pass_at_k_ci = (
            float(np.percentile(pass_at_k_samples, lower_percentile)),
            float(np.percentile(pass_at_k_samples, upper_percentile)),
        )
        pass_k_power_ci = (
            float(np.percentile(pass_k_power_samples, lower_percentile)),
            float(np.percentile(pass_k_power_samples, upper_percentile)),
        )

    # Compute curve for different k values
    pass_at_k_curve = {}
    if compute_curve:
        for k_val in [1, 2, 3, 5, 10, 20]:
            if k_val <= n:
                pass_at_k_curve[k_val] = compute_pass_at_k(results, k_val)

    return PassAtKResult(
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k_val,
        pass_k_power=pass_k_power_val,
        k=k,
        n_samples=n,
        n_passes=c,
        pass_rate=c / n if n > 0 else 0.0,
        pass_at_1_ci=pass_at_1_ci,
        pass_at_k_ci=pass_at_k_ci,
        pass_k_power_ci=pass_k_power_ci,
        pass_at_k_curve=pass_at_k_curve,
    )


def compute_pass_at_k_from_scores(
    scores: Sequence[float],
    k: int = 5,
    threshold: float = 0.8,
    n_bootstrap: int = 1000,
) -> PassAtKResult:
    """
    Compute pass@k from numeric scores with threshold.

    Args:
        scores: Sequence of scores (e.g., F1 scores)
        k: Target k value
        threshold: Score threshold for passing
        n_bootstrap: Bootstrap iterations for CI

    Returns:
        PassAtKResult
    """
    results = [s >= threshold for s in scores]
    return analyze_pass_at_k(results, k=k, n_bootstrap=n_bootstrap)


@dataclass
class AggregatedPassAtK:
    """Aggregated pass@k results across multiple test cases."""

    # Mean metrics across test cases
    mean_pass_at_1: float
    mean_pass_at_k: float
    mean_pass_k_power: float
    k: int

    # Standard deviations
    std_pass_at_1: float
    std_pass_at_k: float
    std_pass_k_power: float

    # Per-test-case results
    per_test_case: dict[str, PassAtKResult] = field(default_factory=dict)

    # Aggregate statistics
    n_test_cases: int = 0
    total_samples: int = 0
    total_passes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_pass_at_1": self.mean_pass_at_1,
            "mean_pass_at_k": self.mean_pass_at_k,
            "mean_pass_k_power": self.mean_pass_k_power,
            "k": self.k,
            "std_pass_at_1": self.std_pass_at_1,
            "std_pass_at_k": self.std_pass_at_k,
            "std_pass_k_power": self.std_pass_k_power,
            "n_test_cases": self.n_test_cases,
            "total_samples": self.total_samples,
            "total_passes": self.total_passes,
            "per_test_case": {
                tc: r.to_dict() for tc, r in self.per_test_case.items()
            },
        }


def aggregate_pass_at_k(
    results_by_test_case: dict[str, Sequence[bool]],
    k: int = 5,
    n_bootstrap: int = 1000,
) -> AggregatedPassAtK:
    """
    Compute aggregated pass@k across multiple test cases.

    Args:
        results_by_test_case: Dict mapping test case ID to pass/fail results
        k: Target k value
        n_bootstrap: Bootstrap iterations

    Returns:
        AggregatedPassAtK with per-case and aggregate metrics
    """
    per_test_case = {}
    pass_at_1_values = []
    pass_at_k_values = []
    pass_k_power_values = []
    total_samples = 0
    total_passes = 0

    for tc_id, results in results_by_test_case.items():
        result = analyze_pass_at_k(results, k=k, n_bootstrap=n_bootstrap)
        per_test_case[tc_id] = result
        pass_at_1_values.append(result.pass_at_1)
        pass_at_k_values.append(result.pass_at_k)
        pass_k_power_values.append(result.pass_k_power)
        total_samples += result.n_samples
        total_passes += result.n_passes

    n_test_cases = len(results_by_test_case)

    return AggregatedPassAtK(
        mean_pass_at_1=float(np.mean(pass_at_1_values)) if pass_at_1_values else 0.0,
        mean_pass_at_k=float(np.mean(pass_at_k_values)) if pass_at_k_values else 0.0,
        mean_pass_k_power=float(np.mean(pass_k_power_values)) if pass_k_power_values else 0.0,
        k=k,
        std_pass_at_1=float(np.std(pass_at_1_values)) if pass_at_1_values else 0.0,
        std_pass_at_k=float(np.std(pass_at_k_values)) if pass_at_k_values else 0.0,
        std_pass_k_power=float(np.std(pass_k_power_values)) if pass_k_power_values else 0.0,
        per_test_case=per_test_case,
        n_test_cases=n_test_cases,
        total_samples=total_samples,
        total_passes=total_passes,
    )


def compute_pass_at_k_from_scoring_results(
    results: Sequence[ScoringResult],
    k: int = 5,
    threshold: float = 0.8,
    metric: str = "hierarchy_f1",
    group_by: str = "test_case_id",
    n_bootstrap: int = 1000,
) -> AggregatedPassAtK:
    """
    Compute pass@k from ScoringResult objects.

    Args:
        results: Sequence of ScoringResult objects
        k: Target k value
        threshold: Score threshold for passing
        metric: Metric to use for pass/fail determination
        group_by: Field to group results by ('test_case_id', 'model', 'condition')
        n_bootstrap: Bootstrap iterations

    Returns:
        AggregatedPassAtK with per-group results
    """
    # Group results
    grouped: dict[str, list[bool]] = defaultdict(list)

    for result in results:
        if not result.evaluation:
            continue

        key = getattr(result, group_by, "unknown")
        score = getattr(result.evaluation, metric, 0.0)
        grouped[key].append(score >= threshold)

    return aggregate_pass_at_k(dict(grouped), k=k, n_bootstrap=n_bootstrap)


@dataclass
class ModelComparisonResult:
    """Result of comparing pass@k between models."""
    model_a: str
    model_b: str
    metric: str
    k: int

    # Pass@k values
    pass_at_k_a: float
    pass_at_k_b: float
    difference: float

    # Statistical significance
    p_value: float | None
    is_significant: bool
    confidence_level: float

    # Sample sizes
    n_samples_a: int
    n_samples_b: int


def compare_models_pass_at_k(
    results_a: Sequence[bool],
    results_b: Sequence[bool],
    k: int = 5,
    n_permutations: int = 10000,
    confidence_level: float = 0.95,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> ModelComparisonResult:
    """
    Compare pass@k between two models using permutation test.

    Args:
        results_a: Pass/fail results for model A
        results_b: Pass/fail results for model B
        k: Target k value
        n_permutations: Number of permutations for significance test
        confidence_level: Confidence level for significance
        model_a_name: Name of model A
        model_b_name: Name of model B

    Returns:
        ModelComparisonResult with comparison statistics
    """
    pass_at_k_a = compute_pass_at_k(results_a, k)
    pass_at_k_b = compute_pass_at_k(results_b, k)
    observed_diff = pass_at_k_a - pass_at_k_b

    # Permutation test
    combined = list(results_a) + list(results_b)
    n_a = len(results_a)
    n_b = len(results_b)

    null_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = compute_pass_at_k(perm_a, k) - compute_pass_at_k(perm_b, k)
        null_diffs.append(perm_diff)

    # Two-tailed p-value
    p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))
    alpha = 1 - confidence_level
    is_significant = p_value < alpha

    return ModelComparisonResult(
        model_a=model_a_name,
        model_b=model_b_name,
        metric="pass@k",
        k=k,
        pass_at_k_a=pass_at_k_a,
        pass_at_k_b=pass_at_k_b,
        difference=observed_diff,
        p_value=p_value,
        is_significant=is_significant,
        confidence_level=confidence_level,
        n_samples_a=n_a,
        n_samples_b=n_b,
    )


def format_pass_at_k_report(
    aggregated: AggregatedPassAtK,
    title: str = "Pass@k Analysis",
) -> str:
    """
    Format a human-readable pass@k report.

    Args:
        aggregated: AggregatedPassAtK result
        title: Report title

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        title.center(60),
        "=" * 60,
        "",
        f"Summary (k={aggregated.k})",
        "-" * 40,
        f"  Test cases:     {aggregated.n_test_cases}",
        f"  Total samples:  {aggregated.total_samples}",
        f"  Total passes:   {aggregated.total_passes}",
        f"  Overall rate:   {aggregated.total_passes / aggregated.total_samples:.1%}" if aggregated.total_samples > 0 else "  Overall rate:   N/A",
        "",
        "Aggregate Metrics",
        "-" * 40,
        f"  pass@1:    {aggregated.mean_pass_at_1:.3f} (+/- {aggregated.std_pass_at_1:.3f})",
        f"  pass@{aggregated.k}:    {aggregated.mean_pass_at_k:.3f} (+/- {aggregated.std_pass_at_k:.3f})",
        f"  pass^{aggregated.k}:    {aggregated.mean_pass_k_power:.3f} (+/- {aggregated.std_pass_k_power:.3f})",
        "",
    ]

    # Per-test-case breakdown (top 10 by variance)
    if aggregated.per_test_case:
        sorted_cases = sorted(
            aggregated.per_test_case.items(),
            key=lambda x: x[1].pass_rate,
        )

        lines.append("Per-Test-Case Results (sorted by pass rate)")
        lines.append("-" * 40)

        for tc_id, result in sorted_cases[:10]:
            lines.append(
                f"  {tc_id[:30]:<30} "
                f"pass@1={result.pass_at_1:.2f} "
                f"pass@{aggregated.k}={result.pass_at_k:.2f} "
                f"({result.n_passes}/{result.n_samples})"
            )

        if len(sorted_cases) > 10:
            lines.append(f"  ... and {len(sorted_cases) - 10} more test cases")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
