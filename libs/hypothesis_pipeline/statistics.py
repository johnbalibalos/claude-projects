"""
Statistical rigor module for LLM evaluation.

Provides:
- Bootstrap confidence intervals
- Permutation tests for significance
- Effect size calculations (Cohen's d, etc.)
- Multiple comparison correction (Bonferroni, Holm, FDR)
- Power analysis utilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""

    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    standard_error: float
    bootstrap_distribution: np.ndarray = field(repr=False)

    def __str__(self) -> str:
        return (
            f"{self.estimate:.4f} "
            f"({self.ci_level:.0%} CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )

    def contains(self, value: float) -> bool:
        """Check if a value falls within the confidence interval."""
        return self.ci_lower <= value <= self.ci_upper


def bootstrap_ci(
    scores: ArrayLike,
    statistic: str | callable = "mean",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    method: Literal["percentile", "bca", "basic"] = "percentile",
    random_state: int | np.random.Generator | None = None,
) -> BootstrapResult:
    """
    Compute bootstrap confidence intervals.

    Args:
        scores: Array of scores/measurements
        statistic: Statistic to compute ("mean", "median", "std") or callable
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        method: CI method - "percentile", "bca" (bias-corrected accelerated), or "basic"
        random_state: Random seed or generator for reproducibility

    Returns:
        BootstrapResult with estimate, CI bounds, and distribution

    Example:
        >>> scores = [0.85, 0.78, 0.92, 0.88, 0.79]
        >>> result = bootstrap_ci(scores, statistic="mean", ci_level=0.95)
        >>> print(result)
        0.8440 (95% CI: [0.7920, 0.8920])
    """
    scores = np.asarray(scores)
    n = len(scores)

    if n < 2:
        raise ValueError("Need at least 2 samples for bootstrap")

    # Setup random generator
    rng = np.random.default_rng(random_state)

    # Define statistic function
    if isinstance(statistic, str):
        stat_funcs = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "var": np.var,
            "min": np.min,
            "max": np.max,
        }
        if statistic not in stat_funcs:
            raise ValueError(f"Unknown statistic: {statistic}. Use one of {list(stat_funcs.keys())}")
        stat_fn = stat_funcs[statistic]
    else:
        stat_fn = statistic

    # Compute original estimate
    original_estimate = float(stat_fn(scores))

    # Generate bootstrap samples
    bootstrap_indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_samples = scores[bootstrap_indices]
    bootstrap_stats = np.array([stat_fn(sample) for sample in bootstrap_samples])

    # Compute CI based on method
    alpha = 1 - ci_level

    if method == "percentile":
        ci_lower = float(np.percentile(bootstrap_stats, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrap_stats, (1 - alpha / 2) * 100))

    elif method == "basic":
        # Basic bootstrap: 2*theta - percentiles
        lower_pct = float(np.percentile(bootstrap_stats, (1 - alpha / 2) * 100))
        upper_pct = float(np.percentile(bootstrap_stats, alpha / 2 * 100))
        ci_lower = 2 * original_estimate - lower_pct
        ci_upper = 2 * original_estimate - upper_pct

    elif method == "bca":
        # Bias-corrected and accelerated bootstrap
        # Compute bias correction factor
        z0 = _norm_ppf(np.mean(bootstrap_stats < original_estimate))

        # Compute acceleration factor using jackknife
        jackknife_stats = np.array([
            stat_fn(np.delete(scores, i)) for i in range(n)
        ])
        jack_mean = np.mean(jackknife_stats)
        numerator = np.sum((jack_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = numerator / denominator if denominator != 0 else 0

        # Compute adjusted percentiles
        z_alpha_lower = _norm_ppf(alpha / 2)
        z_alpha_upper = _norm_ppf(1 - alpha / 2)

        alpha_lower = _norm_cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        alpha_upper = _norm_cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        ci_lower = float(np.percentile(bootstrap_stats, alpha_lower * 100))
        ci_upper = float(np.percentile(bootstrap_stats, alpha_upper * 100))

    else:
        raise ValueError(f"Unknown method: {method}")

    return BootstrapResult(
        estimate=original_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        standard_error=float(np.std(bootstrap_stats)),
        bootstrap_distribution=bootstrap_stats,
    )


def bootstrap_compare(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    statistic: str | callable = "mean",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for difference between two groups.

    Args:
        scores_a: Scores from condition A
        scores_b: Scores from condition B
        statistic: Statistic to compare
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        random_state: Random seed

    Returns:
        BootstrapResult for the difference (A - B)
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    rng = np.random.default_rng(random_state)

    if isinstance(statistic, str):
        stat_fn = {"mean": np.mean, "median": np.median}[statistic]
    else:
        stat_fn = statistic

    # Original difference
    original_diff = float(stat_fn(scores_a) - stat_fn(scores_b))

    # Bootstrap differences
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(scores_a, size=len(scores_a), replace=True)
        sample_b = rng.choice(scores_b, size=len(scores_b), replace=True)
        diffs.append(stat_fn(sample_a) - stat_fn(sample_b))

    diffs = np.array(diffs)
    alpha = 1 - ci_level

    return BootstrapResult(
        estimate=original_diff,
        ci_lower=float(np.percentile(diffs, alpha / 2 * 100)),
        ci_upper=float(np.percentile(diffs, (1 - alpha / 2) * 100)),
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        standard_error=float(np.std(diffs)),
        bootstrap_distribution=diffs,
    )


# =============================================================================
# PERMUTATION TESTS
# =============================================================================


@dataclass
class PermutationTestResult:
    """Result of a permutation test."""

    observed_statistic: float
    p_value: float
    n_permutations: int
    alternative: str
    effect_direction: str  # "positive", "negative", or "none"
    null_distribution: np.ndarray = field(repr=False)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < alpha


def paired_permutation_test(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    n_permutations: int = 10000,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_state: int | np.random.Generator | None = None,
) -> PermutationTestResult:
    """
    Paired permutation test for comparing two conditions.

    Tests whether the difference between paired observations is significant
    by randomly flipping the sign of differences.

    Args:
        scores_a: Scores from condition A (same samples)
        scores_b: Scores from condition B (same samples)
        n_permutations: Number of permutations
        alternative: "two-sided", "greater" (A > B), or "less" (A < B)
        random_state: Random seed

    Returns:
        PermutationTestResult with p-value and null distribution

    Example:
        >>> # Test if CoT prompting improves over direct prompting
        >>> direct_scores = [0.75, 0.80, 0.72, 0.78, 0.85]
        >>> cot_scores = [0.82, 0.85, 0.79, 0.84, 0.88]
        >>> result = paired_permutation_test(cot_scores, direct_scores, alternative="greater")
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Paired test requires equal-length arrays")

    rng = np.random.default_rng(random_state)
    n = len(scores_a)

    # Compute differences
    differences = scores_a - scores_b
    observed_diff = float(np.mean(differences))

    # Generate permutation distribution
    null_diffs = []
    for _ in range(n_permutations):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=n)
        permuted_diff = np.mean(signs * differences)
        null_diffs.append(permuted_diff)

    null_diffs = np.array(null_diffs)

    # Compute p-value based on alternative
    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))
    elif alternative == "greater":
        p_value = float(np.mean(null_diffs >= observed_diff))
    elif alternative == "less":
        p_value = float(np.mean(null_diffs <= observed_diff))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Determine effect direction
    if observed_diff > 0:
        effect_direction = "positive"
    elif observed_diff < 0:
        effect_direction = "negative"
    else:
        effect_direction = "none"

    return PermutationTestResult(
        observed_statistic=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        alternative=alternative,
        effect_direction=effect_direction,
        null_distribution=null_diffs,
    )


def independent_permutation_test(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    n_permutations: int = 10000,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_state: int | np.random.Generator | None = None,
) -> PermutationTestResult:
    """
    Independent (unpaired) permutation test for comparing two groups.

    Tests whether two independent groups have different means by
    randomly shuffling group assignments.

    Args:
        scores_a: Scores from group A
        scores_b: Scores from group B
        n_permutations: Number of permutations
        alternative: "two-sided", "greater" (A > B), or "less" (A < B)
        random_state: Random seed

    Returns:
        PermutationTestResult with p-value and null distribution
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    rng = np.random.default_rng(random_state)

    n_a, n_b = len(scores_a), len(scores_b)
    combined = np.concatenate([scores_a, scores_b])

    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))

    # Generate permutation distribution
    null_diffs = []
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        null_diffs.append(np.mean(perm_a) - np.mean(perm_b))

    null_diffs = np.array(null_diffs)

    # Compute p-value
    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))
    elif alternative == "greater":
        p_value = float(np.mean(null_diffs >= observed_diff))
    elif alternative == "less":
        p_value = float(np.mean(null_diffs <= observed_diff))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    effect_direction = "positive" if observed_diff > 0 else ("negative" if observed_diff < 0 else "none")

    return PermutationTestResult(
        observed_statistic=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        alternative=alternative,
        effect_direction=effect_direction,
        null_distribution=null_diffs,
    )


# =============================================================================
# EFFECT SIZES
# =============================================================================


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""

    value: float
    effect_type: str
    interpretation: str
    ci_lower: float | None = None
    ci_upper: float | None = None

    def __str__(self) -> str:
        ci_str = ""
        if self.ci_lower is not None and self.ci_upper is not None:
            ci_str = f" (95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}])"
        return f"{self.effect_type} = {self.value:.3f}{ci_str} ({self.interpretation})"


def cohens_d(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    pooled: bool = True,
    compute_ci: bool = True,
    ci_level: float = 0.95,
) -> EffectSizeResult:
    """
    Compute Cohen's d effect size.

    Args:
        scores_a: Scores from condition A
        scores_b: Scores from condition B
        pooled: If True, use pooled standard deviation (default)
        compute_ci: Whether to compute confidence interval
        ci_level: Confidence level for CI

    Returns:
        EffectSizeResult with d value and interpretation

    Interpretation (conventional):
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    n_a, n_b = len(scores_a), len(scores_b)

    if pooled:
        # Pooled standard deviation
        var_a = np.var(scores_a, ddof=1)
        var_b = np.var(scores_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        d = float((mean_a - mean_b) / pooled_std) if pooled_std > 0 else 0.0
    else:
        # Use std of control group (scores_b)
        std_b = np.std(scores_b, ddof=1)
        d = float((mean_a - mean_b) / std_b) if std_b > 0 else 0.0

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    # Compute CI using noncentral t-distribution approximation
    ci_lower, ci_upper = None, None
    if compute_ci:
        # Standard error of d
        se_d = np.sqrt((n_a + n_b) / (n_a * n_b) + d**2 / (2 * (n_a + n_b)))
        z = _norm_ppf((1 + ci_level) / 2)
        ci_lower = d - z * se_d
        ci_upper = d + z * se_d

    return EffectSizeResult(
        value=d,
        effect_type="Cohen's d",
        interpretation=interpretation,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def hedges_g(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    compute_ci: bool = True,
    ci_level: float = 0.95,
) -> EffectSizeResult:
    """
    Compute Hedges' g effect size (bias-corrected Cohen's d).

    Better for small sample sizes than Cohen's d.
    """
    result = cohens_d(scores_a, scores_b, pooled=True, compute_ci=False)

    n_a, n_b = len(scores_a), len(scores_b)
    df = n_a + n_b - 2

    # Bias correction factor
    j = 1 - (3 / (4 * df - 1))
    g = result.value * j

    # Reinterpret
    abs_g = abs(g)
    if abs_g < 0.2:
        interpretation = "negligible"
    elif abs_g < 0.5:
        interpretation = "small"
    elif abs_g < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    ci_lower, ci_upper = None, None
    if compute_ci:
        se_g = np.sqrt((n_a + n_b) / (n_a * n_b) + g**2 / (2 * (n_a + n_b))) * j
        z = _norm_ppf((1 + ci_level) / 2)
        ci_lower = g - z * se_g
        ci_upper = g + z * se_g

    return EffectSizeResult(
        value=g,
        effect_type="Hedges' g",
        interpretation=interpretation,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def glass_delta(
    treatment_scores: ArrayLike,
    control_scores: ArrayLike,
) -> EffectSizeResult:
    """
    Compute Glass's delta effect size.

    Uses control group standard deviation only.
    Useful when variances are unequal.
    """
    treatment = np.asarray(treatment_scores)
    control = np.asarray(control_scores)

    mean_diff = np.mean(treatment) - np.mean(control)
    control_std = np.std(control, ddof=1)

    delta = float(mean_diff / control_std) if control_std > 0 else 0.0

    abs_delta = abs(delta)
    if abs_delta < 0.2:
        interpretation = "negligible"
    elif abs_delta < 0.5:
        interpretation = "small"
    elif abs_delta < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(
        value=delta,
        effect_type="Glass's Δ",
        interpretation=interpretation,
    )


def cliff_delta(scores_a: ArrayLike, scores_b: ArrayLike) -> EffectSizeResult:
    """
    Compute Cliff's delta (non-parametric effect size).

    Measures how often values in A are greater than values in B.
    Range: [-1, 1] where 0 means no effect.

    Interpretation:
        |d| < 0.147: negligible
        0.147 <= |d| < 0.33: small
        0.33 <= |d| < 0.474: medium
        |d| >= 0.474: large
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    n_a, n_b = len(scores_a), len(scores_b)

    # Count dominance
    greater = 0
    less = 0
    for a in scores_a:
        for b in scores_b:
            if a > b:
                greater += 1
            elif a < b:
                less += 1

    delta = float((greater - less) / (n_a * n_b))

    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(
        value=delta,
        effect_type="Cliff's δ",
        interpretation=interpretation,
    )


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# =============================================================================


@dataclass
class MultipleComparisonResult:
    """Result of multiple comparison correction."""

    original_p_values: list[float]
    corrected_p_values: list[float]
    significant: list[bool]
    method: str
    alpha: float
    n_significant: int

    def summary(self) -> str:
        """Generate summary of significant results."""
        lines = [f"Multiple Comparison Correction ({self.method}, α={self.alpha})"]
        lines.append(f"Total comparisons: {len(self.original_p_values)}")
        lines.append(f"Significant after correction: {self.n_significant}")
        lines.append("")
        lines.append("| # | Original p | Corrected p | Significant |")
        lines.append("|---|------------|-------------|-------------|")
        for i, (orig, corr, sig) in enumerate(
            zip(self.original_p_values, self.corrected_p_values, self.significant)
        ):
            sig_str = "Yes" if sig else "No"
            lines.append(f"| {i+1} | {orig:.4f} | {corr:.4f} | {sig_str} |")
        return "\n".join(lines)


def bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """
    Apply Bonferroni correction for multiple comparisons.

    Most conservative method - controls family-wise error rate (FWER).

    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level

    Returns:
        MultipleComparisonResult with corrected p-values
    """
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    significant = [p < alpha for p in corrected]

    return MultipleComparisonResult(
        original_p_values=list(p_values),
        corrected_p_values=corrected,
        significant=significant,
        method="Bonferroni",
        alpha=alpha,
        n_significant=sum(significant),
    )


def holm_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """
    Apply Holm-Bonferroni (step-down) correction.

    Less conservative than Bonferroni while still controlling FWER.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level

    Returns:
        MultipleComparisonResult with corrected p-values
    """
    n = len(p_values)
    p_array = np.array(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    # Compute corrected p-values
    corrected = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        corrected[idx] = min(sorted_p[i] * (n - i), 1.0)

    # Ensure monotonicity (corrected p-values should not decrease)
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])

    significant = [p < alpha for p in corrected]

    return MultipleComparisonResult(
        original_p_values=list(p_values),
        corrected_p_values=corrected.tolist(),
        significant=significant,
        method="Holm-Bonferroni",
        alpha=alpha,
        n_significant=sum(significant),
    )


def benjamini_hochberg_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """
    Apply Benjamini-Hochberg FDR correction.

    Controls false discovery rate (FDR) instead of FWER.
    Less conservative, more power, but allows more false positives.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Target FDR level

    Returns:
        MultipleComparisonResult with corrected p-values
    """
    n = len(p_values)
    p_array = np.array(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    # Compute corrected p-values (q-values)
    corrected = np.zeros(n)
    for i in range(n):
        corrected[sorted_indices[i]] = sorted_p[i] * n / (i + 1)

    # Ensure monotonicity (from largest to smallest)
    for i in range(n - 2, -1, -1):
        idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]
        corrected[idx] = min(corrected[idx], corrected[next_idx])

    corrected = np.minimum(corrected, 1.0)
    significant = [p < alpha for p in corrected]

    return MultipleComparisonResult(
        original_p_values=list(p_values),
        corrected_p_values=corrected.tolist(),
        significant=significant,
        method="Benjamini-Hochberg (FDR)",
        alpha=alpha,
        n_significant=sum(significant),
    )


def apply_correction(
    p_values: Sequence[float],
    method: Literal["bonferroni", "holm", "fdr", "none"] = "holm",
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """
    Apply multiple comparison correction using specified method.

    Args:
        p_values: List of p-values
        method: Correction method
        alpha: Significance level

    Returns:
        MultipleComparisonResult
    """
    if method == "bonferroni":
        return bonferroni_correction(p_values, alpha)
    elif method == "holm":
        return holm_correction(p_values, alpha)
    elif method == "fdr":
        return benjamini_hochberg_correction(p_values, alpha)
    elif method == "none":
        significant = [p < alpha for p in p_values]
        return MultipleComparisonResult(
            original_p_values=list(p_values),
            corrected_p_values=list(p_values),
            significant=significant,
            method="None (uncorrected)",
            alpha=alpha,
            n_significant=sum(significant),
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# POWER ANALYSIS
# =============================================================================


@dataclass
class PowerAnalysisResult:
    """Result of power analysis."""

    required_n: int | None
    achieved_power: float | None
    effect_size: float
    alpha: float
    analysis_type: str

    def __str__(self) -> str:
        if self.required_n is not None:
            return (
                f"Power Analysis: n={self.required_n} samples needed "
                f"(effect_size={self.effect_size:.2f}, α={self.alpha}, power=0.80)"
            )
        else:
            return (
                f"Power Analysis: achieved power={self.achieved_power:.2f} "
                f"(effect_size={self.effect_size:.2f}, α={self.alpha})"
            )


def power_analysis_two_sample(
    effect_size: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    n_per_group: int | None = None,
    ratio: float = 1.0,
) -> PowerAnalysisResult:
    """
    Power analysis for two-sample t-test.

    Either compute required sample size (if n_per_group is None)
    or achieved power (if n_per_group is provided).

    Args:
        effect_size: Expected Cohen's d effect size
        alpha: Significance level
        power: Desired power (if computing n)
        n_per_group: Sample size per group (if computing power)
        ratio: Ratio of n2/n1 (default 1.0 for equal groups)

    Returns:
        PowerAnalysisResult with required n or achieved power
    """
    # Use approximation based on normal distribution
    z_alpha = _norm_ppf(1 - alpha / 2)  # Two-tailed
    z_beta = _norm_ppf(power)

    if n_per_group is None:
        # Compute required sample size
        n1 = ((z_alpha + z_beta) ** 2 * (1 + 1/ratio)) / (effect_size ** 2)
        n1 = int(np.ceil(n1))

        return PowerAnalysisResult(
            required_n=n1,
            achieved_power=None,
            effect_size=effect_size,
            alpha=alpha,
            analysis_type="sample_size",
        )
    else:
        # Compute achieved power
        se = np.sqrt((1 + 1/ratio) / n_per_group)
        ncp = effect_size / se  # Non-centrality parameter
        achieved_power = _norm_cdf(ncp - z_alpha)

        return PowerAnalysisResult(
            required_n=None,
            achieved_power=float(achieved_power),
            effect_size=effect_size,
            alpha=alpha,
            analysis_type="power",
        )


def minimum_detectable_effect(
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Compute minimum detectable effect size given sample size.

    Args:
        n_per_group: Sample size per group
        alpha: Significance level
        power: Desired power

    Returns:
        Minimum detectable Cohen's d
    """
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_beta = _norm_ppf(power)

    mde = (z_alpha + z_beta) * np.sqrt(2 / n_per_group)
    return float(mde)


# =============================================================================
# COMPREHENSIVE COMPARISON
# =============================================================================


@dataclass
class ComprehensiveComparison:
    """Complete statistical comparison between two conditions."""

    condition_a: str
    condition_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_difference: float
    bootstrap_ci: BootstrapResult
    permutation_test: PermutationTestResult
    effect_size: EffectSizeResult
    is_significant: bool
    significance_level: float

    def summary(self) -> str:
        """Generate a human-readable summary."""
        sig_str = "SIGNIFICANT" if self.is_significant else "not significant"

        lines = [
            f"Comparison: {self.condition_a} vs {self.condition_b}",
            "=" * 50,
            f"",
            f"Descriptive Statistics:",
            f"  {self.condition_a}: mean={self.mean_a:.4f}, std={self.std_a:.4f}, n={self.n_a}",
            f"  {self.condition_b}: mean={self.mean_b:.4f}, std={self.std_b:.4f}, n={self.n_b}",
            f"  Difference (A - B): {self.mean_difference:.4f}",
            f"",
            f"Bootstrap CI ({self.bootstrap_ci.ci_level:.0%}): "
            f"[{self.bootstrap_ci.ci_lower:.4f}, {self.bootstrap_ci.ci_upper:.4f}]",
            f"",
            f"Permutation Test:",
            f"  p-value: {self.permutation_test.p_value:.4f}",
            f"  Alternative: {self.permutation_test.alternative}",
            f"",
            f"Effect Size:",
            f"  {self.effect_size}",
            f"",
            f"Conclusion: {sig_str} at α={self.significance_level}",
        ]

        return "\n".join(lines)


def compare_conditions(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    condition_a_name: str = "Condition A",
    condition_b_name: str = "Condition B",
    paired: bool = False,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    random_state: int | np.random.Generator | None = None,
) -> ComprehensiveComparison:
    """
    Perform comprehensive statistical comparison between two conditions.

    Includes:
    - Descriptive statistics
    - Bootstrap confidence intervals
    - Permutation test
    - Effect size (Cohen's d or Hedges' g)

    Args:
        scores_a: Scores from condition A
        scores_b: Scores from condition B
        condition_a_name: Name for condition A
        condition_b_name: Name for condition B
        paired: Whether samples are paired
        alpha: Significance level
        alternative: Alternative hypothesis direction
        n_bootstrap: Number of bootstrap samples
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        ComprehensiveComparison with all statistics

    Example:
        >>> direct = [0.72, 0.78, 0.75, 0.80, 0.77]
        >>> cot = [0.85, 0.82, 0.88, 0.84, 0.86]
        >>> result = compare_conditions(cot, direct, "CoT", "Direct", paired=True)
        >>> print(result.summary())
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    # Descriptive statistics
    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = float(np.mean(scores_a)), float(np.mean(scores_b))
    std_a, std_b = float(np.std(scores_a, ddof=1)), float(np.std(scores_b, ddof=1))
    mean_diff = mean_a - mean_b

    # Bootstrap CI for difference
    boot_ci = bootstrap_compare(
        scores_a, scores_b,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # Permutation test
    if paired:
        perm_test = paired_permutation_test(
            scores_a, scores_b,
            n_permutations=n_permutations,
            alternative=alternative,
            random_state=random_state,
        )
    else:
        perm_test = independent_permutation_test(
            scores_a, scores_b,
            n_permutations=n_permutations,
            alternative=alternative,
            random_state=random_state,
        )

    # Effect size (use Hedges' g for small samples)
    if n_a < 20 or n_b < 20:
        effect = hedges_g(scores_a, scores_b)
    else:
        effect = cohens_d(scores_a, scores_b)

    # Determine significance
    is_significant = perm_test.p_value < alpha

    return ComprehensiveComparison(
        condition_a=condition_a_name,
        condition_b=condition_b_name,
        n_a=n_a,
        n_b=n_b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        mean_difference=mean_diff,
        bootstrap_ci=boot_ci,
        permutation_test=perm_test,
        effect_size=effect,
        is_significant=is_significant,
        significance_level=alpha,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _norm_ppf(p: float) -> float:
    """Percent point function (inverse CDF) of standard normal distribution."""
    # Use approximation to avoid scipy dependency
    # Abramowitz and Stegun approximation 26.2.23
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return 0.0

    if p < 0.5:
        sign = -1
        p = 1 - p
    else:
        sign = 1
        p = p

    t = np.sqrt(-2 * np.log(1 - p))

    # Coefficients
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    result = t - (c0 + c1*t + c2*t**2) / (1 + d1*t + d2*t**2 + d3*t**3)
    return float(sign * result)


def _norm_cdf(x: float) -> float:
    """CDF of standard normal distribution."""
    # Use approximation to avoid scipy dependency
    # Abramowitz and Stegun approximation
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741,
        -1.453152027, 1.061405429,
    )
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / np.sqrt(2)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return float(0.5 * (1.0 + sign * y))


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a set of scores."""

    n: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    se: float  # Standard error of mean
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound

    def __str__(self) -> str:
        return (
            f"n={self.n}, mean={self.mean:.4f}±{self.std:.4f}, "
            f"median={self.median:.4f}, range=[{self.min:.4f}, {self.max:.4f}]"
        )


def describe(scores: ArrayLike) -> DescriptiveStats:
    """Compute descriptive statistics for a set of scores."""
    scores = np.asarray(scores)
    n = len(scores)

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0

    # 95% CI using t-distribution approximation (normal for large n)
    z = _norm_ppf(0.975)
    ci_lower = mean - z * se
    ci_upper = mean + z * se

    return DescriptiveStats(
        n=n,
        mean=mean,
        std=std,
        median=float(np.median(scores)),
        min=float(np.min(scores)),
        max=float(np.max(scores)),
        q25=float(np.percentile(scores, 25)),
        q75=float(np.percentile(scores, 75)),
        iqr=float(np.percentile(scores, 75) - np.percentile(scores, 25)),
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
