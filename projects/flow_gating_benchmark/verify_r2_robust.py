#!/usr/bin/env python3
"""
Robust R² Verification for Frequency Confound Analysis.

Validates that the correlation between PubMed frequency and model performance
is statistically robust through:
1. Bootstrap confidence intervals for R²
2. Leave-one-out outlier influence analysis
3. Multiple correlation methods (Pearson, Spearman, robust regression)
4. Permutation test for significance
5. Sensitivity analysis excluding extreme frequencies

Usage:
    python verify_r2_robust.py \
        --frequencies data/cache/pubmed_frequencies.json \
        --results results/experiment_results_*.json \
        --output results/r2_robust_analysis.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Extract Pearson r correlation coefficient with proper typing."""
    result = stats.pearsonr(x, y)
    return float(result[0])  # type: ignore[arg-type]


def _pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Compute R² from Pearson correlation with proper typing."""
    r = _pearson_r(x, y)
    return r * r


@dataclass
class RobustnessMetrics:
    """Container for R² robustness analysis results."""
    # Primary correlation
    pearson_r: float
    pearson_r2: float
    pearson_p: float

    # Spearman (rank-based, robust to outliers)
    spearman_r: float
    spearman_p: float

    # Bootstrap CI for R²
    r2_bootstrap_mean: float
    r2_bootstrap_std: float
    r2_ci_lower: float  # 2.5th percentile
    r2_ci_upper: float  # 97.5th percentile

    # Outlier analysis
    n_outliers_detected: int
    r2_without_outliers: float
    max_influence_population: str
    max_influence_r2_change: float

    # Permutation test
    permutation_p: float

    # Sensitivity analysis
    r2_trimmed_5pct: float  # Excluding 5% extreme frequencies
    r2_trimmed_10pct: float  # Excluding 10% extreme frequencies

    # Sample info
    n_populations: int

    # Additional diagnostics
    cook_d_max: float = 0.0
    leverage_max: float = 0.0


@dataclass
class PopulationDataPoint:
    """Data point for correlation analysis."""
    name: str
    log_frequency: float  # log10(pubmed_count + 1)
    detection_rate: float
    raw_frequency: int
    match_count: int = 0
    miss_count: int = 0


def load_frequencies(freq_path: Path) -> dict[str, int]:
    """Load PubMed frequency data."""
    return json.loads(freq_path.read_text())


def extract_gates_from_hierarchy(node: dict, gates: set[str] | None = None) -> set[str]:
    """Recursively extract all gate names from a hierarchy."""
    if gates is None:
        gates = set()

    if isinstance(node, dict):
        name = node.get("name", "")
        if name and name.lower() not in {"all events", "root", "ungated"}:
            gates.add(name)
        for child in node.get("children", []):
            extract_gates_from_hierarchy(child, gates)

    return gates


def load_ground_truth(gt_dir: Path) -> dict[str, set[str]]:
    """Load ground truth gate names for each test case."""
    ground_truth = {}

    for f in gt_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            test_case_id = data.get("test_case_id", f.stem)

            hierarchy = data.get("gating_hierarchy", {})
            if isinstance(hierarchy, dict):
                root = hierarchy.get("root", hierarchy)
                gates = extract_gates_from_hierarchy(root)
                ground_truth[test_case_id] = gates
        except (json.JSONDecodeError, KeyError):
            continue

    return ground_truth


def normalize_gate_name(name: str) -> str:
    """Normalize gate name for comparison."""
    # Lowercase, remove extra spaces
    name = " ".join(name.lower().split())
    # Remove common suffixes/prefixes
    name = name.replace("+ cells", "+").replace("cells", "").strip()
    return name


def extract_detection_rates(
    results: dict,
    ground_truth: dict[str, set[str]],
) -> dict[str, tuple[int, int]]:
    """
    Extract per-population match/miss counts from scoring results.
    Returns {population_name: (match_count, miss_count)}.

    Works with both:
    - Original format with matching_gates/missing_gates in evaluation
    - New format with parsed_hierarchy compared to ground truth
    """
    counts: dict[str, list[int]] = {}  # name -> [matches, misses]

    for result in results.get("results", []):
        # Try original format first
        eval_data = result.get("evaluation", {})

        if "matching_gates" in eval_data or "missing_gates" in eval_data:
            # Original format
            for gate in eval_data.get("matching_gates", []):
                name = gate.strip()
                if name not in counts:
                    counts[name] = [0, 0]
                counts[name][0] += 1

            for gate in eval_data.get("missing_gates", []):
                name = gate.strip()
                if name not in counts:
                    counts[name] = [0, 0]
                counts[name][1] += 1
        else:
            # New format: compare parsed_hierarchy with ground truth
            test_case_id = result.get("test_case_id", "")
            parsed = result.get("parsed_hierarchy")

            if not parsed or test_case_id not in ground_truth:
                continue

            gt_gates = ground_truth[test_case_id]
            pred_gates = extract_gates_from_hierarchy(parsed)

            # Normalize for comparison
            gt_normalized = {normalize_gate_name(g): g for g in gt_gates}
            pred_normalized = {normalize_gate_name(g): g for g in pred_gates}

            # Match gates (using normalized names)
            for norm_name, orig_name in gt_normalized.items():
                if orig_name not in counts:
                    counts[orig_name] = [0, 0]

                # Check if prediction contains this gate (or similar)
                matched = False
                for pred_norm, _pred_orig in pred_normalized.items():
                    # Simple string containment for fuzzy matching
                    if norm_name == pred_norm or norm_name in pred_norm or pred_norm in norm_name:
                        matched = True
                        break

                if matched:
                    counts[orig_name][0] += 1
                else:
                    counts[orig_name][1] += 1

    return {k: (v[0], v[1]) for k, v in counts.items()}


def prepare_data(
    frequencies: dict[str, int],
    detection_data: dict[str, tuple[int, int]],
    min_observations: int = 3,
) -> list[PopulationDataPoint]:
    """
    Merge frequency and detection data, filter by minimum observations.
    """
    data_points = []

    for name, (matches, misses) in detection_data.items():
        total = matches + misses
        if total < min_observations:
            continue

        # Find matching frequency (case-insensitive)
        freq = frequencies.get(name)
        if freq is None:
            # Try case-insensitive match
            for freq_name, f in frequencies.items():
                if freq_name.lower() == name.lower():
                    freq = f
                    break

        if freq is not None:
            data_points.append(PopulationDataPoint(
                name=name,
                log_frequency=np.log10(freq + 1),
                detection_rate=matches / total,
                raw_frequency=freq,
                match_count=matches,
                miss_count=misses,
            ))

    return data_points


def bootstrap_r2(
    x: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 10000,
    random_seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Compute bootstrap confidence interval for R².

    Returns: (mean_r2, std_r2, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(random_seed)
    n = len(x)
    r2_samples = []

    for _ in range(n_iterations):
        # Sample with replacement
        idx = rng.integers(0, n, size=n)
        x_boot = x[idx]
        y_boot = y[idx]

        # Compute R² (handle edge cases)
        if np.std(x_boot) > 0 and np.std(y_boot) > 0:
            r2_samples.append(_pearson_r2(x_boot, y_boot))

    r2_arr = np.array(r2_samples)

    return (
        float(np.mean(r2_arr)),
        float(np.std(r2_arr)),
        float(np.percentile(r2_arr, 2.5)),
        float(np.percentile(r2_arr, 97.5)),
    )


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 10000,
    random_seed: int = 42,
) -> float:
    """
    Permutation test for correlation significance.

    Returns: p-value (proportion of permutations with |r| >= observed |r|)
    """
    rng = np.random.default_rng(random_seed)
    observed_r = abs(_pearson_r(x, y))

    count_extreme = 0
    for _ in range(n_iterations):
        y_perm = rng.permutation(y)
        perm_r = abs(_pearson_r(x, y_perm))
        if perm_r >= observed_r:
            count_extreme += 1

    return count_extreme / n_iterations


def leave_one_out_influence(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Leave-one-out analysis to identify influential points.

    Returns: (most_influential_name, max_r2_change, all_influences)
    """
    n = len(x)
    base_r2 = _pearson_r2(x, y)

    influences = []
    for i in range(n):
        x_loo = np.delete(x, i)
        y_loo = np.delete(y, i)

        if len(x_loo) > 2:
            loo_r2 = _pearson_r2(x_loo, y_loo)
            r2_change = abs(loo_r2 - base_r2)
            influences.append((names[i], r2_change, loo_r2))

    # Find most influential
    if influences:
        max_inf = max(influences, key=lambda x: x[1])
        return max_inf[0], max_inf[1], [(n, c) for n, c, _ in influences]

    return "", 0.0, []


def detect_outliers(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float = 2.5,  # MAD units
) -> np.ndarray:
    """
    Detect outliers using Median Absolute Deviation (robust to outliers).

    Returns: boolean mask of outliers
    """
    # Fit robust regression (median-based)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # MAD-based outlier detection
    median_resid = np.median(residuals)
    mad = np.median(np.abs(residuals - median_resid))

    if mad == 0:
        # Fall back to IQR method
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (residuals < lower) | (residuals > upper)

    # MAD-based threshold (1.4826 is consistency constant for normal dist)
    mad_units = np.abs(residuals - median_resid) / (1.4826 * mad)
    return mad_units > threshold


def compute_cook_distance(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Compute Cook's distance and leverage for outlier influence.

    Returns: (max_cook_d, max_leverage)
    """
    n = len(x)

    # Simple linear regression
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))

    if ss_xx == 0:
        return 0.0, 0.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    residuals = y - y_pred

    # MSE
    mse = np.sum(residuals ** 2) / (n - 2) if n > 2 else 0

    # Leverage: h_ii = 1/n + (x_i - x_mean)^2 / SS_xx
    leverage = 1/n + (x - x_mean) ** 2 / ss_xx

    # Cook's distance
    if mse > 0:
        cook_d = residuals ** 2 / (2 * mse) * leverage / (1 - leverage) ** 2
    else:
        cook_d = np.zeros_like(residuals)

    return float(np.max(cook_d)), float(np.max(leverage))


def analyze_robustness(
    data_points: list[PopulationDataPoint],
    n_bootstrap: int = 10000,
) -> RobustnessMetrics:
    """
    Perform comprehensive R² robustness analysis.
    """
    # Extract arrays
    x = np.array([dp.log_frequency for dp in data_points])
    y = np.array([dp.detection_rate for dp in data_points])
    names = [dp.name for dp in data_points]
    n = len(x)

    # 1. Primary correlations
    pearson_result = stats.pearsonr(x, y)
    pearson_r_val = float(pearson_result[0])  # type: ignore[arg-type]
    pearson_p_val = float(pearson_result[1])  # type: ignore[arg-type]
    spearman_result = stats.spearmanr(x, y)
    spearman_r_val = float(spearman_result[0])  # type: ignore[arg-type]
    spearman_p_val = float(spearman_result[1])  # type: ignore[arg-type]

    # 2. Bootstrap CI
    r2_mean, r2_std, r2_ci_lo, r2_ci_hi = bootstrap_r2(x, y, n_bootstrap)

    # 3. Leave-one-out influence
    max_inf_name, max_inf_change, _ = leave_one_out_influence(x, y, names)

    # 4. Outlier detection and R² without outliers
    outlier_mask = detect_outliers(x, y)
    n_outliers = int(np.sum(outlier_mask))

    if n_outliers > 0 and n - n_outliers > 2:
        x_clean = x[~outlier_mask]
        y_clean = y[~outlier_mask]
        r2_no_outliers = _pearson_r2(x_clean, y_clean)
    else:
        r2_no_outliers = pearson_r_val * pearson_r_val

    # 5. Permutation test
    perm_p = permutation_test(x, y, n_iterations=n_bootstrap)

    # 6. Sensitivity analysis - trimmed analysis
    # Sort by frequency and trim extremes
    sorted_idx = np.argsort(x)

    # Trim 5% from each end
    trim_5 = int(n * 0.05)
    if trim_5 > 0 and n - 2 * trim_5 > 2:
        idx_5 = sorted_idx[trim_5:-trim_5]
        r2_trim_5 = _pearson_r2(x[idx_5], y[idx_5])
    else:
        r2_trim_5 = pearson_r_val * pearson_r_val

    # Trim 10% from each end
    trim_10 = int(n * 0.10)
    if trim_10 > 0 and n - 2 * trim_10 > 2:
        idx_10 = sorted_idx[trim_10:-trim_10]
        r2_trim_10 = _pearson_r2(x[idx_10], y[idx_10])
    else:
        r2_trim_10 = pearson_r_val * pearson_r_val

    # 7. Cook's distance
    cook_d_max, leverage_max = compute_cook_distance(x, y)

    return RobustnessMetrics(
        pearson_r=pearson_r_val,
        pearson_r2=pearson_r_val * pearson_r_val,
        pearson_p=pearson_p_val,
        spearman_r=spearman_r_val,
        spearman_p=spearman_p_val,
        r2_bootstrap_mean=float(r2_mean),
        r2_bootstrap_std=float(r2_std),
        r2_ci_lower=float(r2_ci_lo),
        r2_ci_upper=float(r2_ci_hi),
        n_outliers_detected=n_outliers,
        r2_without_outliers=float(r2_no_outliers),
        max_influence_population=max_inf_name,
        max_influence_r2_change=float(max_inf_change),
        permutation_p=float(perm_p),
        r2_trimmed_5pct=float(r2_trim_5),
        r2_trimmed_10pct=float(r2_trim_10),
        n_populations=n,
        cook_d_max=float(cook_d_max),
        leverage_max=float(leverage_max),
    )


def generate_plots(
    data_points: list[PopulationDataPoint],
    metrics: RobustnessMetrics,
    output_dir: Path,
):
    """Generate visualization plots."""
    x = np.array([dp.log_frequency for dp in data_points])
    y = np.array([dp.detection_rate for dp in data_points])
    names = [dp.name for dp in data_points]

    # Detect outliers for plotting
    outlier_mask = detect_outliers(x, y)

    # Figure 1: Main scatter with regression
    _fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Main correlation plot
    ax = axes[0, 0]
    ax.scatter(x[~outlier_mask], y[~outlier_mask], c='blue', alpha=0.6, s=50, label='Normal')
    ax.scatter(x[outlier_mask], y[outlier_mask], c='red', alpha=0.8, s=50, label='Outlier')

    # Regression line with CI
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(min(x), max(x), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k-', linewidth=2, label='Regression')

    ax.set_xlabel('Log₁₀(PubMed Frequency + 1)', fontsize=11)
    ax.set_ylabel('Detection Rate', fontsize=11)
    ax.set_title(
        f'Frequency vs Performance\n'
        f'R² = {metrics.pearson_r2:.4f} [95% CI: {metrics.r2_ci_lower:.4f}, {metrics.r2_ci_upper:.4f}]',
        fontsize=11
    )
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel B: Bootstrap R² distribution
    ax = axes[0, 1]
    # Re-run bootstrap for histogram
    rng = np.random.default_rng(42)
    r2_samples = []
    for _ in range(10000):
        idx = rng.integers(0, len(x), size=len(x))
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            r2_samples.append(_pearson_r2(x[idx], y[idx]))

    ax.hist(r2_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(metrics.pearson_r2, color='red', linestyle='--', linewidth=2, label=f'Observed R² = {metrics.pearson_r2:.4f}')
    ax.axvline(metrics.r2_ci_lower, color='gray', linestyle=':', linewidth=1.5, label=f'95% CI: [{metrics.r2_ci_lower:.4f}, {metrics.r2_ci_upper:.4f}]')
    ax.axvline(metrics.r2_ci_upper, color='gray', linestyle=':', linewidth=1.5)
    ax.set_xlabel('R²', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Bootstrap Distribution of R² (n={len(r2_samples):,} iterations)', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel C: Leave-one-out influence
    ax = axes[1, 0]
    _, _, influences = leave_one_out_influence(x, y, names)
    influences_sorted = sorted(influences, key=lambda t: t[1], reverse=True)[:20]

    inf_names = [i[0][:20] for i in influences_sorted]
    inf_values = [i[1] for i in influences_sorted]

    bars = ax.barh(range(len(inf_names)), inf_values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(inf_names)))
    ax.set_yticklabels(inf_names, fontsize=8)
    ax.set_xlabel('|ΔR²| when excluded', fontsize=11)
    ax.set_title(f'Top 20 Most Influential Populations\n(Max influence: {metrics.max_influence_r2_change:.4f})', fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Panel D: Robustness summary
    ax = axes[1, 1]
    methods = ['Full R²', 'Without\nOutliers', 'Trimmed\n5%', 'Trimmed\n10%', 'Bootstrap\nMean']
    values = [
        metrics.pearson_r2,
        metrics.r2_without_outliers,
        metrics.r2_trimmed_5pct,
        metrics.r2_trimmed_10pct,
        metrics.r2_bootstrap_mean,
    ]

    colors = ['steelblue' if v < 0.1 else 'orange' if v < 0.25 else 'red' for v in values]
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')

    ax.axhline(0.25, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold (0.25)')
    ax.axhline(0.50, color='red', linestyle='--', alpha=0.7, label='Strong threshold (0.50)')

    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('R² Robustness Across Methods\n(All < 0.1 = frequency effect ruled out)', fontsize=11)
    ax.set_ylim(0, max(0.3, max(values) * 1.2))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'r2_robustness_analysis.png', dpi=150)
    plt.close()

    print(f"Saved plot to {output_dir / 'r2_robustness_analysis.png'}")


def print_report(metrics: RobustnessMetrics):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("R² ROBUSTNESS VERIFICATION REPORT")
    print("=" * 70)

    print(f"\nSample: n = {metrics.n_populations} populations with sufficient observations")

    print("\n" + "-" * 70)
    print("PRIMARY CORRELATIONS")
    print("-" * 70)
    print(f"  Pearson r:     {metrics.pearson_r:>8.4f}  (p = {metrics.pearson_p:.4e})")
    print(f"  Pearson R²:    {metrics.pearson_r2:>8.4f}")
    print(f"  Spearman ρ:    {metrics.spearman_r:>8.4f}  (p = {metrics.spearman_p:.4e})")

    print("\n" + "-" * 70)
    print("BOOTSTRAP ANALYSIS (10,000 iterations)")
    print("-" * 70)
    print(f"  Bootstrap R² mean:  {metrics.r2_bootstrap_mean:>8.4f}")
    print(f"  Bootstrap R² std:   {metrics.r2_bootstrap_std:>8.4f}")
    print(f"  95% CI:             [{metrics.r2_ci_lower:.4f}, {metrics.r2_ci_upper:.4f}]")

    print("\n" + "-" * 70)
    print("PERMUTATION TEST")
    print("-" * 70)
    print(f"  p-value:  {metrics.permutation_p:.4f}")
    if metrics.permutation_p > 0.05:
        print("  → Correlation NOT significant at α = 0.05")
    else:
        print("  → Correlation significant at α = 0.05")

    print("\n" + "-" * 70)
    print("OUTLIER ANALYSIS")
    print("-" * 70)
    print(f"  Outliers detected:  {metrics.n_outliers_detected}")
    print(f"  R² without outliers: {metrics.r2_without_outliers:.4f}")
    print(f"  Most influential:    {metrics.max_influence_population}")
    print(f"  Max influence (ΔR²): {metrics.max_influence_r2_change:.4f}")
    print(f"  Max Cook's D:        {metrics.cook_d_max:.4f}")
    print(f"  Max Leverage:        {metrics.leverage_max:.4f}")

    print("\n" + "-" * 70)
    print("SENSITIVITY ANALYSIS (trimmed data)")
    print("-" * 70)
    print(f"  R² (trimmed 5%):   {metrics.r2_trimmed_5pct:.4f}")
    print(f"  R² (trimmed 10%):  {metrics.r2_trimmed_10pct:.4f}")

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    r2_values = [
        metrics.pearson_r2,
        metrics.r2_without_outliers,
        metrics.r2_trimmed_5pct,
        metrics.r2_trimmed_10pct,
        metrics.r2_bootstrap_mean,
    ]
    max_r2 = max(r2_values)

    if max_r2 < 0.10:
        verdict = "FREQUENCY EFFECT RULED OUT"
        explanation = (
            "All R² estimates are below 0.10, meaning frequency explains less than 10% "
            "of performance variance under all conditions tested."
        )
    elif max_r2 < 0.25:
        verdict = "WEAK FREQUENCY EFFECT"
        explanation = (
            "R² estimates are between 0.10-0.25. Frequency explains some variance, "
            "but is not the dominant factor."
        )
    elif max_r2 < 0.50:
        verdict = "MODERATE FREQUENCY EFFECT"
        explanation = (
            "R² estimates are between 0.25-0.50. Frequency contributes meaningfully "
            "but reasoning/complexity also plays a role."
        )
    else:
        verdict = "STRONG FREQUENCY EFFECT"
        explanation = (
            "R² estimates exceed 0.50. Frequency is a major driver of performance. "
            "Consider the memorization hypothesis."
        )

    print(f"\n  >>> {verdict} <<<")
    print(f"\n  {explanation}")

    if metrics.r2_ci_upper < 0.15:
        print("\n  The 95% CI upper bound is below 0.15, providing strong statistical")
        print("  confidence that the true R² is small.")

    return {
        "verdict": verdict,
        "max_r2": max_r2,
        "ci_upper": metrics.r2_ci_upper,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Robust R² verification for frequency confound analysis"
    )
    parser.add_argument(
        "--frequencies",
        type=Path,
        default=Path("data/cache/pubmed_frequencies.json"),
        help="Path to PubMed frequencies JSON",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Path to benchmark results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/r2_robust_analysis.json"),
        help="Output path for analysis results",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=3,
        help="Minimum observations per population",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data/verified"),
        help="Ground truth directory (for scoring_results format)",
    )
    args = parser.parse_args()

    # Find results file if not specified
    if args.results is None:
        results_dir = Path(__file__).parent / "results"
        # Try scoring_results first, then experiment_results
        result_files = sorted(results_dir.glob("**/scoring_results.json"), reverse=True)
        if not result_files:
            result_files = sorted(results_dir.glob("experiment_results_*.json"), reverse=True)
        if not result_files:
            result_files = sorted(results_dir.glob("benchmark_results_*.json"), reverse=True)
        if not result_files:
            print("ERROR: No results files found. Specify with --results")
            sys.exit(1)
        args.results = result_files[0]

    print(f"Loading frequencies from: {args.frequencies}")
    print(f"Loading results from: {args.results}")

    # Load data
    frequencies = load_frequencies(args.frequencies)
    results = json.loads(args.results.read_text())

    # Load ground truth for comparing parsed hierarchies
    gt_dir = Path(__file__).parent / args.gt_dir
    ground_truth = load_ground_truth(gt_dir)
    print(f"Loaded ground truth for {len(ground_truth)} test cases")

    # Extract detection rates
    detection_data = extract_detection_rates(results, ground_truth)
    print(f"Found {len(detection_data)} populations in results")

    # Prepare merged data
    data_points = prepare_data(frequencies, detection_data, args.min_obs)
    print(f"Merged data: {len(data_points)} populations with frequency data and >= {args.min_obs} observations")

    if len(data_points) < 10:
        print("ERROR: Insufficient data points for robust analysis")
        sys.exit(1)

    # Run analysis
    print(f"\nRunning robust analysis with {args.n_bootstrap} bootstrap iterations...")
    metrics = analyze_robustness(data_points, args.n_bootstrap)

    # Print report
    summary = print_report(metrics)

    # Generate plots
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_plots(data_points, metrics, output_dir)

    # Save results
    results_dict = {
        "analysis_date": "2026-01-15",
        "frequencies_file": str(args.frequencies),
        "results_file": str(args.results),
        "n_bootstrap": args.n_bootstrap,
        "min_observations": args.min_obs,
        "metrics": {
            "pearson_r": metrics.pearson_r,
            "pearson_r2": metrics.pearson_r2,
            "pearson_p": metrics.pearson_p,
            "spearman_r": metrics.spearman_r,
            "spearman_p": metrics.spearman_p,
            "r2_bootstrap_mean": metrics.r2_bootstrap_mean,
            "r2_bootstrap_std": metrics.r2_bootstrap_std,
            "r2_ci_95_lower": metrics.r2_ci_lower,
            "r2_ci_95_upper": metrics.r2_ci_upper,
            "n_outliers": metrics.n_outliers_detected,
            "r2_without_outliers": metrics.r2_without_outliers,
            "max_influence_population": metrics.max_influence_population,
            "max_influence_r2_change": metrics.max_influence_r2_change,
            "permutation_p": metrics.permutation_p,
            "r2_trimmed_5pct": metrics.r2_trimmed_5pct,
            "r2_trimmed_10pct": metrics.r2_trimmed_10pct,
            "n_populations": metrics.n_populations,
            "cook_d_max": metrics.cook_d_max,
            "leverage_max": metrics.leverage_max,
        },
        "summary": summary,
    }

    args.output.write_text(json.dumps(results_dict, indent=2))
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
