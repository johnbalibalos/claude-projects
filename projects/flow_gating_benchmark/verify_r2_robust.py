#!/usr/bin/env python3
"""
Robust verification of the low R² finding for token frequency correlation.

This script performs multiple statistical tests to verify that the low correlation
between PubMed frequency and model detection rate is real and robust.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path


def extract_gates_from_hierarchy(node: dict) -> set[str]:
    """Recursively extract all gate names from a hierarchy."""
    gates = set()
    if node.get("name"):
        gates.add(node["name"].strip())
    for child in node.get("children", []):
        gates.update(extract_gates_from_hierarchy(child))
    return gates


def load_scoring_results_with_gate_matching(scoring_path: Path) -> dict[str, dict]:
    """Extract per-gate match/miss stats from scoring results."""
    with open(scoring_path) as f:
        data = json.load(f)

    pop_stats = defaultdict(lambda: {"matches": 0, "misses": 0})

    for result in data.get("results", []):
        parsed = result.get("parsed_hierarchy", {})
        predicted_gates = extract_gates_from_hierarchy(parsed) if parsed else set()
        gt_gates = set(result.get("ground_truth_gates", []))

        predicted_normalized = {g.lower().strip(): g for g in predicted_gates}
        gt_normalized = {g.lower().strip(): g for g in gt_gates}

        for gt_norm, gt_orig in gt_normalized.items():
            if gt_norm in predicted_normalized:
                pop_stats[gt_orig]["matches"] += 1
            else:
                pop_stats[gt_orig]["misses"] += 1

    return dict(pop_stats)


def pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    ss_x = sum((xi - mean_x) ** 2 for xi in x)
    ss_y = sum((yi - mean_y) ** 2 for yi in y)

    if ss_x == 0 or ss_y == 0:
        return 0.0

    return numerator / (math.sqrt(ss_x) * math.sqrt(ss_y))


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    def rank(values):
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = rank_val + 1
        return ranks

    return pearson_correlation(rank(x), rank(y))


def bootstrap_r2(x: list[float], y: list[float], n_bootstrap: int = 1000) -> tuple[float, float, float]:
    """
    Compute R² with 95% bootstrap confidence interval.
    Returns (r2, ci_low, ci_high).
    """
    n = len(x)
    r2_values = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = [random.randint(0, n - 1) for _ in range(n)]
        x_boot = [x[i] for i in indices]
        y_boot = [y[i] for i in indices]

        r = pearson_correlation(x_boot, y_boot)
        r2_values.append(r ** 2)

    r2_values.sort()
    r2_original = pearson_correlation(x, y) ** 2

    ci_low = r2_values[int(0.025 * n_bootstrap)]
    ci_high = r2_values[int(0.975 * n_bootstrap)]

    return r2_original, ci_low, ci_high


def permutation_test(x: list[float], y: list[float], n_permutations: int = 10000) -> float:
    """
    Permutation test for significance of correlation.
    Returns p-value.
    """
    observed_r = abs(pearson_correlation(x, y))

    more_extreme = 0
    for _ in range(n_permutations):
        y_shuffled = y.copy()
        random.shuffle(y_shuffled)
        perm_r = abs(pearson_correlation(x, y_shuffled))
        if perm_r >= observed_r:
            more_extreme += 1

    return more_extreme / n_permutations


def main():
    random.seed(42)  # For reproducibility

    project_dir = Path(__file__).parent
    results_dir = project_dir / "results" / "full_benchmark_20260114"

    # Load PubMed frequencies
    pubmed_path = project_dir / "data" / "cache" / "pubmed_frequencies.json"
    with open(pubmed_path) as f:
        pubmed_freqs = json.load(f)

    print(f"Loaded {len(pubmed_freqs)} PubMed frequencies")
    print(f"Range: {min(pubmed_freqs.values())} - {max(pubmed_freqs.values())}")

    # Load scoring results
    scoring_path = results_dir / "scoring_results.json"
    pop_stats = load_scoring_results_with_gate_matching(scoring_path)
    print(f"Extracted stats for {len(pop_stats)} populations")

    # Match populations
    pubmed_normalized = {k.lower().strip(): (k, v) for k, v in pubmed_freqs.items()}

    matched_data = []
    for pop_name, stats in pop_stats.items():
        total = stats["matches"] + stats["misses"]
        if total == 0:
            continue

        detection_rate = stats["matches"] / total
        normalized = pop_name.lower().strip()

        if normalized in pubmed_normalized:
            orig_name, freq = pubmed_normalized[normalized]
            matched_data.append({
                "name": pop_name,
                "pubmed_count": freq,
                "detection_rate": detection_rate,
            })
        else:
            # Partial matching
            for norm_key, (orig, freq) in pubmed_normalized.items():
                if norm_key in normalized or normalized in norm_key:
                    matched_data.append({
                        "name": pop_name,
                        "pubmed_count": freq,
                        "detection_rate": detection_rate,
                    })
                    break

    print(f"\nMatched {len(matched_data)} populations with PubMed data")

    # Prepare data
    x_log = [math.log10(max(d["pubmed_count"], 1)) for d in matched_data]
    x_raw = [d["pubmed_count"] for d in matched_data]
    y_det = [d["detection_rate"] for d in matched_data]

    print("\n" + "=" * 80)
    print("ROBUST R² VERIFICATION")
    print("=" * 80)

    # 1. Basic correlations
    print("\n## 1. BASIC CORRELATIONS")
    print("-" * 60)

    r_log = pearson_correlation(x_log, y_det)
    r_raw = pearson_correlation(x_raw, y_det)
    r_spearman = spearman_correlation(x_raw, y_det)

    print(f"Pearson r (log-frequency):     {r_log:>8.4f}  →  R² = {r_log**2:.4f}")
    print(f"Pearson r (raw frequency):     {r_raw:>8.4f}  →  R² = {r_raw**2:.4f}")
    print(f"Spearman rho (rank-based):     {r_spearman:>8.4f}")

    # 2. Bootstrap confidence intervals
    print("\n## 2. BOOTSTRAP CONFIDENCE INTERVALS (n=1000)")
    print("-" * 60)

    r2_log, ci_low_log, ci_high_log = bootstrap_r2(x_log, y_det)
    r2_raw, ci_low_raw, ci_high_raw = bootstrap_r2(x_raw, y_det)

    print(f"Log-frequency R²:  {r2_log:.4f}  (95% CI: [{ci_low_log:.4f}, {ci_high_log:.4f}])")
    print(f"Raw frequency R²:  {r2_raw:.4f}  (95% CI: [{ci_low_raw:.4f}, {ci_high_raw:.4f}])")

    # 3. Permutation test
    print("\n## 3. PERMUTATION TEST (n=10000)")
    print("-" * 60)

    p_log = permutation_test(x_log, y_det)
    p_raw = permutation_test(x_raw, y_det)

    print(f"Log-frequency:  p = {p_log:.4f}")
    print(f"Raw frequency:  p = {p_raw:.4f}")

    if p_log > 0.05:
        print("\n→ Correlation is NOT statistically significant (p > 0.05)")
        print("  The observed correlation could be due to chance alone.")
    else:
        print(f"\n→ Correlation is statistically significant (p = {p_log:.4f})")

    # 4. Effect size interpretation
    print("\n## 4. EFFECT SIZE INTERPRETATION")
    print("-" * 60)

    print("""
    Cohen's guidelines for R²:
    - R² < 0.01:  Negligible effect
    - R² < 0.09:  Small effect
    - R² < 0.25:  Medium effect
    - R² ≥ 0.25:  Large effect

    Our observed R² = {:.4f}

    """.format(r2_log))

    if r2_log < 0.01:
        print("→ NEGLIGIBLE effect: Frequency explains virtually none of the variance")
    elif r2_log < 0.09:
        print("→ SMALL effect: Frequency explains very little variance")
    elif r2_log < 0.25:
        print("→ MEDIUM effect: Frequency explains some variance")
    else:
        print("→ LARGE effect: Frequency explains substantial variance")

    # 5. Sensitivity analysis: try different frequency transformations
    print("\n## 5. SENSITIVITY ANALYSIS: Different Transformations")
    print("-" * 60)

    # Try sqrt transformation
    x_sqrt = [math.sqrt(d["pubmed_count"]) for d in matched_data]
    r_sqrt = pearson_correlation(x_sqrt, y_det)

    # Try quintile binning
    sorted_by_freq = sorted(enumerate(x_raw), key=lambda x: x[1])
    quintile_size = len(sorted_by_freq) // 5
    x_quintile = [0] * len(x_raw)
    for i, (orig_idx, _) in enumerate(sorted_by_freq):
        x_quintile[orig_idx] = min(i // max(quintile_size, 1), 4)
    r_quintile = pearson_correlation(x_quintile, y_det)

    print(f"Log transform:      R² = {r_log**2:.4f}")
    print(f"Sqrt transform:     R² = {r_sqrt**2:.4f}")
    print(f"Quintile binning:   R² = {r_quintile**2:.4f}")
    print(f"No transform:       R² = {r_raw**2:.4f}")

    # 6. Subgroup analysis
    print("\n## 6. SUBGROUP ANALYSIS")
    print("-" * 60)

    # High vs low frequency subgroups
    median_freq = sorted(x_raw)[len(x_raw) // 2]

    high_freq_data = [(x, y) for x, y in zip(x_log, y_det) if 10**x >= median_freq]
    low_freq_data = [(x, y) for x, y in zip(x_log, y_det) if 10**x < median_freq]

    if len(high_freq_data) > 5 and len(low_freq_data) > 5:
        x_high, y_high = zip(*high_freq_data)
        x_low, y_low = zip(*low_freq_data)

        r_high = pearson_correlation(list(x_high), list(y_high))
        r_low = pearson_correlation(list(x_low), list(y_low))

        print(f"High-frequency subset (n={len(high_freq_data)}):  R² = {r_high**2:.4f}")
        print(f"Low-frequency subset (n={len(low_freq_data)}):   R² = {r_low**2:.4f}")
    else:
        print("Insufficient data for subgroup analysis")

    # 7. Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    all_r2 = [r_log**2, r_raw**2, r_sqrt**2, r_quintile**2]
    max_r2 = max(all_r2)

    print(f"""
    Sample size:              n = {len(matched_data)}
    Maximum R² observed:      {max_r2:.4f}
    Bootstrap 95% CI upper:   {ci_high_log:.4f}
    Permutation p-value:      {p_log:.4f}
    """)

    if max_r2 < 0.05 and ci_high_log < 0.10:
        print("✓ THE LOW R² IS CONFIRMED AND ROBUST")
        print("")
        print("  Even using different transformations and bootstrapping,")
        print("  frequency explains < 5% of variance in model performance.")
        print("  The confidence interval confirms this finding is reliable.")
        print("")
        print("  CONCLUSION: Token frequency does NOT explain model performance.")
    elif max_r2 < 0.10:
        print("~ The R² is low but with some uncertainty")
        print("  Further investigation with more data may be warranted.")
    else:
        print("✗ The analysis shows frequency may have some effect")
        print("  R² suggests frequency could explain some variance.")


if __name__ == "__main__":
    main()
