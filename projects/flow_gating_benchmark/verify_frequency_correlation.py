#!/usr/bin/env python3
"""
Verify the frequency confound R² analysis using actual PubMed data.

This script addresses the methodological concern that the original analysis
used estimated frequency levels (1-5) instead of actual PubMed citation counts.
"""

import json
import math
from collections import defaultdict
from pathlib import Path


def load_pubmed_frequencies(cache_path: Path) -> dict[str, int]:
    """Load actual PubMed citation counts."""
    with open(cache_path) as f:
        return json.load(f)


def extract_population_detection_rates(results_path: Path) -> dict[str, dict]:
    """Extract per-population detection rates from benchmark results."""
    with open(results_path) as f:
        data = json.load(f)

    # We need to look at the raw predictions to extract gate matching info
    # The scoring_results.json has per-result metrics but not per-population breakdown
    # Let's check what data is available
    pop_stats = defaultdict(lambda: {"matches": 0, "misses": 0, "total_appearances": 0})

    for result in data.get("results", []):
        # Extract from parsed hierarchy if available
        hierarchy = result.get("parsed_hierarchy", {})
        if hierarchy:
            # Count all predicted gates
            def count_gates(node):
                gates = []
                if node.get("name"):
                    gates.append(node["name"])
                for child in node.get("children", []):
                    gates.extend(count_gates(child))
                return gates

            predicted = count_gates(hierarchy)
            for gate in predicted:
                pop_stats[gate]["total_appearances"] += 1

    return dict(pop_stats)


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
        # Get predicted gates
        parsed = result.get("parsed_hierarchy", {})
        predicted_gates = extract_gates_from_hierarchy(parsed) if parsed else set()

        # Get ground truth gates
        gt_gates = set(result.get("ground_truth_gates", []))

        # Normalize names for comparison
        predicted_normalized = {g.lower().strip(): g for g in predicted_gates}
        gt_normalized = {g.lower().strip(): g for g in gt_gates}

        # For each ground truth gate, check if it was predicted
        for gt_norm, gt_orig in gt_normalized.items():
            if gt_norm in predicted_normalized:
                pop_stats[gt_orig]["matches"] += 1
            else:
                pop_stats[gt_orig]["misses"] += 1

    return dict(pop_stats)


def load_predictions_with_evaluation(predictions_path: Path) -> dict[str, dict]:
    """Load predictions file which may have matching/missing gates."""
    with open(predictions_path) as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, list):
        predictions = data
    else:
        predictions = data.get("predictions", [])

    pop_stats = defaultdict(lambda: {"matches": 0, "misses": 0})

    for pred in predictions:
        eval_data = pred.get("evaluation", {})

        for gate in eval_data.get("matching_gates", []):
            pop_stats[gate.strip()]["matches"] += 1

        for gate in eval_data.get("missing_gates", []):
            pop_stats[gate.strip()]["misses"] += 1

    return dict(pop_stats)


def pearson_correlation(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Compute Pearson r, slope, intercept."""
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    ss_x = sum((xi - mean_x) ** 2 for xi in x)
    ss_y = sum((yi - mean_y) ** 2 for yi in y)

    if ss_x == 0 or ss_y == 0:
        return 0.0, 0.0, 0.0

    r = numerator / (math.sqrt(ss_x) * math.sqrt(ss_y))
    slope = numerator / ss_x if ss_x > 0 else 0
    intercept = mean_y - slope * mean_x

    return r, slope, intercept


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    def rank(values):
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0] * len(values)
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = rank_val + 1
        return ranks

    rank_x = rank(x)
    rank_y = rank(y)
    r, _, _ = pearson_correlation(rank_x, rank_y)
    return r


def normalize_name(name: str) -> str:
    """Normalize population name for matching."""
    return name.lower().strip()


def main():
    project_dir = Path(__file__).parent

    # Load actual PubMed frequencies
    pubmed_path = project_dir / "data" / "cache" / "pubmed_frequencies.json"
    if not pubmed_path.exists():
        print(f"ERROR: PubMed cache not found at {pubmed_path}")
        return

    pubmed_freqs = load_pubmed_frequencies(pubmed_path)
    print(f"Loaded {len(pubmed_freqs)} PubMed frequencies")
    print(f"Range: {min(pubmed_freqs.values())} - {max(pubmed_freqs.values())}")

    # Find results files
    results_dir = project_dir / "results" / "full_benchmark_20260114"
    predictions_path = results_dir / "predictions.json"

    if not predictions_path.exists():
        print(f"ERROR: Predictions not found at {predictions_path}")
        return

    # Load scoring results and extract per-gate match/miss stats
    scoring_path = results_dir / "scoring_results.json"
    pop_stats = load_scoring_results_with_gate_matching(scoring_path)
    print(f"\nExtracted stats for {len(pop_stats)} populations from scoring results")

    # Create normalized lookup for matching
    pubmed_normalized = {normalize_name(k): (k, v) for k, v in pubmed_freqs.items()}

    # Match populations and compute correlation
    matched_data = []
    unmatched = []

    for pop_name, stats in pop_stats.items():
        total = stats.get("matches", 0) + stats.get("misses", 0)
        if total == 0:
            continue

        detection_rate = stats["matches"] / total

        # Try to find matching PubMed frequency
        normalized = normalize_name(pop_name)

        if normalized in pubmed_normalized:
            orig_name, freq = pubmed_normalized[normalized]
            matched_data.append({
                "name": pop_name,
                "pubmed_count": freq,
                "detection_rate": detection_rate,
                "matches": stats["matches"],
                "misses": stats["misses"],
            })
        else:
            # Try partial matching
            matched = False
            for norm_key, (orig, freq) in pubmed_normalized.items():
                if norm_key in normalized or normalized in norm_key:
                    matched_data.append({
                        "name": pop_name,
                        "pubmed_count": freq,
                        "detection_rate": detection_rate,
                        "matches": stats["matches"],
                        "misses": stats["misses"],
                    })
                    matched = True
                    break

            if not matched:
                unmatched.append(pop_name)

    print(f"\nMatched {len(matched_data)} populations with PubMed data")
    if unmatched:
        print(f"Unmatched: {len(unmatched)} - {unmatched[:5]}...")

    if len(matched_data) < 10:
        print("\nInsufficient matched data for meaningful correlation analysis")
        print("Falling back to using original visualization script methodology comparison...")

        # Fall back: compute correlation using the estimated frequency approach
        # to verify the original R² = 0.034 claim
        print("\n" + "=" * 70)
        print("VERIFICATION: Using original estimated frequency methodology")
        print("=" * 70)

        # Use the same estimation approach as visualize_frequency_confound.py
        FREQ_ESTIMATES = {
            "t cells": 5, "t cell": 5, "cd4": 5, "cd8": 5, "cd3": 5,
            "b cells": 5, "b cell": 5, "cd19": 5, "cd20": 5,
            "monocytes": 5, "cd14": 5, "lymphocytes": 5,
            "nk cells": 4, "nk cell": 4, "cd56": 4,
            "neutrophils": 4, "basophils": 4, "eosinophils": 4,
            "dendritic": 4, "macrophages": 4, "memory": 4, "naive": 4,
            "regulatory": 4, "treg": 4,
            "pdc": 3, "mdc": 3, "cd1c": 3, "cd141": 3,
            "plasma cells": 3, "plasmablasts": 3, "mait": 3, "nkt": 3,
            "effector": 3, "central memory": 3,
            "temra": 2, "tfh": 2, "th1": 2, "th2": 2, "th17": 2,
            "cd27": 2, "cd45ra": 2, "ccr7": 2,
            "igd": 2, "igm": 2, "igg": 2, "iga": 2,
        }

        def estimate_freq(name):
            name_lower = name.lower()
            for term, level in FREQ_ESTIMATES.items():
                if term in name_lower:
                    return level
            return 1  # Default: rare

        estimated_data = []
        for pop_name, stats in pop_stats.items():
            total = stats.get("matches", 0) + stats.get("misses", 0)
            if total == 0:
                continue
            detection_rate = stats["matches"] / total
            est_freq = estimate_freq(pop_name)
            estimated_data.append({
                "name": pop_name,
                "estimated_freq": est_freq,
                "detection_rate": detection_rate,
            })

        if estimated_data:
            x_est = [d["estimated_freq"] for d in estimated_data]
            y_det = [d["detection_rate"] for d in estimated_data]

            r_est, slope_est, intercept_est = pearson_correlation(x_est, y_det)
            r2_est = r_est ** 2

            print(f"\nEstimated Frequency (1-5 scale) Results:")
            print(f"  n = {len(estimated_data)}")
            print(f"  Pearson r = {r_est:.3f}")
            print(f"  R² = {r2_est:.3f}")
            print(f"  Regression: y = {slope_est:.3f}x + {intercept_est:.3f}")

        return

    # Compute correlations with actual PubMed data
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS WITH ACTUAL PUBMED FREQUENCIES")
    print("=" * 70)

    # Log-transform frequencies (standard practice for count data)
    x_log = [math.log10(max(d["pubmed_count"], 1)) for d in matched_data]
    y_det = [d["detection_rate"] for d in matched_data]

    # Raw frequencies
    x_raw = [d["pubmed_count"] for d in matched_data]

    # Pearson correlation with log-transformed frequencies
    r_log, slope_log, intercept_log = pearson_correlation(x_log, y_det)
    r2_log = r_log ** 2

    # Pearson correlation with raw frequencies
    r_raw, slope_raw, intercept_raw = pearson_correlation(x_raw, y_det)
    r2_raw = r_raw ** 2

    # Spearman correlation (rank-based, more robust)
    r_spearman = spearman_correlation(x_raw, y_det)

    print(f"\nSample size: n = {len(matched_data)}")
    print(f"\nPubMed frequency range in matched data:")
    print(f"  Min: {min(x_raw):,}")
    print(f"  Max: {max(x_raw):,}")
    print(f"  Median: {sorted(x_raw)[len(x_raw)//2]:,}")

    print(f"\n1. Log-transformed PubMed counts vs Detection Rate:")
    print(f"   Pearson r = {r_log:.4f}")
    print(f"   R² = {r2_log:.4f}")
    print(f"   Regression: detection = {slope_log:.4f} * log10(freq) + {intercept_log:.4f}")

    print(f"\n2. Raw PubMed counts vs Detection Rate:")
    print(f"   Pearson r = {r_raw:.4f}")
    print(f"   R² = {r2_raw:.4f}")

    print(f"\n3. Spearman rank correlation (most robust):")
    print(f"   Spearman rho = {r_spearman:.4f}")

    # Compare with estimated frequency approach
    print("\n" + "=" * 70)
    print("COMPARISON: Estimated (1-5) vs Actual PubMed Frequencies")
    print("=" * 70)

    # Compute estimated frequency for matched data
    FREQ_ESTIMATES = {
        "t cells": 5, "t cell": 5, "cd4": 5, "cd8": 5, "cd3": 5,
        "b cells": 5, "b cell": 5, "cd19": 5, "cd20": 5,
        "monocytes": 5, "cd14": 5, "lymphocytes": 5,
        "nk cells": 4, "nk cell": 4, "cd56": 4,
        "neutrophils": 4, "basophils": 4, "eosinophils": 4,
        "dendritic": 4, "macrophages": 4, "memory": 4, "naive": 4,
        "regulatory": 4, "treg": 4,
        "pdc": 3, "mdc": 3, "cd1c": 3, "cd141": 3,
        "plasma cells": 3, "plasmablasts": 3, "mait": 3, "nkt": 3,
        "effector": 3, "central memory": 3,
        "temra": 2, "tfh": 2, "th1": 2, "th2": 2, "th17": 2,
        "cd27": 2, "cd45ra": 2, "ccr7": 2,
        "igd": 2, "igm": 2, "igg": 2, "iga": 2,
    }

    def estimate_freq(name):
        name_lower = name.lower()
        for term, level in FREQ_ESTIMATES.items():
            if term in name_lower:
                return level
        return 1

    x_est = [estimate_freq(d["name"]) for d in matched_data]
    r_est, slope_est, intercept_est = pearson_correlation(x_est, y_det)
    r2_est = r_est ** 2

    print(f"\nEstimated Frequency (1-5 scale):")
    print(f"   Pearson r = {r_est:.4f}")
    print(f"   R² = {r2_est:.4f}")

    print(f"\nActual PubMed Frequency (log-transformed):")
    print(f"   Pearson r = {r_log:.4f}")
    print(f"   R² = {r2_log:.4f}")

    # Show correlation between estimated and actual
    r_est_vs_actual, _, _ = pearson_correlation(x_est, x_log)
    print(f"\nCorrelation between estimated and actual (log) frequencies:")
    print(f"   r = {r_est_vs_actual:.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if r2_log < 0.1:
        print(f"\n✓ R² = {r2_log:.3f} confirms LOW correlation")
        print("  Frequency does NOT explain model performance")
    elif r2_log < 0.25:
        print(f"\n~ R² = {r2_log:.3f} shows WEAK correlation")
        print("  Frequency has minor influence")
    elif r2_log < 0.5:
        print(f"\n! R² = {r2_log:.3f} shows MODERATE correlation")
        print("  Frequency contributes but isn't dominant")
    else:
        print(f"\n✗ R² = {r2_log:.3f} shows STRONG correlation")
        print("  Frequency may explain performance")

    # Show some examples
    print("\n" + "=" * 70)
    print("EXAMPLE DATA POINTS")
    print("=" * 70)

    sorted_by_freq = sorted(matched_data, key=lambda x: x["pubmed_count"], reverse=True)

    print("\nHighest frequency populations:")
    for d in sorted_by_freq[:5]:
        print(f"  {d['name'][:40]:<40} freq={d['pubmed_count']:>7,}  det={d['detection_rate']:.1%}")

    print("\nLowest frequency populations:")
    for d in sorted_by_freq[-5:]:
        print(f"  {d['name'][:40]:<40} freq={d['pubmed_count']:>7,}  det={d['detection_rate']:.1%}")

    # Paradox cases
    print("\nPotential paradox cases (high freq + low detection OR low freq + high detection):")
    for d in matched_data:
        if d["pubmed_count"] > 10000 and d["detection_rate"] < 0.3:
            print(f"  HIGH FREQ, LOW DET: {d['name'][:35]:<35} freq={d['pubmed_count']:>7,} det={d['detection_rate']:.1%}")
        elif d["pubmed_count"] < 100 and d["detection_rate"] > 0.7:
            print(f"  LOW FREQ, HIGH DET: {d['name'][:35]:<35} freq={d['pubmed_count']:>7,} det={d['detection_rate']:.1%}")


if __name__ == "__main__":
    main()
