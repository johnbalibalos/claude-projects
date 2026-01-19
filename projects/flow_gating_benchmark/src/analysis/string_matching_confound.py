#!/usr/bin/env python3
"""
String Matching Confound Analysis.

Tests whether the observed correlation between PubMed frequency and model performance
is confounded by the evaluation's synonym matching system.

Hypothesis: Canonical/well-known populations have:
  1. More synonyms defined in the evaluator → higher match rates (evaluator bias)
  2. More PubMed citations → higher frequency scores

If both are true, correlation could be an artifact of string matching, not LLM performance.

This script:
  1. Counts synonym coverage per canonical population
  2. Computes detection rates with and without synonym matching
  3. Tests if synonym coverage explains the frequency-performance correlation
  4. Identifies populations where matching differences affect results
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Import normalization module - use direct file import to avoid __init__.py issues
import sys
import importlib.util

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

# Direct import from normalization.py to avoid dependency chain issues
_norm_path = _project_root / "src" / "evaluation" / "normalization.py"
_norm_spec = importlib.util.spec_from_file_location("normalization", _norm_path)
_norm_module = importlib.util.module_from_spec(_norm_spec)
_norm_spec.loader.exec_module(_norm_module)

CELL_TYPE_SYNONYMS = _norm_module.CELL_TYPE_SYNONYMS
CELL_TYPE_HIERARCHY = _norm_module.CELL_TYPE_HIERARCHY
normalize_gate_name = _norm_module.normalize_gate_name
normalize_gate_semantic = _norm_module.normalize_gate_semantic
are_gates_equivalent = _norm_module.are_gates_equivalent


@dataclass
class SynonymCoverage:
    """Synonym coverage statistics for a canonical population."""
    canonical_form: str
    synonym_count: int
    synonyms: list[str] = field(default_factory=list)

    @property
    def coverage_score(self) -> float:
        """Normalized coverage score (0-1 scale based on max observed)."""
        # Max observed is ~10 synonyms for common types
        return min(self.synonym_count / 10.0, 1.0)


@dataclass
class MatchingComparison:
    """Comparison of matching with vs without synonyms."""
    population: str
    ground_truth_name: str
    predicted_name: str
    exact_match: bool           # Raw string match (lowercase, stripped)
    normalized_match: bool      # After basic normalization
    semantic_match: bool        # After synonym resolution

    @property
    def match_difference(self) -> str:
        """Describe what level of matching was required."""
        if self.exact_match:
            return "exact"
        elif self.normalized_match:
            return "normalized"
        elif self.semantic_match:
            return "semantic"
        else:
            return "no_match"


def compute_synonym_coverage() -> dict[str, SynonymCoverage]:
    """
    Compute synonym coverage for each canonical population.

    Returns mapping from canonical form to coverage stats.
    """
    # Invert the synonym mapping: canonical -> list of synonyms
    canonical_to_synonyms: dict[str, list[str]] = defaultdict(list)

    for synonym, canonical in CELL_TYPE_SYNONYMS.items():
        canonical_to_synonyms[canonical].append(synonym)

    # Build coverage objects
    coverage = {}
    for canonical, synonyms in canonical_to_synonyms.items():
        coverage[canonical] = SynonymCoverage(
            canonical_form=canonical,
            synonym_count=len(synonyms),
            synonyms=sorted(synonyms),
        )

    return coverage


def exact_string_match(name1: str, name2: str) -> bool:
    """Strict exact match (lowercase, stripped only)."""
    return name1.lower().strip() == name2.lower().strip()


def normalized_string_match(name1: str, name2: str) -> bool:
    """Match after basic normalization (no semantic synonyms)."""
    return normalize_gate_name(name1) == normalize_gate_name(name2)


def semantic_string_match(name1: str, name2: str) -> bool:
    """Match after full semantic normalization with synonyms."""
    return are_gates_equivalent(name1, name2, semantic=True)


def compare_matching_levels(
    ground_truth: str,
    predicted: str,
) -> MatchingComparison:
    """
    Compare different matching levels for a ground truth / predicted pair.
    """
    return MatchingComparison(
        population=ground_truth,
        ground_truth_name=ground_truth,
        predicted_name=predicted,
        exact_match=exact_string_match(ground_truth, predicted),
        normalized_match=normalized_string_match(ground_truth, predicted),
        semantic_match=semantic_string_match(ground_truth, predicted),
    )


def evaluate_with_matcher(
    ground_truth_gates: list[str],
    predicted_gates: list[str],
    matcher: Callable[[str, str], bool],
) -> tuple[list[str], list[str], list[str]]:
    """
    Evaluate gate matching using a specific matcher function.

    Returns: (matched, missing, extra)
    """
    matched = []
    missing = []
    extra = []

    gt_remaining = set(ground_truth_gates)
    pred_remaining = set(predicted_gates)

    # Find matches
    for gt in ground_truth_gates:
        for pred in list(pred_remaining):
            if matcher(gt, pred):
                matched.append(gt)
                gt_remaining.discard(gt)
                pred_remaining.discard(pred)
                break

    missing = list(gt_remaining)
    extra = list(pred_remaining)

    return matched, missing, extra


def analyze_results_file(
    results_path: Path,
    ground_truth_dir: Path | None = None,
) -> dict:
    """
    Analyze a benchmark results file for string matching confound.

    Returns detailed analysis including:
    - Per-population synonym coverage vs detection rate
    - Comparison of detection rates at different matching levels
    - Statistical tests for confound hypothesis
    """
    with open(results_path) as f:
        results = json.load(f)

    # Collect per-population statistics at each matching level
    stats = {
        "exact": defaultdict(lambda: {"matches": 0, "misses": 0}),
        "normalized": defaultdict(lambda: {"matches": 0, "misses": 0}),
        "semantic": defaultdict(lambda: {"matches": 0, "misses": 0}),
    }

    # Track individual match comparisons
    match_comparisons: list[MatchingComparison] = []

    for result in results.get("results", []):
        eval_data = result.get("evaluation", {})

        # Get ground truth and predicted gates
        gt_gates = eval_data.get("matching_gates", []) + eval_data.get("missing_gates", [])
        pred_gates = eval_data.get("matching_gates", []) + eval_data.get("extra_gates", [])

        if not gt_gates:
            continue

        # Evaluate at each matching level
        for level, matcher in [
            ("exact", exact_string_match),
            ("normalized", normalized_string_match),
            ("semantic", semantic_string_match),
        ]:
            matched, missing, _ = evaluate_with_matcher(gt_gates, pred_gates, matcher)

            for gate in matched:
                stats[level][gate]["matches"] += 1
            for gate in missing:
                stats[level][gate]["misses"] += 1

        # Track detailed comparisons for matched gates
        for gt in gt_gates:
            for pred in pred_gates:
                if semantic_string_match(gt, pred):
                    match_comparisons.append(compare_matching_levels(gt, pred))
                    break

    return {
        "stats": {k: dict(v) for k, v in stats.items()},
        "comparisons": match_comparisons,
        "n_results": len(results.get("results", [])),
    }


def compute_detection_rates(stats: dict) -> dict[str, dict[str, float]]:
    """
    Compute detection rates at each matching level.

    Returns: {population: {level: rate}}
    """
    rates = defaultdict(dict)

    for level in ["exact", "normalized", "semantic"]:
        level_stats = stats.get(level, {})
        for pop, data in level_stats.items():
            total = data["matches"] + data["misses"]
            if total > 0:
                rates[pop][level] = data["matches"] / total
            else:
                rates[pop][level] = 0.0

    return dict(rates)


def compute_synonym_boost(detection_rates: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    Compute the 'synonym boost' - increase in detection from using synonyms.

    synonym_boost = semantic_rate - exact_rate

    Populations with high synonym boost may be inflating correlation.
    """
    boosts = {}

    for pop, rates in detection_rates.items():
        exact = rates.get("exact", 0.0)
        semantic = rates.get("semantic", 0.0)
        boosts[pop] = semantic - exact

    return boosts


def correlate_coverage_with_detection(
    coverage: dict[str, SynonymCoverage],
    detection_rates: dict[str, dict[str, float]],
) -> dict:
    """
    Test correlation between synonym coverage and detection rate.

    If high correlation: evaluator bias may explain frequency correlation.
    """
    try:
        from scipy import stats as scipy_stats
        import numpy as np
    except ImportError:
        return {"error": "scipy/numpy not available"}

    # Match populations
    common_pops = []
    for pop, rates in detection_rates.items():
        canonical = normalize_gate_semantic(pop)
        if canonical in coverage:
            common_pops.append((pop, canonical, rates.get("semantic", 0.0)))

    if len(common_pops) < 5:
        return {"error": f"Too few common populations ({len(common_pops)})"}

    # Build arrays
    syn_counts = []
    det_rates = []

    for pop, canonical, rate in common_pops:
        syn_counts.append(coverage[canonical].synonym_count)
        det_rates.append(rate)

    syn_counts = np.array(syn_counts)
    det_rates = np.array(det_rates)

    # Correlations
    pearson_r, pearson_p = scipy_stats.pearsonr(syn_counts, det_rates)
    spearman_r, spearman_p = scipy_stats.spearmanr(syn_counts, det_rates)

    return {
        "n_populations": len(common_pops),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "interpretation": interpret_coverage_correlation(spearman_r),
    }


def interpret_coverage_correlation(r: float) -> str:
    """Interpret the synonym coverage correlation."""
    if abs(r) < 0.1:
        return "No evidence of synonym coverage confound"
    elif abs(r) < 0.3:
        return "Weak synonym coverage effect - unlikely to explain frequency correlation"
    elif abs(r) < 0.5:
        return "Moderate synonym coverage effect - may partially explain frequency correlation"
    else:
        return "Strong synonym coverage effect - STRING MATCHING CONFOUND LIKELY"


def analyze_match_level_distribution(comparisons: list[MatchingComparison]) -> dict:
    """
    Analyze the distribution of match levels.

    If most matches require semantic resolution, synonym bias is high.
    """
    counts = {"exact": 0, "normalized": 0, "semantic": 0}

    for comp in comparisons:
        if comp.exact_match:
            counts["exact"] += 1
        elif comp.normalized_match:
            counts["normalized"] += 1
        elif comp.semantic_match:
            counts["semantic"] += 1

    total = sum(counts.values())
    if total == 0:
        return {"error": "No matches found"}

    percentages = {k: v / total * 100 for k, v in counts.items()}

    return {
        "counts": counts,
        "percentages": percentages,
        "semantic_dependency": percentages.get("semantic", 0),
        "interpretation": interpret_semantic_dependency(percentages.get("semantic", 0)),
    }


def interpret_semantic_dependency(semantic_pct: float) -> str:
    """Interpret semantic matching dependency."""
    if semantic_pct < 10:
        return "Low semantic dependency - most matches are exact or normalized"
    elif semantic_pct < 30:
        return "Moderate semantic dependency - some synonym matching needed"
    else:
        return "High semantic dependency - EVALUATION RELIES HEAVILY ON SYNONYMS"


def generate_confound_report(
    results_path: Path,
    output_path: Path | None = None,
) -> str:
    """
    Generate a comprehensive report on string matching confound.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STRING MATCHING CONFOUND ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Results file: {results_path.name}")
    lines.append("")

    # Compute synonym coverage
    coverage = compute_synonym_coverage()
    lines.append(f"Synonym coverage: {len(coverage)} canonical forms defined")
    lines.append("")

    # Show top/bottom coverage
    sorted_coverage = sorted(coverage.items(), key=lambda x: x[1].synonym_count, reverse=True)

    lines.append("TOP 10 - Most synonyms defined:")
    for canonical, cov in sorted_coverage[:10]:
        lines.append(f"  {cov.synonym_count:>3} synonyms: {canonical}")

    lines.append("")
    lines.append("BOTTOM 10 - Fewest synonyms defined:")
    for canonical, cov in sorted_coverage[-10:]:
        lines.append(f"  {cov.synonym_count:>3} synonyms: {canonical}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("DETECTION RATE COMPARISON BY MATCHING LEVEL")
    lines.append("-" * 80)

    # Analyze results
    analysis = analyze_results_file(results_path)
    detection_rates = compute_detection_rates(analysis["stats"])

    # Compute overall detection rates at each level
    for level in ["exact", "normalized", "semantic"]:
        level_stats = analysis["stats"].get(level, {})
        total_matches = sum(d["matches"] for d in level_stats.values())
        total_misses = sum(d["misses"] for d in level_stats.values())
        total = total_matches + total_misses
        rate = total_matches / total if total > 0 else 0
        lines.append(f"  {level:>12}: {rate:.1%} detection rate ({total_matches}/{total})")

    # Compute synonym boost
    lines.append("")
    lines.append("-" * 80)
    lines.append("SYNONYM BOOST ANALYSIS")
    lines.append("-" * 80)
    lines.append("(Increase in detection rate from using synonyms)")
    lines.append("")

    boosts = compute_synonym_boost(detection_rates)
    sorted_boosts = sorted(boosts.items(), key=lambda x: x[1], reverse=True)

    lines.append("Populations with HIGHEST synonym boost (most inflated by matching):")
    for pop, boost in sorted_boosts[:10]:
        if boost > 0:
            rates = detection_rates.get(pop, {})
            lines.append(f"  +{boost:.0%}: {pop}")
            lines.append(f"         exact={rates.get('exact', 0):.0%} → semantic={rates.get('semantic', 0):.0%}")

    # Match level distribution
    lines.append("")
    lines.append("-" * 80)
    lines.append("MATCH LEVEL DISTRIBUTION")
    lines.append("-" * 80)

    match_dist = analyze_match_level_distribution(analysis["comparisons"])
    if "error" not in match_dist:
        lines.append(f"  Exact matches:      {match_dist['percentages'].get('exact', 0):>5.1f}%")
        lines.append(f"  Normalized matches: {match_dist['percentages'].get('normalized', 0):>5.1f}%")
        lines.append(f"  Semantic matches:   {match_dist['percentages'].get('semantic', 0):>5.1f}%")
        lines.append("")
        lines.append(f"  → {match_dist['interpretation']}")

    # Coverage correlation
    lines.append("")
    lines.append("-" * 80)
    lines.append("SYNONYM COVERAGE vs DETECTION RATE CORRELATION")
    lines.append("-" * 80)

    corr_results = correlate_coverage_with_detection(coverage, detection_rates)
    if "error" not in corr_results:
        lines.append(f"  n populations: {corr_results['n_populations']}")
        lines.append(f"  Spearman r:    {corr_results['spearman_r']:.3f} (p={corr_results['spearman_p']:.4f})")
        lines.append(f"  Pearson r:     {corr_results['pearson_r']:.3f} (p={corr_results['pearson_p']:.4f})")
        lines.append("")
        lines.append(f"  → {corr_results['interpretation']}")
    else:
        lines.append(f"  Error: {corr_results['error']}")

    # Final verdict
    lines.append("")
    lines.append("=" * 80)
    lines.append("CONFOUND ASSESSMENT SUMMARY")
    lines.append("=" * 80)

    confound_score = 0
    confound_reasons = []

    # Check semantic dependency
    if "error" not in match_dist:
        semantic_dep = match_dist.get("semantic_dependency", 0)
        if semantic_dep > 30:
            confound_score += 2
            confound_reasons.append(f"High semantic dependency ({semantic_dep:.0f}%)")
        elif semantic_dep > 10:
            confound_score += 1
            confound_reasons.append(f"Moderate semantic dependency ({semantic_dep:.0f}%)")

    # Check coverage correlation
    if "error" not in corr_results:
        coverage_r = abs(corr_results.get("spearman_r", 0))
        if coverage_r > 0.5:
            confound_score += 2
            confound_reasons.append(f"Strong coverage-detection correlation (r={coverage_r:.2f})")
        elif coverage_r > 0.3:
            confound_score += 1
            confound_reasons.append(f"Moderate coverage-detection correlation (r={coverage_r:.2f})")

    # Check synonym boost spread
    boost_values = [b for b in boosts.values() if b > 0]
    if boost_values:
        avg_boost = sum(boost_values) / len(boost_values)
        if avg_boost > 0.3:
            confound_score += 2
            confound_reasons.append(f"High average synonym boost ({avg_boost:.0%})")
        elif avg_boost > 0.1:
            confound_score += 1
            confound_reasons.append(f"Moderate average synonym boost ({avg_boost:.0%})")

    if confound_score >= 4:
        verdict = "HIGH RISK: String matching likely confounds frequency correlation"
    elif confound_score >= 2:
        verdict = "MODERATE RISK: String matching may partially explain frequency correlation"
    else:
        verdict = "LOW RISK: String matching unlikely to explain frequency correlation"

    lines.append("")
    lines.append(f"Confound score: {confound_score}/6")
    lines.append(f"Verdict: {verdict}")
    lines.append("")
    if confound_reasons:
        lines.append("Contributing factors:")
        for reason in confound_reasons:
            lines.append(f"  - {reason}")

    lines.append("")
    lines.append("RECOMMENDATIONS:")
    if confound_score >= 2:
        lines.append("  1. Re-run correlation analysis using exact matching only")
        lines.append("  2. Control for synonym coverage in regression model")
        lines.append("  3. Manually review high-boost populations for evaluation bias")
    else:
        lines.append("  1. String matching confound appears minimal")
        lines.append("  2. Frequency correlation likely reflects real LLM performance differences")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report saved to {output_path}")

    return report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="String Matching Confound Analysis")
    parser.add_argument("--results", type=Path, help="Results JSON file")
    parser.add_argument("--output", type=Path, help="Output report path")
    parser.add_argument("--coverage-only", action="store_true",
                        help="Only show synonym coverage statistics")
    args = parser.parse_args()

    if args.coverage_only:
        # Just show coverage stats
        coverage = compute_synonym_coverage()
        print(f"\nSynonym Coverage Analysis ({len(coverage)} canonical forms)")
        print("=" * 60)

        sorted_coverage = sorted(coverage.items(), key=lambda x: x[1].synonym_count, reverse=True)
        for canonical, cov in sorted_coverage:
            print(f"\n{canonical} ({cov.synonym_count} synonyms):")
            for syn in cov.synonyms[:5]:
                print(f"  - {syn}")
            if len(cov.synonyms) > 5:
                print(f"  ... and {len(cov.synonyms) - 5} more")
        return

    # Find results file
    if args.results:
        results_path = args.results
    else:
        results_dir = Path(__file__).parent.parent.parent / "results"
        result_files = sorted(results_dir.glob("experiment_results_*.json"), reverse=True)
        if not result_files:
            result_files = sorted(results_dir.glob("benchmark_results_*.json"), reverse=True)
        if not result_files:
            print("No results file found. Use --results to specify.")
            return
        results_path = result_files[0]

    # Generate report
    output_path = args.output or Path(__file__).parent.parent.parent / "results" / "string_matching_confound_report.txt"
    report = generate_confound_report(results_path, output_path)
    print(report)


if __name__ == "__main__":
    main()
