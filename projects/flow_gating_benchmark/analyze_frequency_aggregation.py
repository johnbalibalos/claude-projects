#!/usr/bin/env python3
"""
Frequency Aggregation Analysis: Exact Match vs Synonym-Aggregated

Compares R² correlation between model performance and PubMed frequency using:
1. Exact match - raw PubMed counts for exact gate names
2. Synonym-aggregated - summed frequencies across all known synonyms

Hypothesis: Synonym aggregation provides a better proxy for training data exposure,
so the R² should increase (exact match underestimates the true correlation).

Usage:
    python analyze_frequency_aggregation.py
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Inline the synonym dictionary to avoid import issues
# This is a subset of src/evaluation/enhanced_normalization.py CELL_TYPE_SYNONYMS
CELL_TYPE_SYNONYMS: dict[str, str] = {
    # QC gates
    "singlets": "singlets", "single cells": "singlets", "fsc singlets": "singlets",
    "ssc singlets": "singlets", "non-aggregates": "singlets",
    "live cells": "live", "live": "live", "viable cells": "live", "viable": "live",
    "leukocytes": "leukocytes", "cd45+ leukocytes": "leukocytes", "cd45+": "leukocytes",
    "cd45+ cells": "leukocytes", "white blood cells": "leukocytes",
    "lymphocytes": "lymphocytes", "lymphs": "lymphocytes",
    # T cells
    "t cells": "t_cells", "t-cells": "t_cells", "t lymphocytes": "t_cells",
    "cd3+ t cells": "t_cells", "cd3+ t": "t_cells", "cd3+": "t_cells",
    # CD4 T cells
    "cd4+ t cells": "cd4_t_cells", "cd4 t cells": "cd4_t_cells", "cd4+ t": "cd4_t_cells",
    "helper t cells": "cd4_t_cells", "th cells": "cd4_t_cells", "t helper cells": "cd4_t_cells",
    # CD8 T cells
    "cd8+ t cells": "cd8_t_cells", "cd8 t cells": "cd8_t_cells", "cd8+ t": "cd8_t_cells",
    "cytotoxic t cells": "cd8_t_cells", "cytotoxic t": "cd8_t_cells", "ctl": "cd8_t_cells",
    # Memory subsets
    "naive t cells": "naive_t", "naive t": "naive_t",
    "central memory": "cm", "cm": "cm", "tcm": "cm", "central memory t cells": "cm",
    "effector memory": "em", "em": "em", "tem": "em", "effector memory t cells": "em",
    "temra": "temra", "emra": "temra", "effector memory ra": "temra",
    # Tregs
    "tregs": "tregs", "treg": "tregs", "regulatory t cells": "tregs", "regulatory t": "tregs",
    "t regulatory cells": "tregs",
    # Th subsets
    "th1": "th1", "th1 cells": "th1", "th2": "th2", "th2 cells": "th2",
    "th17": "th17", "th17 cells": "th17", "th22": "th22",
    # Tfh
    "tfh": "tfh", "tfh cells": "tfh", "t follicular helper cells": "tfh",
    "t follicular helper": "tfh", "follicular helper t cells": "tfh",
    # B cells
    "b cells": "b_cells", "b-cells": "b_cells", "b lymphocytes": "b_cells",
    "cd19+ b cells": "b_cells", "cd19+ b": "b_cells", "cd19+": "b_cells",
    "cd20+ b cells": "b_cells", "cd20+": "b_cells",
    # B cell subsets
    "naive b cells": "naive_b", "naive b": "naive_b",
    "memory b cells": "memory_b", "memory b": "memory_b",
    "plasma cells": "plasma_cells", "plasmablasts": "plasma_cells",
    "transitional b cells": "transitional_b", "transitional b": "transitional_b",
    "marginal zone b cells": "mz_b_cells",
    # NK cells
    "nk cells": "nk_cells", "nk": "nk_cells", "natural killer cells": "nk_cells",
    "natural killer": "nk_cells", "cd56+ nk cells": "nk_cells", "cd56+cd3-": "nk_cells",
    "cd56bright nk cells": "cd56bright_nk", "cd56bright nk": "cd56bright_nk", "cd56bright": "cd56bright_nk",
    "cd56dim nk cells": "cd56dim_nk", "cd56dim nk": "cd56dim_nk", "cd56dim": "cd56dim_nk",
    # NKT
    "nkt cells": "nkt_cells", "nkt": "nkt_cells", "nk-t cells": "nkt_cells",
    "nkt-like cells": "nkt_cells",
    # Monocytes
    "monocytes": "monocytes", "monos": "monocytes", "cd14+ monocytes": "monocytes",
    "classical monocytes": "classical_monocytes", "classical monos": "classical_monocytes",
    "intermediate monocytes": "intermediate_monocytes",
    "non-classical monocytes": "nonclassical_monocytes", "nonclassical monocytes": "nonclassical_monocytes",
    # Granulocytes
    "granulocytes": "granulocytes", "neutrophils": "neutrophils", "neuts": "neutrophils",
    "eosinophils": "eosinophils", "basophils": "basophils",
    # DCs
    "dendritic cells": "dcs", "dcs": "dcs", "dc": "dcs",
    "myeloid dcs": "mdcs", "mdcs": "mdcs", "mdc": "mdcs",
    "conventional dcs": "cdcs", "cdcs": "cdcs",
    "cdc1": "cdc1", "cdc1s": "cdc1", "cd141+ mdcs": "cdc1",
    "cdc2": "cdc2", "cdc2s": "cdc2", "cd1c+ mdcs": "cdc2",
    "plasmacytoid dcs": "pdcs", "plasmacytoid dendritic cells": "pdcs",
    "pdcs": "pdcs", "pdc": "pdcs",
    # Myeloid
    "myeloid cells": "myeloid", "myeloid": "myeloid",
    # Macrophages
    "macrophages": "macrophages",
    # ILCs
    "ilcs": "ilcs", "innate lymphoid cells": "ilcs",
    # MAIT
    "mait cells": "mait", "mait": "mait",
    # Gamma-delta
    "gamma delta t cells": "gd_t_cells", "gd t cells": "gd_t_cells",
    "γδ t cells": "gd_t_cells",
}


@dataclass
class FrequencyComparison:
    """Comparison data for a single population."""
    name: str
    exact_freq: int
    aggregated_freq: int
    canonical_form: str
    n_synonyms_found: int


def invert_synonym_dict() -> dict[str, list[str]]:
    """
    Invert CELL_TYPE_SYNONYMS: canonical -> [all synonyms].

    Original dict maps synonym -> canonical.
    We want canonical -> [list of synonyms].
    """
    canonical_to_synonyms: dict[str, list[str]] = defaultdict(list)

    for synonym, canonical in CELL_TYPE_SYNONYMS.items():
        canonical_to_synonyms[canonical].append(synonym)

    return dict(canonical_to_synonyms)


def get_canonical_form(name: str) -> str | None:
    """Get canonical form for a population name, if it exists."""
    name_lower = name.lower().strip()

    # Direct match
    if name_lower in CELL_TYPE_SYNONYMS:
        return CELL_TYPE_SYNONYMS[name_lower]

    # Partial match (check if any synonym is contained in the name)
    for synonym, canonical in CELL_TYPE_SYNONYMS.items():
        if synonym in name_lower:
            return canonical

    return None


def aggregate_frequencies(
    pubmed_freqs: dict[str, int],
    canonical_to_synonyms: dict[str, list[str]],
) -> dict[str, int]:
    """
    Compute aggregated frequency for each canonical form.

    For each canonical form, sum the PubMed frequencies of all its synonyms.
    """
    aggregated = {}

    for canonical, synonyms in canonical_to_synonyms.items():
        total = 0
        found_synonyms = []

        for syn in synonyms:
            # Try different capitalizations
            for variant in [syn, syn.title(), syn.upper(), syn.capitalize()]:
                if variant in pubmed_freqs:
                    total += pubmed_freqs[variant]
                    found_synonyms.append(variant)
                    break

        # Also check the original pubmed_freqs keys that map to this canonical
        for pop_name, freq in pubmed_freqs.items():
            pop_canonical = get_canonical_form(pop_name)
            if pop_canonical == canonical and pop_name.lower() not in [s.lower() for s in found_synonyms]:
                total += freq
                found_synonyms.append(pop_name)

        aggregated[canonical] = total

    return aggregated


def compute_pearson_r(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and R²."""
    n = len(x)
    if n < 3:
        return 0.0, 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return 0.0, 0.0

    r = numerator / (denom_x * denom_y)
    return r, r ** 2


def extract_gates_from_hierarchy(node: dict, gates: set[str] | None = None) -> set[str]:
    """Recursively extract gate names from hierarchy."""
    if gates is None:
        gates = set()

    name = node.get("name", "")
    if name and name.lower() not in {"all events", "root", "ungated"}:
        gates.add(name)

    for child in node.get("children", []):
        extract_gates_from_hierarchy(child, gates)

    return gates


def load_detection_rates(results_dir: Path) -> dict[str, float]:
    """
    Load detection rates from benchmark results by comparing predicted vs ground truth gates.

    Returns dict mapping population name -> detection rate.
    """
    # Find the full benchmark results (prefer full_benchmark directory)
    result_files = sorted(results_dir.glob("full_benchmark*/scoring_results.json"), reverse=True)
    if not result_files:
        result_files = sorted(results_dir.glob("**/scoring_results.json"), reverse=True)

    if not result_files:
        print("Warning: No result files found")
        return {}

    results_path = result_files[0]
    print(f"Loading detection rates from: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    # Track per-population match/miss across all results
    pop_data: dict[str, dict] = defaultdict(lambda: {"matches": 0, "total": 0})

    for result in results.get("results", []):
        gt_gates = set(result.get("ground_truth_gates", []))
        parsed = result.get("parsed_hierarchy")

        if not parsed or not gt_gates:
            continue

        pred_gates = extract_gates_from_hierarchy(parsed)

        # Normalize for comparison
        pred_lower = {g.lower().strip() for g in pred_gates}

        # For each ground truth gate, check if model found it
        for gt_gate in gt_gates:
            gt_norm = gt_gate.lower().strip()
            pop_data[gt_gate]["total"] += 1

            # Check if any predicted gate matches (simple containment check)
            matched = gt_norm in pred_lower or any(gt_norm in p or p in gt_norm for p in pred_lower)
            if matched:
                pop_data[gt_gate]["matches"] += 1

    # Compute detection rates
    detection_rates = {}
    for name, data in pop_data.items():
        if data["total"] > 0:
            detection_rates[name] = data["matches"] / data["total"]

    print(f"  Found {len(detection_rates)} populations with detection data")
    return detection_rates


def main():
    """Main analysis."""
    print("=" * 80)
    print("FREQUENCY AGGREGATION ANALYSIS")
    print("Comparing Exact Match vs Synonym-Aggregated PubMed Frequencies")
    print("=" * 80)

    # Load cached PubMed frequencies
    cache_path = Path(__file__).parent / "data" / "cache" / "pubmed_frequencies.json"
    if not cache_path.exists():
        print(f"Error: Cache file not found: {cache_path}")
        return

    with open(cache_path) as f:
        pubmed_freqs = json.load(f)

    print(f"\nLoaded {len(pubmed_freqs)} cached PubMed frequencies")

    # Invert synonym dictionary
    canonical_to_synonyms = invert_synonym_dict()
    print(f"Found {len(canonical_to_synonyms)} canonical forms with {len(CELL_TYPE_SYNONYMS)} total synonyms")

    # Compute aggregated frequencies
    aggregated_freqs = aggregate_frequencies(pubmed_freqs, canonical_to_synonyms)

    # Load detection rates
    results_dir = Path(__file__).parent / "results"
    detection_rates = load_detection_rates(results_dir)

    # Build comparison data
    comparisons: list[FrequencyComparison] = []

    for pop_name, exact_freq in pubmed_freqs.items():
        canonical = get_canonical_form(pop_name)

        if canonical and canonical in aggregated_freqs:
            agg_freq = aggregated_freqs[canonical]
            n_syns = len(canonical_to_synonyms.get(canonical, []))
        else:
            agg_freq = exact_freq
            n_syns = 1
            canonical = pop_name.lower()

        comparisons.append(FrequencyComparison(
            name=pop_name,
            exact_freq=exact_freq,
            aggregated_freq=agg_freq,
            canonical_form=canonical,
            n_synonyms_found=n_syns,
        ))

    # Print examples of aggregation effect
    print("\n" + "-" * 80)
    print("AGGREGATION EXAMPLES (showing synonym groupings)")
    print("-" * 80)

    # Sort by ratio of aggregated/exact (biggest gains)
    sorted_by_gain = sorted(
        [c for c in comparisons if c.exact_freq > 0],
        key=lambda c: c.aggregated_freq / c.exact_freq if c.exact_freq > 0 else 0,
        reverse=True
    )

    print(f"\n{'Population':<35} {'Exact':>10} {'Aggreg':>10} {'Ratio':>8} {'Canonical':<20}")
    print("-" * 95)

    for c in sorted_by_gain[:20]:
        ratio = c.aggregated_freq / c.exact_freq if c.exact_freq > 0 else 0
        print(f"{c.name[:34]:<35} {c.exact_freq:>10,} {c.aggregated_freq:>10,} {ratio:>7.1f}x {c.canonical_form[:19]:<20}")

    # Print populations with zero exact match but non-zero aggregated
    zero_exact = [c for c in comparisons if c.exact_freq == 0 and c.aggregated_freq > 0]
    if zero_exact:
        print(f"\n\nPOPULATIONS WITH ZERO EXACT MATCH BUT NON-ZERO AGGREGATED ({len(zero_exact)}):")
        print("-" * 80)
        for c in zero_exact[:10]:
            print(f"  {c.name:<40} → {c.canonical_form:<20} (agg: {c.aggregated_freq:,})")

    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    if detection_rates:
        # Filter to populations with detection rate data
        matched = [(c, detection_rates[c.name]) for c in comparisons if c.name in detection_rates]

        if matched:
            # Use log frequencies to handle wide range
            import math

            exact_log = [math.log10(max(c.exact_freq, 1)) for c, _ in matched]
            agg_log = [math.log10(max(c.aggregated_freq, 1)) for c, _ in matched]
            det_rates = [dr for _, dr in matched]

            r_exact, r2_exact = compute_pearson_r(exact_log, det_rates)
            r_agg, r2_agg = compute_pearson_r(agg_log, det_rates)

            print(f"\nSample size: {len(matched)} populations with detection rate data")
            print("\nExact Match Frequencies:")
            print(f"  Pearson r:  {r_exact:+.4f}")
            print(f"  R²:         {r2_exact:.4f}")

            print("\nSynonym-Aggregated Frequencies:")
            print(f"  Pearson r:  {r_agg:+.4f}")
            print(f"  R²:         {r2_agg:.4f}")

            delta_r2 = r2_agg - r2_exact
            print(f"\n  ΔR²:        {delta_r2:+.4f} ({delta_r2/r2_exact*100:+.1f}% change)" if r2_exact > 0 else "")

            if r2_agg > r2_exact:
                print("\n→ AGGREGATION IMPROVED CORRELATION")
                print("  This supports the hypothesis that exact match underestimates")
                print("  the true frequency-performance relationship.")
            else:
                print("\n→ AGGREGATION DID NOT IMPROVE CORRELATION")
                print("  The frequency confound may be weaker than expected,")
                print("  or aggregation introduced noise from imprecise synonym matching.")
    else:
        print("\nNo detection rate data available.")
        print("Run benchmark first to generate detection rate data.")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_exact = sum(c.exact_freq for c in comparisons)
    total_agg = sum(c.aggregated_freq for c in comparisons)

    zero_exact_count = sum(1 for c in comparisons if c.exact_freq == 0)
    zero_agg_count = sum(1 for c in comparisons if c.aggregated_freq == 0)

    has_canonical = sum(1 for c in comparisons if c.canonical_form != c.name.lower())

    print(f"\nTotal populations:             {len(comparisons)}")
    print(f"Mapped to canonical form:      {has_canonical} ({has_canonical/len(comparisons)*100:.1f}%)")
    print(f"Zero exact frequency:          {zero_exact_count}")
    print(f"Zero aggregated frequency:     {zero_agg_count}")
    print(f"\nMean exact frequency:          {total_exact/len(comparisons):,.0f}")
    print(f"Mean aggregated frequency:     {total_agg/len(comparisons):,.0f}")

    # Print canonical groups with their total frequencies
    print("\n" + "-" * 80)
    print("TOP CANONICAL GROUPS BY AGGREGATED FREQUENCY")
    print("-" * 80)

    sorted_canonical = sorted(aggregated_freqs.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Canonical Form':<25} {'Aggregated':>12} {'# Synonyms':>12}")
    print("-" * 55)
    for canonical, freq in sorted_canonical[:15]:
        n_syns = len(canonical_to_synonyms.get(canonical, []))
        print(f"{canonical:<25} {freq:>12,} {n_syns:>12}")


if __name__ == "__main__":
    main()
