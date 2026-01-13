#!/usr/bin/env python3
"""
Frequency Confound Analysis Script.

Analyzes the correlation between cell population term frequency in PubMed
and model performance (F1 score) to test the hypothesis:
- H_A: Performance correlates with frequency (memorization)
- H_B: Performance is independent of frequency (reasoning)

Generates visualizations for presentation.
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


@dataclass
class PopulationScore:
    """Score data for a single population."""
    name: str
    f1_scores: list[float]
    match_count: int
    miss_count: int

    @property
    def avg_f1(self) -> float:
        return sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0.0

    @property
    def detection_rate(self) -> float:
        total = self.match_count + self.miss_count
        return self.match_count / total if total > 0 else 0.0


def load_results(results_path: str) -> dict:
    """Load benchmark results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def extract_population_scores(results: dict) -> dict[str, PopulationScore]:
    """
    Extract per-population performance metrics from benchmark results.

    Tracks both matches and misses to compute detection rates.
    """
    population_data = defaultdict(lambda: {"f1_scores": [], "matches": 0, "misses": 0})

    for result in results.get("results", []):
        eval_data = result.get("evaluation", {})

        # Track matched gates
        for gate in eval_data.get("matching_gates", []):
            # Normalize gate name
            name = gate.strip()
            population_data[name]["matches"] += 1
            # Matched gates have implicit F1 contribution
            population_data[name]["f1_scores"].append(1.0)

        # Track missing gates (in ground truth but not predicted)
        for gate in eval_data.get("missing_gates", []):
            name = gate.strip()
            population_data[name]["misses"] += 1
            population_data[name]["f1_scores"].append(0.0)

        # Track hallucinated gates (predicted but not in ground truth)
        for gate in eval_data.get("extra_gates", []):
            name = gate.strip()
            # Don't count these as misses, just note the hallucination
            if name not in population_data:
                population_data[name]["f1_scores"].append(0.0)

    # Convert to PopulationScore objects
    scores = {}
    for name, data in population_data.items():
        scores[name] = PopulationScore(
            name=name,
            f1_scores=data["f1_scores"],
            match_count=data["matches"],
            miss_count=data["misses"],
        )

    return scores


def categorize_populations(populations: dict[str, PopulationScore]) -> dict[str, list[str]]:
    """
    Categorize populations by expected frequency in literature.

    This provides a manual sanity check before PubMed queries.
    """
    categories = {
        "common_t_cell": [],      # CD3, CD4, CD8, T cells - very common
        "common_b_cell": [],      # CD19, B cells - common
        "common_nk": [],          # NK cells, CD56 - common
        "common_myeloid": [],     # Monocytes, CD14 - common
        "specialized_dc": [],     # pDCs, mDCs, CD1c+ - less common
        "specialized_subset": [], # Tregs, TEMRA, etc. - specialized
        "rare_subset": [],        # Very specific populations
        "technical_gates": [],    # Singlets, Live, All Events
    }

    for name in populations:
        name_lower = name.lower()

        if any(x in name_lower for x in ["singlet", "live", "dead", "all event", "time", "lymphocyte"]):
            categories["technical_gates"].append(name)
        elif any(x in name_lower for x in ["t cell", "cd3+", "cd4+", "cd8+"]):
            categories["common_t_cell"].append(name)
        elif any(x in name_lower for x in ["b cell", "cd19", "cd20"]):
            categories["common_b_cell"].append(name)
        elif any(x in name_lower for x in ["nk cell", "nk-t", "cd56"]):
            categories["common_nk"].append(name)
        elif any(x in name_lower for x in ["monocyte", "cd14", "neutrophil", "eosinophil", "basophil"]):
            categories["common_myeloid"].append(name)
        elif any(x in name_lower for x in ["dc", "dendritic", "pdc", "mdc", "cd1c", "cd141"]):
            categories["specialized_dc"].append(name)
        elif any(x in name_lower for x in ["treg", "memory", "naive", "effector", "temra", "plasma"]):
            categories["specialized_subset"].append(name)
        else:
            categories["rare_subset"].append(name)

    return categories


def print_frequency_analysis(populations: dict[str, PopulationScore]):
    """Print a formatted analysis of population frequencies."""

    # Sort by detection rate
    sorted_pops = sorted(
        populations.items(),
        key=lambda x: x[1].detection_rate,
        reverse=True
    )

    print("\n" + "=" * 80)
    print("POPULATION DETECTION ANALYSIS")
    print("=" * 80)

    print(f"\n{'Population':<40} {'Det.Rate':>10} {'Matches':>10} {'Misses':>10}")
    print("-" * 80)

    for name, score in sorted_pops[:30]:  # Top 30
        print(f"{name[:39]:<40} {score.detection_rate:>10.1%} {score.match_count:>10} {score.miss_count:>10}")

    print("\n... (showing top 30)")

    # Categorize
    categories = categorize_populations(populations)

    print("\n" + "=" * 80)
    print("POPULATION CATEGORIES (for frequency hypothesis)")
    print("=" * 80)

    for cat_name, members in categories.items():
        if members:
            print(f"\n{cat_name.upper()} ({len(members)} populations):")

            # Calculate average detection rate for category
            cat_scores = [populations[m] for m in members if m in populations]
            if cat_scores:
                avg_det = sum(s.detection_rate for s in cat_scores) / len(cat_scores)
                print(f"  Average detection rate: {avg_det:.1%}")

            for m in sorted(members)[:5]:
                if m in populations:
                    s = populations[m]
                    print(f"  - {m}: {s.detection_rate:.1%}")
            if len(members) > 5:
                print(f"  ... and {len(members) - 5} more")


def generate_visualization_plan():
    """
    Print a plan for visualizations to create.
    """
    print("\n" + "=" * 80)
    print("RECOMMENDED VISUALIZATIONS")
    print("=" * 80)

    visualizations = [
        {
            "name": "1. Scatter Plot: Log(PubMed Citations) vs Detection Rate",
            "x_axis": "Log10(PubMed Citation Count + 1)",
            "y_axis": "Population Detection Rate (0-1)",
            "annotation": "R² value, regression line, 95% CI band",
            "interpretation": """
                - If R² > 0.8: Frequency hypothesis supported
                - Points should show clear linear relationship
                - Label outliers (high freq + low detection OR low freq + high detection)
            """,
        },
        {
            "name": "2. Box Plot: Detection Rate by Frequency Quintile",
            "x_axis": "PubMed Frequency Quintile (1=rare, 5=common)",
            "y_axis": "Detection Rate",
            "annotation": "Median, IQR, significance tests between groups",
            "interpretation": """
                - If frequency matters: clear staircase pattern
                - If reasoning matters: flat boxes across quintiles
            """,
        },
        {
            "name": "3. Heatmap: Population Category vs Model Performance",
            "x_axis": "Population Category (T cells, B cells, DCs, etc.)",
            "y_axis": "Metric (Detection Rate, F1, Hallucination Rate)",
            "annotation": "Color intensity = performance, annotate cells with values",
            "interpretation": """
                - Reveals if certain biological categories are harder
                - Common categories should be brighter if frequency matters
            """,
        },
        {
            "name": "4. Bar Chart: Top 10 vs Bottom 10 by Frequency",
            "x_axis": "Population Name",
            "y_axis": "Detection Rate",
            "annotation": "Color by frequency category, error bars",
            "interpretation": """
                - Direct comparison of most vs least common terms
                - Gap between groups = frequency effect size
            """,
        },
        {
            "name": "5. Residual Plot: Observed vs Predicted Performance",
            "x_axis": "Predicted Detection Rate (from frequency model)",
            "y_axis": "Residual (Observed - Predicted)",
            "annotation": "Highlight populations that deviate from frequency prediction",
            "interpretation": """
                - Populations above the line: better than frequency predicts (reasoning?)
                - Populations below: worse than frequency predicts (complexity?)
            """,
        },
    ]

    for viz in visualizations:
        print(f"\n{viz['name']}")
        print(f"  X-axis: {viz['x_axis']}")
        print(f"  Y-axis: {viz['y_axis']}")
        print(f"  Annotations: {viz['annotation']}")
        print(f"  Interpretation: {viz['interpretation'].strip()}")


def main():
    """Main entry point."""

    # Find the most recent results file
    results_dir = Path(__file__).parent / "results"
    result_files = sorted(results_dir.glob("experiment_results_*.json"), reverse=True)

    if not result_files:
        result_files = sorted(results_dir.glob("benchmark_results_*.json"), reverse=True)

    if not result_files:
        print("No result files found!")
        return

    results_path = result_files[0]
    print(f"Analyzing: {results_path.name}")

    # Load and analyze
    results = load_results(str(results_path))
    print(f"Loaded {results.get('n_results', 0)} results from {results.get('config_name', 'unknown')}")

    # Extract population scores
    populations = extract_population_scores(results)
    print(f"Found {len(populations)} unique populations")

    # Print analysis
    print_frequency_analysis(populations)

    # Print visualization plan
    generate_visualization_plan()

    # Summary statistics for the frequency hypothesis
    print("\n" + "=" * 80)
    print("FREQUENCY CONFOUND HYPOTHESIS SUMMARY")
    print("=" * 80)

    categories = categorize_populations(populations)

    # Compare common vs rare
    common_pops = (
        categories["common_t_cell"] +
        categories["common_b_cell"] +
        categories["common_nk"] +
        categories["common_myeloid"]
    )
    rare_pops = categories["specialized_dc"] + categories["specialized_subset"] + categories["rare_subset"]

    common_scores = [populations[p] for p in common_pops if p in populations]
    rare_scores = [populations[p] for p in rare_pops if p in populations]

    if common_scores and rare_scores:
        common_avg = sum(s.detection_rate for s in common_scores) / len(common_scores)
        rare_avg = sum(s.detection_rate for s in rare_scores) / len(rare_scores)

        print(f"\nCommon populations (n={len(common_scores)}): {common_avg:.1%} avg detection")
        print(f"Rare populations (n={len(rare_scores)}): {rare_avg:.1%} avg detection")
        print(f"Difference: {common_avg - rare_avg:.1%}")

        if common_avg - rare_avg > 0.2:
            print("\n→ PRELIMINARY: Large gap suggests frequency may explain performance")
        elif common_avg - rare_avg > 0.1:
            print("\n→ PRELIMINARY: Moderate gap - mixed evidence")
        else:
            print("\n→ PRELIMINARY: Small gap - frequency may not be the main factor")

    print("\nNote: This is a preliminary analysis using category-based frequency estimates.")
    print("For rigorous testing, run PubMed queries for actual citation counts.")


if __name__ == "__main__":
    main()
