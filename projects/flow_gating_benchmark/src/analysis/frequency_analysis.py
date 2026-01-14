#!/usr/bin/env python3
"""
Frequency Quantification Analysis

Correlates PubMed occurrence frequency with model performance per population.
Tests the hypothesis: Is performance driven by training data frequency or reasoning capability?

Usage:
    python -m src.analysis.frequency_analysis --results results/gemini_debug/debug_results_*.json
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Try to import Biopython, fall back to requests if not available
try:
    from Bio import Entrez
    USE_BIOPYTHON = True
except ImportError:
    import requests
    USE_BIOPYTHON = False


@dataclass
class PopulationFrequency:
    """Frequency data for a cell population."""
    name: str
    pubmed_count: int
    normalized_name: str  # Cleaned version for matching
    is_canonical: bool  # True for well-known populations


# Canonical populations (high expected frequency)
CANONICAL_POPULATIONS = {
    "t cells", "cd3+ t cells", "cd4+ t cells", "cd8+ t cells",
    "b cells", "cd19+ b cells", "nk cells", "natural killer cells",
    "monocytes", "cd14+ monocytes", "lymphocytes", "granulocytes",
    "neutrophils", "dendritic cells", "macrophages", "tregs",
    "regulatory t cells", "naive t cells", "memory t cells",
    "effector t cells", "helper t cells", "cytotoxic t cells",
    "plasma cells", "plasmablasts", "basophils", "eosinophils",
}


def normalize_population_name(name: str) -> str:
    """Normalize population name for matching."""
    # Lowercase
    name = name.lower()
    # Remove common prefixes/suffixes
    name = re.sub(r'\bcells?\b', '', name)
    name = re.sub(r'\bpopulation\b', '', name)
    # Clean up whitespace
    name = ' '.join(name.split()).strip()
    return name


def is_canonical(name: str) -> bool:
    """Check if population is canonical (well-known)."""
    normalized = normalize_population_name(name)
    return normalized in CANONICAL_POPULATIONS or any(
        canon in normalized for canon in CANONICAL_POPULATIONS
    )


def get_pubmed_count_biopython(query: str, email: str = "research@example.com") -> int:
    """Get PubMed count using Biopython Entrez."""
    Entrez.email = email
    try:
        handle = Entrez.esearch(db="pubmed", term=f'"{query}"', retmax=0)
        record = Entrez.read(handle)
        handle.close()
        time.sleep(0.35)  # Rate limit: 3/sec without API key
        return int(record["Count"])
    except Exception as e:
        print(f"  Error querying '{query}': {e}")
        return 0


def get_pubmed_count_requests(query: str) -> int:
    """Get PubMed count using direct API request."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f'"{query}"',
        "retmax": 0,
        "retmode": "json",
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        time.sleep(0.35)  # Rate limit
        return int(data["esearchresult"]["count"])
    except Exception as e:
        print(f"  Error querying '{query}': {e}")
        return 0


def get_pubmed_count(query: str, email: str = "research@example.com") -> int:
    """Get PubMed occurrence count for a query."""
    if USE_BIOPYTHON:
        return get_pubmed_count_biopython(query, email)
    else:
        return get_pubmed_count_requests(query)


def extract_populations_from_ground_truth(gt_dir: Path) -> list[str]:
    """Extract all unique population names from ground truth files."""
    populations = set()

    # Skip generic/QC gates
    skip_gates = {
        "all events", "root", "ungated", "singlets", "live", "live cells",
        "lymphocytes", "time", "doublet exclusion", "viable", "viability",
    }

    for f in gt_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue

        # Recursively extract gate names from hierarchy
        def walk(node):
            if isinstance(node, dict):
                if "name" in node:
                    name = node["name"]
                    # Skip generic gates
                    if name.lower() not in skip_gates:
                        populations.add(name)
                for child in node.get("children", []):
                    walk(child)

        # Handle nested structure: gating_hierarchy.root or gating_hierarchy directly
        hierarchy = data.get("gating_hierarchy", {})
        if isinstance(hierarchy, dict):
            root = hierarchy.get("root", hierarchy)
            walk(root)

    return sorted(populations)


def extract_populations_from_results(results_file: Path) -> dict[str, float]:
    """Extract per-population F1 scores from experiment results."""
    # This requires the results to have per-gate breakdown
    # For now, return empty - will need to modify scorer to track per-population metrics
    data = json.loads(results_file.read_text())

    # Aggregate by matching/missing gates across all conditions
    population_scores = {}

    for _result in data.get("results", []):
        # Would need per-gate metrics in the result format
        # For now, we'll compute from the detailed trial data if available
        pass

    return population_scores


def query_all_populations(populations: list[str], cache_file: Path = None) -> dict[str, int]:
    """Query PubMed for all populations, with optional caching."""
    if cache_file and cache_file.exists():
        print(f"Loading cached frequencies from {cache_file}")
        return json.loads(cache_file.read_text())

    print(f"Querying PubMed for {len(populations)} populations...")
    frequencies = {}

    for i, pop in enumerate(populations):
        print(f"  [{i+1}/{len(populations)}] {pop}...", end=" ", flush=True)
        count = get_pubmed_count(pop)
        frequencies[pop] = count
        print(f"{count:,}")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(frequencies, indent=2))
        print(f"Cached to {cache_file}")

    return frequencies


def plot_frequency_vs_f1(
    frequencies: dict[str, int],
    f1_scores: dict[str, float],
    output_path: Path,
):
    """Generate scatter plot of frequency vs F1 score."""
    from scipy import stats

    # Match populations
    common = set(frequencies.keys()) & set(f1_scores.keys())
    if not common:
        print("Warning: No common populations between frequency and F1 data")
        return

    populations = sorted(common)
    freqs = [max(frequencies[p], 1) for p in populations]  # Avoid log(0)
    f1s = [f1_scores[p] for p in populations]

    # Classify as canonical or derived
    canonical_mask = [is_canonical(p) for p in populations]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot points
    for i, (freq, f1, _pop) in enumerate(zip(freqs, f1s, populations)):
        color = 'blue' if canonical_mask[i] else 'red'
        marker = 'o' if canonical_mask[i] else 's'
        ax.scatter(freq, f1, c=color, marker=marker, s=100, alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('PubMed Occurrence Count (log scale)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)

    # Compute correlation
    log_freqs = np.log10(freqs)
    r_pearson, p_pearson = stats.pearsonr(log_freqs, f1s)
    r_spearman, p_spearman = stats.spearmanr(freqs, f1s)

    ax.set_title(
        f'Population Frequency vs Model Performance\n'
        f'Spearman r={r_spearman:.3f} (p={p_spearman:.4f}), '
        f'Pearson r={r_pearson:.3f} (p={p_pearson:.4f})',
        fontsize=11
    )

    # Add regression line
    z = np.polyfit(log_freqs, f1s, 1)
    x_line = np.logspace(min(log_freqs), max(log_freqs), 100)
    ax.plot(x_line, np.polyval(z, np.log10(x_line)), 'k--', alpha=0.5, label='Regression')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Canonical'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Derived'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # Add annotations for outliers
    for freq, f1, pop in zip(freqs, f1s, populations):
        # Annotate if outlier (high freq low F1, or low freq high F1)
        expected_f1 = np.polyval(z, np.log10(freq))
        residual = abs(f1 - expected_f1)
        if residual > 0.15:  # Threshold for outlier
            ax.annotate(
                pop[:20],  # Truncate long names
                (freq, f1),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points',
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    # Return statistics
    return {
        "n_populations": len(populations),
        "n_canonical": sum(canonical_mask),
        "n_derived": len(populations) - sum(canonical_mask),
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
    }


def main():
    parser = argparse.ArgumentParser(description="Frequency vs Performance Analysis")
    parser.add_argument("--gt-dir", type=Path, default=Path("data/verified"),
                       help="Ground truth directory")
    parser.add_argument("--results", type=Path, help="Results JSON file")
    parser.add_argument("--cache", type=Path, default=Path("data/cache/pubmed_frequencies.json"),
                       help="Cache file for PubMed frequencies")
    parser.add_argument("--output", type=Path, default=Path("results/frequency_vs_f1.png"),
                       help="Output plot path")
    parser.add_argument("--email", default="research@example.com",
                       help="Email for PubMed API")
    args = parser.parse_args()

    # Extract populations
    populations = extract_populations_from_ground_truth(args.gt_dir)
    print(f"Found {len(populations)} unique populations in ground truth")

    # Query PubMed
    frequencies = query_all_populations(populations, args.cache)

    # Print summary
    print("\n" + "=" * 50)
    print("FREQUENCY SUMMARY")
    print("=" * 50)

    sorted_pops = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 by frequency:")
    for pop, count in sorted_pops[:10]:
        canonical = "canonical" if is_canonical(pop) else "derived"
        print(f"  {count:>8,}  {pop} ({canonical})")

    print("\nBottom 10 by frequency:")
    for pop, count in sorted_pops[-10:]:
        canonical = "canonical" if is_canonical(pop) else "derived"
        print(f"  {count:>8,}  {pop} ({canonical})")

    # If results file provided, generate plot
    if args.results and args.results.exists():
        f1_scores = extract_populations_from_results(args.results)
        if f1_scores:
            stats = plot_frequency_vs_f1(frequencies, f1_scores, args.output)
            print("\n" + "=" * 50)
            print("CORRELATION ANALYSIS")
            print("=" * 50)
            print(f"Spearman r = {stats['spearman_r']:.3f} (p = {stats['spearman_p']:.4f})")
            print(f"Pearson r = {stats['pearson_r']:.3f} (p = {stats['pearson_p']:.4f})")

            if stats['spearman_r'] > 0.5:
                print("\n>>> HIGH CORRELATION: Frequency effect is significant")
            elif stats['spearman_r'] > 0.3:
                print("\n>>> MODERATE CORRELATION: Frequency contributes but not dominant")
            else:
                print("\n>>> LOW CORRELATION: Reasoning/complexity may be more important")


if __name__ == "__main__":
    main()
