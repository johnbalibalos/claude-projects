#!/usr/bin/env python3
"""
Generate visualizations for the Frequency Confound Hypothesis Test.

Creates publication-ready figures showing:
1. Scatter plot of frequency vs performance with regression
2. Box plot by frequency quintile
3. Category performance heatmap
4. Top/Bottom comparison bar chart
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Try to import visualization libraries
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available - will generate text-based visualizations")


@dataclass
class PopulationData:
    """Data for a single population."""
    name: str
    detection_rate: float
    match_count: int
    miss_count: int
    category: str
    estimated_frequency: int  # Rough estimate before PubMed lookup


# Rough frequency estimates (order of magnitude) for common terms
FREQUENCY_ESTIMATES = {
    # Very common (100k+ citations)
    "t cells": 5, "t cell": 5, "cd4": 5, "cd8": 5, "cd3": 5,
    "b cells": 5, "b cell": 5, "cd19": 5, "cd20": 5,
    "monocytes": 5, "cd14": 5,
    "lymphocytes": 5,

    # Common (10k-100k)
    "nk cells": 4, "nk cell": 4, "cd56": 4,
    "neutrophils": 4, "basophils": 4, "eosinophils": 4,
    "dendritic": 4, "macrophages": 4,
    "memory": 4, "naive": 4,
    "regulatory": 4, "treg": 4,

    # Moderate (1k-10k)
    "pdc": 3, "mdc": 3, "cd1c": 3, "cd141": 3,
    "plasma cells": 3, "plasmablasts": 3,
    "mait": 3, "nkt": 3,
    "cd4+": 3, "cd8+": 3,
    "effector": 3, "central memory": 3,

    # Less common (100-1k)
    "temra": 2, "tfh": 2, "th1": 2, "th2": 2, "th17": 2,
    "cd27": 2, "cd45ra": 2, "ccr7": 2,
    "igd": 2, "igm": 2, "igg": 2, "iga": 2,

    # Rare (<100 or highly specific)
    "default": 1,
}


def estimate_frequency(name: str) -> int:
    """Estimate frequency level (1-5) for a population name."""
    name_lower = name.lower()
    for term, level in FREQUENCY_ESTIMATES.items():
        if term in name_lower:
            return level
    return FREQUENCY_ESTIMATES["default"]


def categorize(name: str) -> str:
    """Categorize a population by type."""
    name_lower = name.lower()

    if any(x in name_lower for x in ["singlet", "live", "dead", "all event", "time"]):
        return "Technical"
    elif any(x in name_lower for x in ["t cell", "cd3", "cd4", "cd8"]) and "nk" not in name_lower:
        return "T cells"
    elif any(x in name_lower for x in ["b cell", "cd19", "cd20", "plasma", "igd", "igm", "igg", "iga"]):
        return "B cells"
    elif any(x in name_lower for x in ["nk", "cd56"]):
        return "NK cells"
    elif any(x in name_lower for x in ["monocyte", "cd14"]):
        return "Monocytes"
    elif any(x in name_lower for x in ["dc", "dendritic", "pdc", "mdc"]):
        return "DCs"
    elif any(x in name_lower for x in ["neutrophil", "eosinophil", "basophil", "granulocyte"]):
        return "Granulocytes"
    else:
        return "Other"


def load_and_process_results(results_path: str) -> list[PopulationData]:
    """Load results and compute population data."""
    with open(results_path) as f:
        results = json.load(f)

    # Aggregate by population
    pop_data = defaultdict(lambda: {"matches": 0, "misses": 0})

    for result in results.get("results", []):
        eval_data = result.get("evaluation", {})

        for gate in eval_data.get("matching_gates", []):
            pop_data[gate.strip()]["matches"] += 1

        for gate in eval_data.get("missing_gates", []):
            pop_data[gate.strip()]["misses"] += 1

    # Convert to PopulationData
    populations = []
    for name, data in pop_data.items():
        total = data["matches"] + data["misses"]
        if total > 0:
            populations.append(PopulationData(
                name=name,
                detection_rate=data["matches"] / total,
                match_count=data["matches"],
                miss_count=data["misses"],
                category=categorize(name),
                estimated_frequency=estimate_frequency(name),
            ))

    return populations


def compute_correlation(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Compute Pearson correlation and regression."""
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return 0.0, 0.0, 0.0

    r = numerator / (denom_x * denom_y)

    # Regression coefficients
    slope = numerator / (denom_x ** 2) if denom_x > 0 else 0
    intercept = mean_y - slope * mean_x

    return r, slope, intercept


def text_visualization(populations: list[PopulationData]):
    """Generate text-based visualizations when matplotlib is not available."""

    print("\n" + "=" * 80)
    print("FREQUENCY CONFOUND VISUALIZATION (Text Mode)")
    print("=" * 80)

    # Group by frequency level
    by_freq = defaultdict(list)
    for p in populations:
        by_freq[p.estimated_frequency].append(p)

    # 1. Bar chart by frequency level
    print("\n1. DETECTION RATE BY FREQUENCY LEVEL")
    print("-" * 60)

    for level in sorted(by_freq.keys(), reverse=True):
        pops = by_freq[level]
        avg_rate = sum(p.detection_rate for p in pops) / len(pops)
        n = len(pops)
        bar = "█" * int(avg_rate * 40)
        freq_label = ["Rare", "Uncommon", "Moderate", "Common", "Very Common"][level - 1]
        print(f"Level {level} ({freq_label:>12}, n={n:>3}): {bar} {avg_rate:.1%}")

    # 2. Category breakdown
    print("\n2. DETECTION RATE BY CATEGORY")
    print("-" * 60)

    by_cat = defaultdict(list)
    for p in populations:
        by_cat[p.category].append(p)

    for cat in sorted(by_cat.keys()):
        pops = by_cat[cat]
        avg_rate = sum(p.detection_rate for p in pops) / len(pops)
        n = len(pops)
        bar = "█" * int(avg_rate * 40)
        print(f"{cat:<15} (n={n:>3}): {bar} {avg_rate:.1%}")

    # 3. Correlation analysis
    print("\n3. CORRELATION ANALYSIS")
    print("-" * 60)

    x = [p.estimated_frequency for p in populations]
    y = [p.detection_rate for p in populations]
    r, slope, intercept = compute_correlation(x, y)
    r_squared = r ** 2

    print(f"Pearson r: {r:.3f}")
    print(f"R²: {r_squared:.3f}")
    print(f"Regression: y = {slope:.3f}x + {intercept:.3f}")

    if r_squared > 0.5:
        print("\n→ FREQUENCY HYPOTHESIS SUPPORTED: Strong correlation")
    elif r_squared > 0.25:
        print("\n→ MIXED EVIDENCE: Moderate correlation")
    else:
        print("\n→ REASONING HYPOTHESIS SUPPORTED: Weak correlation")

    # 4. Top/Bottom comparison
    print("\n4. TOP 10 vs BOTTOM 10 POPULATIONS")
    print("-" * 60)

    sorted_pops = sorted(populations, key=lambda p: p.detection_rate, reverse=True)

    print("\nTOP 10 (highest detection):")
    for p in sorted_pops[:10]:
        freq_label = ["Rare", "Uncommon", "Moderate", "Common", "V.Common"][p.estimated_frequency - 1]
        print(f"  {p.name[:35]:<35} {p.detection_rate:>6.1%} [{freq_label}]")

    print("\nBOTTOM 10 (lowest detection, excluding 0%):")
    non_zero = [p for p in sorted_pops if p.detection_rate > 0]
    for p in non_zero[-10:]:
        freq_label = ["Rare", "Uncommon", "Moderate", "Common", "V.Common"][p.estimated_frequency - 1]
        print(f"  {p.name[:35]:<35} {p.detection_rate:>6.1%} [{freq_label}]")

    # 5. Scatter plot approximation
    print("\n5. SCATTER PLOT (Frequency vs Detection Rate)")
    print("-" * 60)
    print("          | Detection Rate")
    print("Frequency | 0%      25%      50%      75%      100%")
    print("          |" + "-" * 45)

    for level in range(5, 0, -1):
        pops = by_freq.get(level, [])
        if pops:
            points = ["·"] * 45
            for p in pops:
                pos = min(44, int(p.detection_rate * 44))
                points[pos] = "●"
            print(f"    {level}     |{''.join(points)}")

    print("          |" + "-" * 45)
    print("          | Regression line: ", end="")
    for i in range(45):
        x_val = i / 44 * 4 + 1  # Map to frequency range 1-5
        y_pred = slope * x_val + intercept
        if 0 <= y_pred <= 1:
            pos = int(y_pred * 44)
            if pos == i:
                print("─", end="")
            else:
                print(" ", end="")
        else:
            print(" ", end="")
    print()


def matplotlib_visualization(populations: list[PopulationData], output_dir: Path):
    """Generate matplotlib visualizations."""

    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_size = (10, 6)

    # Prepare data
    by_freq = defaultdict(list)
    by_cat = defaultdict(list)
    for p in populations:
        by_freq[p.estimated_frequency].append(p)
        by_cat[p.category].append(p)

    # 1. Scatter plot with regression
    fig, ax = plt.subplots(figsize=fig_size)

    x = [p.estimated_frequency + (hash(p.name) % 100) / 200 - 0.25 for p in populations]  # Jitter
    y = [p.detection_rate for p in populations]
    colors = [plt.cm.Set2(hash(p.category) % 8) for p in populations]

    ax.scatter(x, y, c=colors, alpha=0.6, s=50)

    # Regression line
    x_clean = [p.estimated_frequency for p in populations]
    r, slope, intercept = compute_correlation(x_clean, y)
    x_line = [0.5, 5.5]
    y_line = [slope * xi + intercept for xi in x_line]
    ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'R² = {r**2:.3f}')

    ax.set_xlabel('Estimated Frequency Level (1=Rare, 5=Very Common)', fontsize=12)
    ax.set_ylabel('Detection Rate', fontsize=12)
    ax.set_title('Frequency Confound Test: PubMed Frequency vs Model Detection Rate', fontsize=14)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_scatter.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'frequency_scatter.png'}")

    # 2. Box plot by frequency level
    fig, ax = plt.subplots(figsize=fig_size)

    box_data = [
        [p.detection_rate for p in by_freq.get(level, [])]
        for level in range(1, 6)
    ]
    labels = [f'Rare\n(n={len(by_freq.get(1, []))})',
              f'Uncommon\n(n={len(by_freq.get(2, []))})',
              f'Moderate\n(n={len(by_freq.get(3, []))})',
              f'Common\n(n={len(by_freq.get(4, []))})',
              f'Very Common\n(n={len(by_freq.get(5, []))})']

    bp = ax.boxplot([d if d else [0] for d in box_data], labels=labels, patch_artist=True)
    colors = plt.cm.RdYlGn([0.2, 0.35, 0.5, 0.65, 0.8])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Detection Rate', fontsize=12)
    ax.set_title('Detection Rate Distribution by Frequency Level', fontsize=14)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_boxplot.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'frequency_boxplot.png'}")

    # 3. Category bar chart
    fig, ax = plt.subplots(figsize=fig_size)

    categories = sorted(by_cat.keys())
    avg_rates = [sum(p.detection_rate for p in by_cat[c]) / len(by_cat[c]) for c in categories]
    counts = [len(by_cat[c]) for c in categories]
    colors = plt.cm.Set2(range(len(categories)))

    bars = ax.bar(categories, avg_rates, color=colors)
    ax.set_ylabel('Average Detection Rate', fontsize=12)
    ax.set_title('Detection Rate by Population Category', fontsize=14)
    ax.set_ylim(0, max(avg_rates) * 1.2)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'category_barplot.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'category_barplot.png'}")

    # 4. Top/Bottom comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_pops = sorted(populations, key=lambda p: p.detection_rate, reverse=True)
    top10 = sorted_pops[:10]
    bottom10 = [p for p in sorted_pops if p.detection_rate > 0][-10:]

    all_pops = top10 + bottom10
    names = [p.name[:25] for p in all_pops]
    rates = [p.detection_rate for p in all_pops]
    freq_colors = [plt.cm.RdYlGn((p.estimated_frequency - 1) / 4) for p in all_pops]

    bars = ax.barh(range(len(all_pops)), rates, color=freq_colors)
    ax.set_yticks(range(len(all_pops)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Detection Rate', fontsize=12)
    ax.set_title('Top 10 vs Bottom 10 Populations (colored by frequency)', fontsize=14)
    ax.axhline(y=9.5, color='black', linestyle='--', linewidth=2)
    ax.text(0.5, 9.5, '← Top 10 | Bottom 10 →', ha='center', va='center',
            transform=ax.get_yaxis_transform(), fontsize=10, fontweight='bold')

    # Legend
    legend_elements = [mpatches.Patch(facecolor=plt.cm.RdYlGn(i/4), label=lbl)
                       for i, lbl in enumerate(['Rare', 'Uncommon', 'Moderate', 'Common', 'V.Common'])]
    ax.legend(handles=legend_elements, loc='lower right', title='Frequency')

    plt.tight_layout()
    plt.savefig(output_dir / 'top_bottom_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'top_bottom_comparison.png'}")


def main():
    # Find results
    results_dir = Path(__file__).parent / "results"
    result_files = sorted(results_dir.glob("experiment_results_*.json"), reverse=True)

    if not result_files:
        result_files = sorted(results_dir.glob("benchmark_results_*.json"), reverse=True)

    if not result_files:
        print("No result files found!")
        return

    results_path = result_files[0]
    print(f"Analyzing: {results_path.name}")

    # Load data
    populations = load_and_process_results(str(results_path))
    print(f"Loaded {len(populations)} populations")

    # Always show text visualization
    text_visualization(populations)

    # Generate matplotlib visualizations if available
    if HAS_MATPLOTLIB:
        output_dir = Path(__file__).parent / "results" / "visualizations"
        output_dir.mkdir(exist_ok=True)
        print(f"\nGenerating matplotlib visualizations to: {output_dir}")
        matplotlib_visualization(populations, output_dir)
    else:
        print("\nNote: Install matplotlib for graphical visualizations:")
        print("  pip install matplotlib")


if __name__ == "__main__":
    main()
