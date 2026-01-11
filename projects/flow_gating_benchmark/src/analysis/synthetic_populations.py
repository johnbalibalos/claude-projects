#!/usr/bin/env python3
"""
Synthetic Population Injection Experiment

Tests whether models can reason about novel populations with zero training frequency.
Creates fictitious populations with pronounceable names but valid marker logic.

Usage:
    python -m src.analysis.synthetic_populations --model gemini-2.0-flash --dry-run
    python -m src.analysis.synthetic_populations --model gemini-2.0-flash --run
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class SyntheticPopulation:
    """A synthetic (fake) cell population for testing."""
    name: str                    # Fake name (e.g., "Bafek cells")
    markers: str                 # Marker logic (e.g., "CD3+ CD4+")
    expected_parent: str         # Where it should appear in hierarchy
    complexity: str              # trivial, simple, medium, hard
    real_equivalent: str         # Real population with same markers
    marker_list: list[str] = field(default_factory=list)  # Parsed markers


@dataclass
class SyntheticTestResult:
    """Result of testing a synthetic population."""
    population: SyntheticPopulation
    model: str
    predicted_parent: str
    predicted_markers: str
    parent_correct: bool
    markers_correct: bool
    f1_score: float
    raw_response: str
    latency_seconds: float


def generate_pronounceable_name(seed: int = None) -> str:
    """Generate a pronounceable nonsense name for a cell population."""
    if seed is not None:
        random.seed(seed)

    # Consonant-vowel patterns for pronounceability
    consonants = 'bdfgklmnprstvz'  # Easy to pronounce
    vowels = 'aeiou'

    # Generate CVCVC pattern (5 letters)
    name = ''
    name += random.choice(consonants).upper()
    name += random.choice(vowels)
    name += random.choice(consonants)
    name += random.choice(vowels)
    name += random.choice(consonants)

    return f"{name} cells"


def generate_hash_name(seed: int) -> str:
    """Generate a hash-based name (less memorable but guaranteed unique)."""
    h = hashlib.md5(str(seed).encode()).hexdigest()[:5].upper()
    return f"{h} cells"


# Define synthetic populations at varying complexity levels
SYNTHETIC_POPULATIONS = [
    # TRIVIAL: Single marker
    SyntheticPopulation(
        name=generate_pronounceable_name(1),
        markers="CD3+",
        expected_parent="Lymphocytes",
        complexity="trivial",
        real_equivalent="T cells",
        marker_list=["CD3"],
    ),
    SyntheticPopulation(
        name=generate_pronounceable_name(2),
        markers="CD19+",
        expected_parent="Lymphocytes",
        complexity="trivial",
        real_equivalent="B cells",
        marker_list=["CD19"],
    ),

    # SIMPLE: Two markers (AND logic)
    SyntheticPopulation(
        name=generate_pronounceable_name(3),
        markers="CD3+ CD4+",
        expected_parent="T cells",
        complexity="simple",
        real_equivalent="CD4 T cells",
        marker_list=["CD3", "CD4"],
    ),
    SyntheticPopulation(
        name=generate_pronounceable_name(4),
        markers="CD3+ CD8+",
        expected_parent="T cells",
        complexity="simple",
        real_equivalent="CD8 T cells",
        marker_list=["CD3", "CD8"],
    ),
    SyntheticPopulation(
        name=generate_pronounceable_name(5),
        markers="CD14+ CD16-",
        expected_parent="Monocytes",
        complexity="simple",
        real_equivalent="Classical monocytes",
        marker_list=["CD14", "CD16"],
    ),

    # MEDIUM: Three markers
    SyntheticPopulation(
        name=generate_pronounceable_name(6),
        markers="CD3+ CD4+ CD45RA+",
        expected_parent="CD4 T cells",
        complexity="medium",
        real_equivalent="Naive CD4 T cells",
        marker_list=["CD3", "CD4", "CD45RA"],
    ),
    SyntheticPopulation(
        name=generate_pronounceable_name(7),
        markers="CD3+ CD8+ CD45RO+",
        expected_parent="CD8 T cells",
        complexity="medium",
        real_equivalent="Memory CD8 T cells",
        marker_list=["CD3", "CD8", "CD45RO"],
    ),
    SyntheticPopulation(
        name=generate_pronounceable_name(8),
        markers="CD3- CD56+ CD16+",
        expected_parent="Lymphocytes",
        complexity="medium",
        real_equivalent="NK cells",
        marker_list=["CD3", "CD56", "CD16"],
    ),

    # HARD: Complex marker combinations
    SyntheticPopulation(
        name=generate_pronounceable_name(9),
        markers="CD3+ CD4+ CD25+ CD127low",
        expected_parent="CD4 T cells",
        complexity="hard",
        real_equivalent="Regulatory T cells (Tregs)",
        marker_list=["CD3", "CD4", "CD25", "CD127"],
    ),
    SyntheticPopulation(
        name=generate_pronounceable_name(10),
        markers="CD14+ CD16+ SLAN+",
        expected_parent="Monocytes",
        complexity="hard",
        real_equivalent="SLAN+ non-classical monocytes",
        marker_list=["CD14", "CD16", "SLAN"],
    ),
    SyntheticPopulation(
        name=generate_pronounceable_name(11),
        markers="CD19+ CD27+ CD38high",
        expected_parent="B cells",
        complexity="hard",
        real_equivalent="Plasmablasts",
        marker_list=["CD19", "CD27", "CD38"],
    ),
    SyntheticPopulation(
        name=generate_pronounceable_name(12),
        markers="CD3+ CD4- CD8- TCRgd+",
        expected_parent="T cells",
        complexity="hard",
        real_equivalent="Gamma-delta T cells",
        marker_list=["CD3", "CD4", "CD8", "TCRgd"],
    ),
]


# Base panel that includes all markers needed for synthetic populations
BASE_PANEL = [
    {"marker": "CD3", "fluorophore": "BUV395", "purpose": "T cells"},
    {"marker": "CD4", "fluorophore": "BUV496", "purpose": "T helper cells"},
    {"marker": "CD8", "fluorophore": "BUV661", "purpose": "Cytotoxic T cells"},
    {"marker": "CD19", "fluorophore": "BV421", "purpose": "B cells"},
    {"marker": "CD14", "fluorophore": "BV510", "purpose": "Monocytes"},
    {"marker": "CD16", "fluorophore": "BV605", "purpose": "NK cells, monocyte subsets"},
    {"marker": "CD56", "fluorophore": "BV650", "purpose": "NK cells"},
    {"marker": "CD45RA", "fluorophore": "BV711", "purpose": "Naive T cells"},
    {"marker": "CD45RO", "fluorophore": "BV785", "purpose": "Memory T cells"},
    {"marker": "CD25", "fluorophore": "PE", "purpose": "Activation, Tregs"},
    {"marker": "CD127", "fluorophore": "PE-CF594", "purpose": "IL-7R, Treg exclusion"},
    {"marker": "CD27", "fluorophore": "PE-Cy5", "purpose": "Memory B cells"},
    {"marker": "CD38", "fluorophore": "PE-Cy7", "purpose": "Plasma cells"},
    {"marker": "SLAN", "fluorophore": "FITC", "purpose": "Non-classical monocytes"},
    {"marker": "TCRgd", "fluorophore": "APC", "purpose": "Gamma-delta T cells"},
    {"marker": "HLA-DR", "fluorophore": "APC-Cy7", "purpose": "Activation, APCs"},
    {"marker": "Zombie NIR", "fluorophore": "NIR", "purpose": "Viability"},
]


def format_panel(panel: list[dict]) -> str:
    """Format panel for prompt."""
    lines = ["| Marker | Fluorophore | Purpose |", "|--------|-------------|---------|"]
    for entry in panel:
        lines.append(f"| {entry['marker']} | {entry['fluorophore']} | {entry['purpose']} |")
    return "\n".join(lines)


def build_synthetic_prompt(population: SyntheticPopulation) -> str:
    """Build prompt for synthetic population test."""
    return f"""You are an expert flow cytometrist. Given this panel:

{format_panel(BASE_PANEL)}

A researcher has defined a novel population called "{population.name}" with the following marker expression:
**{population.markers}**

Task: Determine where "{population.name}" would appear in a standard gating hierarchy.

Respond with JSON only:
```json
{{
  "population_name": "{population.name}",
  "predicted_parent": "<parent gate name>",
  "marker_logic": "{population.markers}",
  "reasoning": "<brief explanation>"
}}
```
"""


def call_model(prompt: str, model: str, config: dict) -> tuple[str, float]:
    """Call the specified model and return response + latency."""
    start = time.time()

    if model.startswith("gemini"):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "[ERROR] GOOGLE_API_KEY not set", 0.0

        client = genai.Client(api_key=api_key)
        gen_config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1024,
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )
            latency = time.time() - start
            return response.text or "[EMPTY]", latency
        except Exception as e:
            latency = time.time() - start
            return f"[ERROR] {e}", latency

    elif model.startswith("claude"):
        import subprocess
        cli_model = "sonnet" if "sonnet" in model else "opus"

        try:
            result = subprocess.run(
                ["claude", "-p", "--model", cli_model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )
            latency = time.time() - start
            if result.returncode != 0:
                return f"[ERROR] {result.stderr}", latency
            return result.stdout, latency
        except Exception as e:
            latency = time.time() - start
            return f"[ERROR] {e}", latency

    else:
        return f"[ERROR] Unknown model: {model}", 0.0


def parse_response(response: str, population: SyntheticPopulation) -> tuple[str, str]:
    """Parse model response to extract predicted parent and markers."""
    # Try to find JSON
    json_match = re.search(r'```json\s*([\s\S]*?)```', response)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data.get("predicted_parent", ""), data.get("marker_logic", "")
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    try:
        data = json.loads(response)
        return data.get("predicted_parent", ""), data.get("marker_logic", "")
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    parent_match = re.search(r'parent["\s:]+([^"]+)', response, re.IGNORECASE)
    parent = parent_match.group(1).strip() if parent_match else ""

    return parent, population.markers


def check_parent_correct(predicted: str, expected: str) -> bool:
    """Check if predicted parent matches expected (fuzzy)."""
    pred = predicted.lower().replace("cells", "").strip()
    exp = expected.lower().replace("cells", "").strip()

    # Exact match
    if pred == exp:
        return True

    # Substring match
    if pred in exp or exp in pred:
        return True

    # Common synonyms
    synonyms = {
        "t cell": ["t cells", "t lymphocyte", "cd3+"],
        "b cell": ["b cells", "b lymphocyte", "cd19+"],
        "monocyte": ["monocytes", "cd14+"],
        "lymphocyte": ["lymphocytes", "lymph"],
    }

    for _key, values in synonyms.items():
        if pred in values and exp in values:
            return True

    return False


def run_synthetic_experiment(
    model: str,
    populations: list[SyntheticPopulation] = None,
    n_runs: int = 1,
) -> list[SyntheticTestResult]:
    """Run the synthetic population experiment."""
    if populations is None:
        populations = SYNTHETIC_POPULATIONS

    results = []
    config = {}

    print(f"\nRunning synthetic population experiment with {model}")
    print(f"Testing {len(populations)} populations × {n_runs} runs = {len(populations) * n_runs} trials")
    print("=" * 60)

    for pop in populations:
        print(f"\n{pop.name} ({pop.complexity}): {pop.markers}")
        print(f"  Expected parent: {pop.expected_parent}")
        print(f"  Real equivalent: {pop.real_equivalent}")

        for run in range(n_runs):
            prompt = build_synthetic_prompt(pop)
            response, latency = call_model(prompt, model, config)

            pred_parent, pred_markers = parse_response(response, pop)
            parent_correct = check_parent_correct(pred_parent, pop.expected_parent)

            result = SyntheticTestResult(
                population=pop,
                model=model,
                predicted_parent=pred_parent,
                predicted_markers=pred_markers,
                parent_correct=parent_correct,
                markers_correct=True,  # We provide the markers, so this is trivial
                f1_score=1.0 if parent_correct else 0.0,  # Simplified scoring
                raw_response=response,
                latency_seconds=latency,
            )
            results.append(result)

            status = "correct" if parent_correct else f"WRONG (got: {pred_parent})"
            print(f"  Run {run+1}: {status} ({latency:.1f}s)")

    return results


def plot_synthetic_results(results: list[SyntheticTestResult], output_dir: Path):
    """Generate visualizations for synthetic population experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by complexity
    complexities = ["trivial", "simple", "medium", "hard"]

    synth_scores = {c: [] for c in complexities}
    for r in results:
        synth_scores[r.population.complexity].append(r.f1_score)

    avg_scores = [np.mean(synth_scores[c]) if synth_scores[c] else 0 for c in complexities]
    std_scores = [np.std(synth_scores[c]) if synth_scores[c] else 0 for c in complexities]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bar chart by complexity
    x = np.arange(len(complexities))
    bars = axes[0].bar(x, avg_scores, yerr=std_scores, capsize=5, edgecolor='black', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.capitalize() for c in complexities])
    axes[0].set_ylabel('Accuracy (Parent Correct)')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title(f'Synthetic Population Performance by Complexity\n(Model: {results[0].model})')

    # Color by performance
    for bar, score in zip(bars, avg_scores):
        if score >= 0.8:
            bar.set_color('green')
        elif score >= 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Right: Individual population results
    pop_names = []
    pop_scores = []
    pop_colors = []
    complexity_colors = {"trivial": "lightgreen", "simple": "lightblue", "medium": "lightyellow", "hard": "lightcoral"}

    for pop in SYNTHETIC_POPULATIONS:
        pop_results = [r for r in results if r.population.name == pop.name]
        if pop_results:
            pop_names.append(pop.name.replace(" cells", ""))
            pop_scores.append(np.mean([r.f1_score for r in pop_results]))
            pop_colors.append(complexity_colors[pop.complexity])

    y_pos = np.arange(len(pop_names))
    bars = axes[1].barh(y_pos, pop_scores, color=pop_colors, edgecolor='black', alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(pop_names, fontsize=9)
    axes[1].set_xlabel('Accuracy')
    axes[1].set_xlim(0, 1.1)
    axes[1].set_title('Per-Population Results')

    # Legend for complexity
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label.capitalize())
                       for label, color in complexity_colors.items()]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'synthetic_results.png', dpi=150)
    print(f"\nSaved plot to {output_dir / 'synthetic_results.png'}")

    # Summary statistics
    overall_accuracy = np.mean([r.f1_score for r in results])
    by_complexity = {c: np.mean([r.f1_score for r in results if r.population.complexity == c])
                     for c in complexities}

    summary = {
        "model": results[0].model if results else "",
        "n_populations": len(SYNTHETIC_POPULATIONS),
        "n_trials": len(results),
        "overall_accuracy": overall_accuracy,
        "accuracy_by_complexity": by_complexity,
        "timestamp": datetime.now().isoformat(),
    }

    (output_dir / 'synthetic_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {output_dir / 'synthetic_summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Synthetic Population Injection Experiment")
    parser.add_argument("--model", default="gemini-2.0-flash",
                       help="Model to test (gemini-2.0-flash, gemini-2.5-flash, claude-sonnet)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per population")
    parser.add_argument("--output", type=Path, default=Path("results/synthetic"),
                       help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show populations without running")
    parser.add_argument("--run", action="store_true",
                       help="Actually run the experiment")
    args = parser.parse_args()

    print("=" * 60)
    print("SYNTHETIC POPULATION INJECTION EXPERIMENT")
    print("=" * 60)
    print("\nThis experiment tests whether LLMs can reason about novel populations")
    print("with zero training frequency but valid marker logic.\n")

    # Print populations
    print("Synthetic populations to test:")
    print("-" * 60)
    for pop in SYNTHETIC_POPULATIONS:
        print(f"  {pop.name:15} | {pop.complexity:7} | {pop.markers:25} | = {pop.real_equivalent}")

    if args.dry_run:
        print("\n[DRY RUN] Would test these populations with model:", args.model)
        print("Run with --run to execute the experiment.")
        return

    if not args.run:
        print("\nUse --dry-run to preview or --run to execute")
        return

    # Run experiment
    results = run_synthetic_experiment(
        model=args.model,
        populations=SYNTHETIC_POPULATIONS,
        n_runs=args.runs,
    )

    # Generate visualizations
    summary = plot_synthetic_results(results, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {summary['model']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%}")
    print("\nBy complexity:")
    for complexity, acc in summary['accuracy_by_complexity'].items():
        print(f"  {complexity:8}: {acc:.1%}")

    if summary['overall_accuracy'] > 0.8:
        print("\n>>> HIGH ACCURACY: Model can reason about novel populations!")
        print("    This suggests reasoning capability, not just retrieval.")
    elif summary['overall_accuracy'] > 0.5:
        print("\n>>> MODERATE ACCURACY: Model partially transfers marker logic")
        print("    Some retrieval dependence but not complete.")
    else:
        print("\n>>> LOW ACCURACY: Model struggles with novel populations")
        print("    This suggests heavy dependence on retrieval/training data.")


if __name__ == "__main__":
    main()
