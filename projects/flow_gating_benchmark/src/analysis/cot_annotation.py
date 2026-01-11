#!/usr/bin/env python3
"""
Chain-of-Thought Trace Annotation

Analyzes CoT responses to understand why CoT hurts performance.
Automatically detects hallucinated markers and classifies error types.

Usage:
    python -m src.analysis.cot_annotation --results results/*/debug_results_*.json
"""

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CoTAnalysisResult:
    """Analysis of a single CoT trace."""
    condition_name: str
    test_case_id: str
    response_length: int
    word_count: int

    # Marker analysis
    panel_markers: list[str]
    mentioned_markers: list[str]
    hallucinated_markers: list[str]
    hallucination_rate: float

    # Error classification (can be multiple)
    error_types: list[str] = field(default_factory=list)
    error_explanations: list[str] = field(default_factory=list)

    # Performance
    f1_score: float = 0.0
    parse_success: bool = True


# Common marker patterns in flow cytometry
MARKER_PATTERNS = [
    r'\b(CD\d+[a-zA-Z]?)\b',              # CD3, CD4, CD45RA, CD45RO
    r'\b(HLA-?[A-Z]+\d*)\b',               # HLA-DR, HLADR
    r'\b(Fc[γεαg]R\w*)\b',                 # FcγRIII, FcgRIII
    r'\b(CCR\d+|CXCR\d+|CX3CR\d+)\b',      # Chemokine receptors
    r'\b(PD-?1|CTLA-?4|TIM-?3|LAG-?3)\b',  # Checkpoint molecules
    r'\b(Ki-?67|BCL-?\d|Bcl-?\d)\b',       # Proliferation/survival
    r'\b(IgG|IgM|IgA|IgD|IgE)\d*\b',       # Immunoglobulins
    r'\b(TCR[αβγδab])\b',                  # T cell receptors
    r'\b(NKG2[A-D]|NKp\d+)\b',             # NK receptors
    r'\b(FOXP3|FoxP3|Foxp3)\b',            # Transcription factors
    r'\b(Zombie|LIVE.?DEAD|7-?AAD|PI)\b',  # Viability dyes
    r'\b(FSC|SSC)[-_]?[AHW]?\b',           # Scatter parameters
]


def extract_markers(text: str) -> set[str]:
    """Extract all marker mentions from text."""
    markers = set()

    for pattern in MARKER_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize: uppercase, remove hyphens
            normalized = match.upper().replace("-", "")
            markers.add(normalized)

    return markers


def normalize_marker(marker: str) -> str:
    """Normalize marker name for comparison."""
    return marker.upper().replace("-", "").replace("_", "").strip()


def analyze_cot_trace(
    response: str,
    panel_markers: list[str],
    f1_score: float = 0.0,
) -> CoTAnalysisResult:
    """Analyze a CoT trace for errors."""
    # Normalize panel markers
    panel_set = {normalize_marker(m) for m in panel_markers}

    # Extract mentioned markers
    mentioned = extract_markers(response)

    # Find hallucinated markers (mentioned but not in panel)
    hallucinated = mentioned - panel_set

    # Calculate hallucination rate
    hall_rate = len(hallucinated) / len(mentioned) if mentioned else 0.0

    return CoTAnalysisResult(
        condition_name="",  # Set by caller
        test_case_id="",    # Set by caller
        response_length=len(response),
        word_count=len(response.split()),
        panel_markers=list(panel_set),
        mentioned_markers=list(mentioned),
        hallucinated_markers=list(hallucinated),
        hallucination_rate=hall_rate,
        f1_score=f1_score,
    )


def classify_errors_with_llm(
    response: str,
    panel_markers: list[str],
    hallucinated: list[str],
    model: str = "haiku",
) -> tuple[list[str], list[str]]:
    """Use Claude to classify error types in CoT trace."""

    prompt = f"""Analyze this Chain-of-Thought response for a flow cytometry gating prediction task.

PANEL MARKERS AVAILABLE: {', '.join(panel_markers[:30])}

HALLUCINATED MARKERS DETECTED: {', '.join(hallucinated) if hallucinated else 'None'}

RESPONSE (truncated):
{response[:3000]}

Classify errors into these categories (can have multiple or none):
1. HALLUCINATED_MARKER - References markers not in the panel (already detected above)
2. INCORRECT_BIOLOGY - Makes factually wrong biological claims about cell types or markers
3. LOGIC_ERROR - Correct facts but wrong reasoning or conclusion about gating hierarchy
4. DISTRACTION - Goes off-topic, over-explains irrelevant details, or tangential content
5. NONE - No significant errors beyond any hallucinated markers

Return ONLY a JSON object (no markdown):
{{"error_types": ["TYPE1", "TYPE2"], "explanations": ["brief reason 1", "brief reason 2"]}}
"""

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return ["CLASSIFICATION_FAILED"], [result.stderr[:100]]

        # Parse JSON from response
        response_text = result.stdout.strip()
        # Try to extract JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("error_types", []), data.get("explanations", [])
        else:
            return ["PARSE_FAILED"], ["Could not parse LLM response"]

    except subprocess.TimeoutExpired:
        return ["TIMEOUT"], ["LLM classification timed out"]
    except json.JSONDecodeError as e:
        return ["JSON_ERROR"], [str(e)]
    except Exception as e:
        return ["ERROR"], [str(e)]


def analyze_results_file(
    results_file: Path,
    use_llm: bool = False,
    llm_model: str = "haiku",
) -> list[CoTAnalysisResult]:
    """Analyze all CoT traces in a results file."""
    data = json.loads(results_file.read_text())
    analyses = []

    # We need access to raw responses - check if they're in the results
    results = data.get("results", [])

    for result in results:
        condition = result.get("condition_name", "")

        # Only analyze CoT conditions
        if "_cot_" not in condition.lower():
            continue

        # Get panel markers from test case
        # This requires access to ground truth - for now, extract from response
        # In practice, you'd load the test case

        # Check if we have trial-level data with raw responses
        trials = result.get("trials", [])
        if not trials:
            continue

        for trial in trials:
            response = trial.get("raw_response", "")
            if not response:
                continue

            # Extract panel from the prompt or response context
            # For now, use empty - would need to load test case
            panel_markers = []  # TODO: Load from test case

            analysis = analyze_cot_trace(
                response=response,
                panel_markers=panel_markers,
                f1_score=trial.get("hierarchy_f1", 0.0),
            )
            analysis.condition_name = condition
            analysis.test_case_id = result.get("test_case_id", "")

            # Optionally classify with LLM
            if use_llm and analysis.hallucinated_markers:
                error_types, explanations = classify_errors_with_llm(
                    response=response,
                    panel_markers=panel_markers,
                    hallucinated=analysis.hallucinated_markers,
                    model=llm_model,
                )
                analysis.error_types = error_types
                analysis.error_explanations = explanations

            analyses.append(analysis)

    return analyses


def plot_cot_analysis(analyses: list[CoTAnalysisResult], output_dir: Path):
    """Generate visualizations for CoT analysis."""
    if not analyses:
        print("No analyses to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hallucination rate distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Hallucination rate histogram
    hall_rates = [a.hallucination_rate for a in analyses]
    axes[0, 0].hist(hall_rates, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Hallucination Rate')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Hallucination Rate Distribution\n(mean={np.mean(hall_rates):.3f})')
    axes[0, 0].axvline(np.mean(hall_rates), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()

    # Hallucination rate vs F1
    f1s = [a.f1_score for a in analyses]
    axes[0, 1].scatter(hall_rates, f1s, alpha=0.6)
    axes[0, 1].set_xlabel('Hallucination Rate')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Hallucination Rate vs Performance')

    # Add regression line
    if len(hall_rates) > 2:
        z = np.polyfit(hall_rates, f1s, 1)
        x_line = np.linspace(min(hall_rates), max(hall_rates), 100)
        axes[0, 1].plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5)
        from scipy import stats
        r, p = stats.pearsonr(hall_rates, f1s)
        axes[0, 1].set_title(f'Hallucination Rate vs Performance\n(r={r:.3f}, p={p:.4f})')

    # Response length vs F1
    lengths = [a.word_count for a in analyses]
    axes[1, 0].scatter(lengths, f1s, alpha=0.6)
    axes[1, 0].set_xlabel('Response Length (words)')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Response Length vs Performance')

    # Error type distribution (if available)
    error_counts = {}
    for a in analyses:
        for et in a.error_types:
            error_counts[et] = error_counts.get(et, 0) + 1

    if error_counts:
        types = list(error_counts.keys())
        counts = list(error_counts.values())
        axes[1, 1].bar(types, counts, edgecolor='black')
        axes[1, 1].set_xlabel('Error Type')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error Type Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No error classification\n(run with --use-llm)',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Error Type Distribution')

    plt.tight_layout()
    plt.savefig(output_dir / 'cot_analysis.png', dpi=150)
    print(f"Saved plot to {output_dir / 'cot_analysis.png'}")

    # Save summary stats
    summary = {
        "n_traces": len(analyses),
        "mean_hallucination_rate": np.mean(hall_rates),
        "std_hallucination_rate": np.std(hall_rates),
        "mean_word_count": np.mean(lengths),
        "mean_f1": np.mean(f1s),
        "correlation_hall_f1": float(np.corrcoef(hall_rates, f1s)[0, 1]) if len(hall_rates) > 1 else 0,
        "error_type_counts": error_counts,
    }

    (output_dir / 'cot_analysis_summary.json').write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {output_dir / 'cot_analysis_summary.json'}")

    return summary


def print_hallucination_examples(analyses: list[CoTAnalysisResult], n: int = 5):
    """Print examples of hallucinated markers."""
    # Sort by hallucination rate
    sorted_analyses = sorted(analyses, key=lambda a: a.hallucination_rate, reverse=True)

    print("\n" + "=" * 60)
    print("TOP HALLUCINATION EXAMPLES")
    print("=" * 60)

    for i, a in enumerate(sorted_analyses[:n]):
        print(f"\n{i+1}. {a.condition_name} / {a.test_case_id}")
        print(f"   Hallucination rate: {a.hallucination_rate:.2%}")
        print(f"   Hallucinated markers: {', '.join(a.hallucinated_markers[:10])}")
        print(f"   F1 Score: {a.f1_score:.3f}")
        if a.error_types:
            print(f"   Error types: {', '.join(a.error_types)}")


def main():
    parser = argparse.ArgumentParser(description="CoT Trace Annotation Analysis")
    parser.add_argument("--results", type=Path, nargs="+",
                       help="Results JSON files to analyze")
    parser.add_argument("--output", type=Path, default=Path("results/cot_analysis"),
                       help="Output directory")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use Claude to classify error types")
    parser.add_argument("--llm-model", default="haiku",
                       help="Model for error classification (haiku, sonnet)")
    parser.add_argument("--sample", type=int, default=0,
                       help="Random sample N traces (0 = all)")
    args = parser.parse_args()

    all_analyses = []

    if args.results:
        for results_file in args.results:
            if results_file.exists():
                print(f"Analyzing {results_file}...")
                analyses = analyze_results_file(
                    results_file,
                    use_llm=args.use_llm,
                    llm_model=args.llm_model,
                )
                all_analyses.extend(analyses)
                print(f"  Found {len(analyses)} CoT traces")

    if not all_analyses:
        print("No CoT traces found. Make sure results files contain trial-level data.")
        print("\nRunning demo analysis on sample text...")

        # Demo with sample responses
        demo_response = """
        Let me think through this step by step.

        First, I'll gate on singlets using FSC-A vs FSC-H.
        Then I'll identify live cells using the Zombie NIR viability dye.

        For T cells, I'll look for CD3+ cells. CD3 is the canonical T cell marker.
        Within T cells, CD4 and CD8 define helper and cytotoxic T cells.

        For B cells, CD19+ CD20+ would identify the B cell population.
        Note that CD21 is also expressed on follicular dendritic cells.

        I should also check for NK cells using CD56 and CD16.
        SLAN expression on monocytes indicates non-classical monocytes.
        """

        demo_panel = ["CD3", "CD4", "CD8", "CD19", "Zombie NIR"]

        analysis = analyze_cot_trace(demo_response, demo_panel)
        print("\nDemo analysis:")
        print(f"  Panel: {demo_panel}")
        print(f"  Mentioned: {analysis.mentioned_markers}")
        print(f"  Hallucinated: {analysis.hallucinated_markers}")
        print(f"  Hallucination rate: {analysis.hallucination_rate:.2%}")
        return

    # Sample if requested
    if args.sample > 0 and len(all_analyses) > args.sample:
        import random
        all_analyses = random.sample(all_analyses, args.sample)

    # Generate visualizations
    summary = plot_cot_analysis(all_analyses, args.output)

    # Print examples
    print_hallucination_examples(all_analyses)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total traces analyzed: {summary['n_traces']}")
    print(f"Mean hallucination rate: {summary['mean_hallucination_rate']:.2%}")
    print(f"Correlation (hallucination vs F1): {summary['correlation_hall_f1']:.3f}")

    if summary['correlation_hall_f1'] < -0.3:
        print("\n>>> NEGATIVE CORRELATION: Hallucinations hurt performance")
    elif summary['correlation_hall_f1'] < 0:
        print("\n>>> WEAK NEGATIVE: Hallucinations slightly hurt performance")
    else:
        print("\n>>> NO CLEAR RELATIONSHIP: Other factors may dominate")


if __name__ == "__main__":
    main()
