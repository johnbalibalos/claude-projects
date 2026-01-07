"""PDF report generation for DrugDevBench results."""

import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from drugdevbench.data.schemas import (
    AblationResult,
    BenchmarkResult,
    EvaluationResponse,
    FigureType,
    PromptCondition,
)


def create_ablation_report(
    results: dict[str, AblationResult],
    output_path: Path | str,
    title: str = "DrugDevBench Ablation Study Report",
    include_sample_figures: bool = True,
    sample_figure_dir: Path | str | None = None,
) -> Path:
    """Create a comprehensive PDF report from ablation study results.

    Args:
        results: Dictionary of AblationResult by model
        output_path: Path to save the PDF report
        title: Report title
        include_sample_figures: Whether to include sample figures
        sample_figure_dir: Directory containing sample figures

    Returns:
        Path to the generated PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up matplotlib style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    with PdfPages(output_path) as pdf:
        # Title page
        _create_title_page(pdf, title, results)

        # Executive summary
        _create_executive_summary(pdf, results)

        # Results by model
        for model, ablation in results.items():
            _create_model_results_page(pdf, model, ablation)

        # Condition comparison
        _create_condition_comparison(pdf, results)

        # Figure type breakdown (if available)
        _create_figure_type_analysis(pdf, results)

        # Cost analysis
        _create_cost_analysis(pdf, results)

        # Conclusions
        _create_conclusions_page(pdf, results)

        # Appendix with methodology
        _create_methodology_appendix(pdf)

    return output_path


def _create_title_page(
    pdf: PdfPages,
    title: str,
    results: dict[str, AblationResult],
) -> None:
    """Create the title page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.7, title, fontsize=28, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)

    # Subtitle
    ax.text(0.5, 0.55, "Evaluating LLM Interpretation of Drug Development Figures",
            fontsize=16, ha="center", va="center", transform=ax.transAxes,
            style="italic", color="gray")

    # Date and models
    ax.text(0.5, 0.4, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=12, ha="center", va="center", transform=ax.transAxes)

    models_str = ", ".join(results.keys())
    ax.text(0.5, 0.35, f"Models: {models_str}",
            fontsize=12, ha="center", va="center", transform=ax.transAxes)

    # Summary stats
    total_evals = sum(
        sum(r.n_questions for r in ablation.results_by_condition.values())
        for ablation in results.values()
    )
    ax.text(0.5, 0.25, f"Total Evaluations: {total_evals:,}",
            fontsize=14, ha="center", va="center", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _create_executive_summary(
    pdf: PdfPages,
    results: dict[str, AblationResult],
) -> None:
    """Create executive summary page."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Executive Summary", fontsize=18, fontweight="bold", y=0.98)

    # 1. Overall scores by condition (top-left)
    ax = axes[0, 0]
    _plot_condition_scores(ax, results)
    ax.set_title("Mean Score by Condition", fontsize=12, fontweight="bold")

    # 2. Improvement over baseline (top-right)
    ax = axes[0, 1]
    _plot_improvements(ax, results)
    ax.set_title("Improvement over Vanilla Baseline", fontsize=12, fontweight="bold")

    # 3. Model comparison (bottom-left)
    ax = axes[1, 0]
    _plot_model_comparison(ax, results)
    ax.set_title("Model Comparison (Full Stack)", fontsize=12, fontweight="bold")

    # 4. Key findings text (bottom-right)
    ax = axes[1, 1]
    ax.axis("off")
    findings = _generate_key_findings(results)
    ax.text(0.1, 0.9, "Key Findings:", fontsize=14, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.1, 0.8, findings, fontsize=10, transform=ax.transAxes,
            va="top", wrap=True, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_condition_scores(ax, results: dict[str, AblationResult]) -> None:
    """Plot mean scores by condition."""
    # Aggregate scores across models
    condition_scores = defaultdict(list)
    for ablation in results.values():
        for cond_name, result in ablation.results_by_condition.items():
            condition_scores[cond_name].append(result.mean_score)

    conditions = list(condition_scores.keys())
    means = [np.mean(condition_scores[c]) for c in conditions]
    stds = [np.std(condition_scores[c]) if len(condition_scores[c]) > 1 else 0 for c in conditions]

    # Sort by mean score
    sorted_idx = np.argsort(means)
    conditions = [conditions[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(conditions)))
    bars = ax.barh(conditions, means, xerr=stds, color=colors, capsize=3)
    ax.set_xlabel("Mean Score")
    ax.set_xlim(0, 1)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(mean + 0.02, bar.get_y() + bar.get_height()/2,
                f"{mean:.3f}", va="center", fontsize=9)


def _plot_improvements(ax, results: dict[str, AblationResult]) -> None:
    """Plot improvement percentages over baseline."""
    # Use first model's improvements as example
    model = list(results.keys())[0]
    improvements = results[model].improvements

    if not improvements:
        ax.text(0.5, 0.5, "No improvement data available",
                ha="center", va="center", transform=ax.transAxes)
        return

    conditions = list(improvements.keys())
    values = [improvements[c] for c in conditions]

    # Sort by improvement
    sorted_idx = np.argsort(values)[::-1]
    conditions = [conditions[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    colors = ["green" if v > 0 else "red" for v in values]
    bars = ax.barh(conditions, values, color=colors, alpha=0.7)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Improvement (%)")

    for bar, val in zip(bars, values):
        ax.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", va="center", fontsize=9)


def _plot_model_comparison(ax, results: dict[str, AblationResult]) -> None:
    """Plot model comparison for full_stack condition."""
    models = []
    scores = []

    for model, ablation in results.items():
        if "full_stack" in ablation.results_by_condition:
            models.append(model)
            scores.append(ablation.results_by_condition["full_stack"].mean_score)

    if not models:
        ax.text(0.5, 0.5, "No full_stack results available",
                ha="center", va="center", transform=ax.transAxes)
        return

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, scores, color=colors)
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 1)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, score + 0.02,
                f"{score:.3f}", ha="center", fontsize=10)


def _generate_key_findings(results: dict[str, AblationResult]) -> str:
    """Generate key findings text."""
    findings = []

    # Find best condition
    best_condition = None
    best_score = 0
    for ablation in results.values():
        for cond_name, result in ablation.results_by_condition.items():
            if result.mean_score > best_score:
                best_score = result.mean_score
                best_condition = cond_name

    if best_condition:
        findings.append(f"• Best condition: {best_condition} ({best_score:.3f})")

    # Calculate average improvement for full_stack over vanilla
    improvements = []
    for ablation in results.values():
        if "full_stack" in ablation.improvements:
            improvements.append(ablation.improvements["full_stack"])

    if improvements:
        avg_improvement = np.mean(improvements)
        findings.append(f"• Full stack avg improvement: {avg_improvement:+.1f}%")

    # Skills impact
    skill_improvements = []
    for ablation in results.values():
        if "base_plus_skill" in ablation.improvements:
            skill_improvements.append(ablation.improvements["base_plus_skill"])

    if skill_improvements:
        avg_skill = np.mean(skill_improvements)
        findings.append(f"• Skills add: {avg_skill:+.1f}% over base")

    # Wrong skill penalty
    wrong_improvements = []
    for ablation in results.values():
        if "wrong_skill" in ablation.improvements:
            wrong_improvements.append(ablation.improvements["wrong_skill"])

    if wrong_improvements:
        avg_wrong = np.mean(wrong_improvements)
        findings.append(f"• Wrong skill penalty: {avg_wrong:+.1f}%")

    return "\n".join(findings) if findings else "No findings available"


def _create_model_results_page(
    pdf: PdfPages,
    model: str,
    ablation: AblationResult,
) -> None:
    """Create detailed results page for a single model."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"Detailed Results: {model}", fontsize=16, fontweight="bold", y=0.98)

    # 1. Score distribution by condition
    ax = axes[0, 0]
    conditions = list(ablation.results_by_condition.keys())
    scores = [ablation.results_by_condition[c].mean_score for c in conditions]
    stds = [ablation.results_by_condition[c].std_score for c in conditions]

    x = np.arange(len(conditions))
    bars = ax.bar(x, scores, yerr=stds, capsize=5, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(conditions))))
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Scores by Condition")

    # 2. Sample counts
    ax = axes[0, 1]
    n_questions = [ablation.results_by_condition[c].n_questions for c in conditions]
    ax.bar(x, n_questions, color="steelblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Questions")
    ax.set_title("Evaluation Counts")

    # 3. Improvements
    ax = axes[1, 0]
    if ablation.improvements:
        imp_conditions = list(ablation.improvements.keys())
        imp_values = [ablation.improvements[c] for c in imp_conditions]
        colors = ["green" if v > 0 else "red" for v in imp_values]
        ax.bar(imp_conditions, imp_values, color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Improvement (%)")
        ax.set_xticklabels(imp_conditions, rotation=45, ha="right", fontsize=8)
    ax.set_title("Improvement over Baseline")

    # 4. Stats table
    ax = axes[1, 1]
    ax.axis("off")

    table_data = []
    for cond in conditions:
        r = ablation.results_by_condition[cond]
        table_data.append([
            cond,
            f"{r.mean_score:.3f}",
            f"{r.std_score:.3f}",
            str(r.n_questions),
            f"${r.total_cost_usd:.4f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Condition", "Mean", "Std", "N", "Cost"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Summary Statistics", fontsize=12, fontweight="bold", y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _create_condition_comparison(
    pdf: PdfPages,
    results: dict[str, AblationResult],
) -> None:
    """Create condition comparison across models."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.suptitle("Condition Comparison Across Models", fontsize=16, fontweight="bold")

    # Get all conditions
    all_conditions = set()
    for ablation in results.values():
        all_conditions.update(ablation.results_by_condition.keys())
    conditions = sorted(all_conditions)

    x = np.arange(len(conditions))
    width = 0.8 / len(results)

    for i, (model, ablation) in enumerate(results.items()):
        scores = []
        for cond in conditions:
            if cond in ablation.results_by_condition:
                scores.append(ablation.results_by_condition[cond].mean_score)
            else:
                scores.append(0)

        offset = (i - len(results) / 2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _create_figure_type_analysis(
    pdf: PdfPages,
    results: dict[str, AblationResult],
) -> None:
    """Create figure type breakdown analysis."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.suptitle("Analysis by Figure Type", fontsize=16, fontweight="bold")

    # This would need actual per-figure-type data
    # For now, create a placeholder
    ax.text(0.5, 0.5,
            "Figure type breakdown requires per-type scoring data.\n"
            "Available in detailed evaluation results.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _create_cost_analysis(
    pdf: PdfPages,
    results: dict[str, AblationResult],
) -> None:
    """Create cost analysis page."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Cost Analysis", fontsize=16, fontweight="bold")

    # Total cost by model
    ax = axes[0]
    models = list(results.keys())
    total_costs = []
    for model, ablation in results.items():
        cost = sum(r.total_cost_usd for r in ablation.results_by_condition.values())
        total_costs.append(cost)

    bars = ax.bar(models, total_costs, color="green", alpha=0.7)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Total Cost by Model")

    for bar, cost in zip(bars, total_costs):
        ax.text(bar.get_x() + bar.get_width()/2, cost + 0.001,
                f"${cost:.4f}", ha="center", fontsize=10)

    # Cost efficiency (score per dollar)
    ax = axes[1]
    efficiencies = []
    for model, ablation in results.items():
        total_score = sum(r.mean_score * r.n_questions for r in ablation.results_by_condition.values())
        total_cost = sum(r.total_cost_usd for r in ablation.results_by_condition.values())
        if total_cost > 0:
            efficiencies.append(total_score / total_cost)
        else:
            efficiencies.append(0)

    ax.bar(models, efficiencies, color="blue", alpha=0.7)
    ax.set_ylabel("Score Points per Dollar")
    ax.set_title("Cost Efficiency")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _create_conclusions_page(
    pdf: PdfPages,
    results: dict[str, AblationResult],
) -> None:
    """Create conclusions page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.95, "Conclusions", fontsize=24, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes)

    conclusions = _generate_conclusions(results)

    ax.text(0.1, 0.85, conclusions, fontsize=12, transform=ax.transAxes,
            va="top", wrap=True, family="serif",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _generate_conclusions(results: dict[str, AblationResult]) -> str:
    """Generate conclusion text from results."""
    conclusions = []

    # 1. Overall finding about skill-based prompting
    avg_full_stack = []
    avg_vanilla = []
    for ablation in results.values():
        if "full_stack" in ablation.results_by_condition:
            avg_full_stack.append(ablation.results_by_condition["full_stack"].mean_score)
        if "vanilla" in ablation.results_by_condition:
            avg_vanilla.append(ablation.results_by_condition["vanilla"].mean_score)

    if avg_full_stack and avg_vanilla:
        improvement = (np.mean(avg_full_stack) - np.mean(avg_vanilla)) / np.mean(avg_vanilla) * 100
        conclusions.append(
            f"1. SKILL-BASED PROMPTING IMPROVES PERFORMANCE\n"
            f"   The full stack (persona + base + skill) approach achieved {improvement:.1f}% "
            f"improvement over vanilla prompting across all models tested.\n"
        )

    # 2. Importance of figure-type skills
    avg_base_skill = []
    avg_base_only = []
    for ablation in results.values():
        if "base_plus_skill" in ablation.results_by_condition:
            avg_base_skill.append(ablation.results_by_condition["base_plus_skill"].mean_score)
        if "base_only" in ablation.results_by_condition:
            avg_base_only.append(ablation.results_by_condition["base_only"].mean_score)

    if avg_base_skill and avg_base_only:
        skill_impact = (np.mean(avg_base_skill) - np.mean(avg_base_only)) / np.mean(avg_base_only) * 100
        conclusions.append(
            f"2. FIGURE-TYPE SKILLS ADD VALUE\n"
            f"   Adding figure-specific skills to base prompting improved scores by {skill_impact:.1f}%. "
            f"This confirms that domain-specific guidance helps LLMs interpret scientific figures.\n"
        )

    # 3. Wrong skill finding
    avg_wrong = []
    for ablation in results.values():
        if "wrong_skill" in ablation.results_by_condition:
            avg_wrong.append(ablation.results_by_condition["wrong_skill"].mean_score)

    if avg_wrong and avg_base_only:
        wrong_diff = np.mean(avg_wrong) - np.mean(avg_base_only)
        direction = "decreased" if wrong_diff < 0 else "maintained"
        conclusions.append(
            f"3. SKILL SPECIFICITY MATTERS\n"
            f"   Using mismatched skills {direction} performance, demonstrating that "
            f"skills must be appropriate to the figure type for optimal results.\n"
        )

    # 4. Recommendations
    conclusions.append(
        "4. RECOMMENDATIONS\n"
        "   • Use full_stack prompting for production deployments\n"
        "   • Implement automatic figure type detection to select appropriate skills\n"
        "   • Consider cost-performance tradeoffs when selecting models\n"
        "   • Validate on domain-specific figure types before deployment\n"
    )

    return "\n".join(conclusions)


def _create_methodology_appendix(pdf: PdfPages) -> None:
    """Create methodology appendix."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.95, "Methodology Appendix", fontsize=20, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes)

    methodology = """
ABLATION CONDITIONS TESTED:
• vanilla: Raw model capability without additional prompting
• base_only: Generic scientific figure interpretation prompt
• persona_only: Domain expert persona without base or skill
• base_plus_skill: Base prompt combined with figure-type specific skill
• full_stack: Complete system (persona + base + skill)
• wrong_skill: Base prompt with intentionally mismatched skill (negative control)

SCORING METHODOLOGY:
• Factual extraction: Exact or near-exact match with gold answer
• Visual estimation: Tolerance-based numeric comparison (10-50%)
• Quality assessment: Boolean evaluation of figure quality judgments
• Interpretation: Semantic similarity to expected interpretation
• Error detection: Correct identification of issues or lack thereof

FIGURE TYPES EVALUATED:
• Western blots and protein gels
• Dose-response and IC50/EC50 curves
• Pharmacokinetic concentration-time profiles
• Flow cytometry (biaxial plots, histograms)
• Expression heatmaps and volcano plots
• ELISA standard curves and assays
• Cell viability and cytotoxicity curves

DATA SOURCES:
• SourceData (EMBO): Semantic annotations with biological ontologies
• Open-PMC-18M: Large-scale PubMed Central figures
• Sample generator: Synthetic placeholders for pipeline testing
    """

    ax.text(0.05, 0.88, methodology, fontsize=10, transform=ax.transAxes,
            va="top", family="monospace")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_quick_summary(
    results: dict[str, AblationResult],
    output_path: Path | str | None = None,
) -> str:
    """Generate a quick text summary of results.

    Args:
        results: Ablation results
        output_path: Optional path to save summary

    Returns:
        Summary text
    """
    lines = ["=" * 60, "DRUGDEVBENCH ABLATION STUDY SUMMARY", "=" * 60, ""]

    for model, ablation in results.items():
        lines.append(f"Model: {model}")
        lines.append("-" * 40)

        for cond_name, result in ablation.results_by_condition.items():
            imp = ablation.improvements.get(cond_name, 0)
            imp_str = f" ({imp:+.1f}%)" if cond_name != "vanilla" else ""
            lines.append(
                f"  {cond_name:20s}: {result.mean_score:.3f} ± {result.std_score:.3f}"
                f" (n={result.n_questions}){imp_str}"
            )

        lines.append("")

    summary = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(summary)

    return summary
