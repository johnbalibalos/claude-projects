#!/usr/bin/env python3
"""
Generate detailed per-OMIP reports for experiment runs.

Creates a folder structure:
    results/runs/{run_id}/
        ├── summary.md
        ├── OMIP-022/
        │   ├── report.md
        │   ├── prompt_minimal_direct.txt
        │   ├── response_minimal_direct.json
        │   └── ...
        ├── OMIP-074/
        │   └── ...
        └── ...

Usage:
    python scripts/generate_detailed_reports.py results/experiment_results_*.json
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from curation.schemas import TestCase
from experiments.prompts import build_prompt


def format_hierarchy_tree(node: dict, indent: int = 0) -> str:
    """Format hierarchy as ASCII tree."""
    if not node:
        return ""

    lines = []
    prefix = "  " * indent + "├─ " if indent > 0 else ""
    name = node.get("name", "Unknown")
    markers = node.get("markers", [])

    marker_str = f" [{', '.join(markers)}]" if markers else ""
    lines.append(f"{prefix}{name}{marker_str}")

    for child in node.get("children", []):
        lines.append(format_hierarchy_tree(child, indent + 1))

    return "\n".join(lines)


def generate_omip_report(
    omip_id: str,
    test_case: TestCase,
    results: list[dict],
    output_dir: Path,
) -> None:
    """Generate detailed report for a single OMIP."""

    omip_dir = output_dir / omip_id
    omip_dir.mkdir(parents=True, exist_ok=True)

    # Get panel markers
    markers = test_case.panel.markers

    # Generate report
    report_lines = [
        f"# {omip_id} Detailed Report",
        "",
        f"**Application:** {test_case.context.application}",
        f"**Species:** {test_case.context.species}",
        f"**Sample Type:** {test_case.context.sample_type}",
        "",
        "## Panel",
        "",
        f"**{len(markers)} Markers:** {', '.join(markers)}",
        "",
        "| Marker | Fluorophore | Clone |",
        "|--------|-------------|-------|",
    ]

    for entry in test_case.panel.entries:
        clone = entry.clone or "—"
        report_lines.append(f"| {entry.marker} | {entry.fluorophore} | {clone} |")

    report_lines.extend([
        "",
        "## Ground Truth Hierarchy",
        "",
        "```",
        format_hierarchy_tree(test_case.gating_hierarchy.root.model_dump()),
        "```",
        "",
        "---",
        "",
        "## Results by Condition",
        "",
    ])

    # Sort results by condition
    sorted_results = sorted(results, key=lambda r: r['condition'])

    for result in sorted_results:
        condition = result['condition'].split('_', 1)[1] if '_' in result['condition'] else result['condition']
        eval_data = result.get('evaluation', {})

        f1 = eval_data.get('hierarchy_f1', 0) * 100
        precision = eval_data.get('hierarchy_precision', 0) * 100
        recall = eval_data.get('hierarchy_recall', 0) * 100
        structure = eval_data.get('structure_accuracy', 0) * 100
        critical = eval_data.get('critical_gate_recall', 0) * 100
        halluc = eval_data.get('hallucination_rate', 0) * 100

        report_lines.extend([
            f"### {condition}",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Hierarchy F1 | {f1:.1f}% |",
            f"| Precision | {precision:.1f}% |",
            f"| Recall | {recall:.1f}% |",
            f"| Structure Accuracy | {structure:.1f}% |",
            f"| Critical Gate Recall | {critical:.1f}% |",
            f"| Hallucination Rate | {halluc:.1f}% |",
            "",
        ])

        # Side-by-side comparison
        predicted = result.get('parsed_hierarchy', {})
        if predicted:
            report_lines.extend([
                "<table>",
                "<tr>",
                '<th width="50%">Ground Truth</th>',
                '<th width="50%">Predicted</th>',
                "</tr>",
                "<tr>",
                "<td>",
                "",
                "```",
                format_hierarchy_tree(test_case.gating_hierarchy.root.model_dump()),
                "```",
                "",
                "</td>",
                "<td>",
                "",
                "```",
                format_hierarchy_tree(predicted),
                "```",
                "",
                "</td>",
                "</tr>",
                "</table>",
                "",
            ])

        # Gate analysis
        matching = eval_data.get('matching_gates', [])
        missing = eval_data.get('missing_gates', [])
        extra = eval_data.get('extra_gates', [])
        missing_critical = eval_data.get('missing_critical', [])

        report_lines.extend([
            "**Gate Analysis:**",
            "",
            f"- ✓ Matching ({len(matching)}): {', '.join(matching[:10])}{'...' if len(matching) > 10 else ''}",
            f"- ✗ Missing ({len(missing)}): {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}",
            f"- ⚠ Extra ({len(extra)}): {', '.join(extra[:10])}{'...' if len(extra) > 10 else ''}",
        ])

        if missing_critical:
            report_lines.append(f"- 🚨 Missing Critical: {', '.join(missing_critical)}")

        report_lines.extend(["", "---", ""])

        # Save prompt for this condition
        context_level = condition.split('_')[0]  # minimal, standard, rich
        prompt_strategy = condition.split('_')[1] if '_' in condition else 'direct'

        try:
            prompt = build_prompt(test_case, prompt_strategy, context_level)
            prompt_file = omip_dir / f"prompt_{condition}.txt"
            prompt_file.write_text(prompt)
        except Exception:
            pass  # Skip if prompt generation fails

    # Write report
    report_file = omip_dir / "report.md"
    report_file.write_text("\n".join(report_lines))

    print(f"  Generated {omip_id}/report.md")


def generate_summary(
    run_id: str,
    results_data: dict,
    ground_truths: dict[str, TestCase],
    output_dir: Path,
) -> None:
    """Generate run summary."""

    summary_lines = [
        f"# Experiment Run: {run_id}",
        "",
        f"**Config:** {results_data.get('config_name', 'unknown')}",
        f"**Started:** {results_data.get('start_time', 'unknown')}",
        f"**Ended:** {results_data.get('end_time', 'unknown')}",
        f"**Total Results:** {results_data.get('n_results', 0)}",
        f"**Errors:** {results_data.get('n_errors', 0)}",
        "",
        "## Results by Test Case",
        "",
        "| OMIP | Markers | Best Condition | Best F1 | Avg F1 |",
        "|------|---------|----------------|---------|--------|",
    ]

    # Group results by test case
    by_tc: dict[str, list[dict]] = {}
    for r in results_data.get('results', []):
        tc_id = r['test_case_id']
        if tc_id not in by_tc:
            by_tc[tc_id] = []
        by_tc[tc_id].append(r)

    for tc_id in sorted(by_tc.keys()):
        tc_results = by_tc[tc_id]
        tc = ground_truths.get(tc_id)

        n_markers = len(tc.panel.markers) if tc else 0

        # Find best condition
        best = max(tc_results, key=lambda r: r.get('evaluation', {}).get('hierarchy_f1', 0))
        best_cond = best['condition'].split('_', 1)[1] if '_' in best['condition'] else best['condition']
        best_f1 = best.get('evaluation', {}).get('hierarchy_f1', 0) * 100

        avg_f1 = sum(r.get('evaluation', {}).get('hierarchy_f1', 0) for r in tc_results) / len(tc_results) * 100

        summary_lines.append(f"| [{tc_id}](./{tc_id}/report.md) | {n_markers} | {best_cond} | {best_f1:.1f}% | {avg_f1:.1f}% |")

    summary_lines.extend([
        "",
        "## Overall Statistics",
        "",
    ])

    # Compute overall stats
    all_f1 = [r.get('evaluation', {}).get('hierarchy_f1', 0) for r in results_data.get('results', [])]
    if all_f1:
        summary_lines.extend([
            f"- **Mean F1:** {sum(all_f1)/len(all_f1)*100:.1f}%",
            f"- **Min F1:** {min(all_f1)*100:.1f}%",
            f"- **Max F1:** {max(all_f1)*100:.1f}%",
        ])

    summary_file = output_dir / "summary.md"
    summary_file.write_text("\n".join(summary_lines))
    print("Generated summary.md")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate detailed per-OMIP reports")
    parser.add_argument("results_file", type=Path, help="Experiment results JSON file")
    parser.add_argument("--ground-truth-dir", type=Path,
                       default=Path("data/verified"),
                       help="Directory with ground truth files")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("results/runs"),
                       help="Output directory for reports")

    args = parser.parse_args()

    # Load results
    with open(args.results_file) as f:
        results_data = json.load(f)

    # Create run ID from filename
    run_id = args.results_file.stem.replace("experiment_results_", "run_")

    # Load ground truths (handle validation errors gracefully)
    ground_truths = {}
    for gt_file in args.ground_truth_dir.glob("omip_*.json"):
        with open(gt_file) as f:
            data = json.load(f)

        # Fix null fluorophores
        for entry in data.get('panel', {}).get('entries', []):
            if entry.get('fluorophore') is None:
                entry['fluorophore'] = 'Unknown'

        try:
            tc = TestCase(**data)
            ground_truths[tc.omip_id] = tc
        except Exception as e:
            print(f"Warning: Could not load {gt_file.name}: {e}")

    print(f"Loaded {len(ground_truths)} ground truth files")

    # Create output directory
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating reports in {output_dir}/")

    # Group results by test case
    by_tc: dict[str, list[dict]] = {}
    for r in results_data.get('results', []):
        tc_id = r['test_case_id']
        if tc_id not in by_tc:
            by_tc[tc_id] = []
        by_tc[tc_id].append(r)

    # Generate per-OMIP reports
    for tc_id, tc_results in sorted(by_tc.items()):
        if tc_id in ground_truths:
            generate_omip_report(tc_id, ground_truths[tc_id], tc_results, output_dir)
        else:
            print(f"  Skipping {tc_id} - no ground truth found")

    # Generate summary
    generate_summary(run_id, results_data, ground_truths, output_dir)

    print(f"\nDone! Reports in {output_dir}/")


if __name__ == "__main__":
    main()
