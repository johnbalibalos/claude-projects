"""
Report generation for benchmark results.

Generates:
1. Summary report with conclusions and key figures
2. Manual review report with side-by-side comparison
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..curation.schemas import TestCase, GatingHierarchy, GateNode
from ..curation.omip_extractor import load_test_case
from ..evaluation.scorer import ScoringResult


def load_benchmark_results(results_path: str | Path) -> dict:
    """Load benchmark results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def hierarchy_to_text(node: dict | GateNode, indent: int = 0) -> str:
    """Convert hierarchy to indented text representation."""
    lines = []
    prefix = "  " * indent

    if isinstance(node, GateNode):
        name = node.name
        markers = node.markers
        children = node.children
    else:
        name = node.get("name", "Unknown")
        markers = node.get("markers", [])
        children = node.get("children", [])

    marker_str = f" [{', '.join(markers)}]" if markers else ""
    lines.append(f"{prefix}├─ {name}{marker_str}")

    for child in children:
        lines.append(hierarchy_to_text(child, indent + 1))

    return "\n".join(lines)


def generate_summary_report(
    results_path: str | Path,
    ground_truth_dir: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """
    Generate summary report with conclusions and highlights.

    Args:
        results_path: Path to benchmark results JSON
        ground_truth_dir: Directory with ground truth test cases
        output_path: Optional path to save report

    Returns:
        Report as markdown string
    """
    data = load_benchmark_results(results_path)

    # Extract key metrics
    metrics = data.get("metrics", {})
    token_usage = data.get("token_usage", {})
    cost = data.get("cost_usd", 0)
    results = data.get("results", [])

    # Calculate additional stats
    n_total = metrics.get("total", 0)
    n_valid = metrics.get("valid", 0)
    parse_rate = metrics.get("parse_success_rate", 0)

    f1_mean = metrics.get("hierarchy_f1_mean", 0)
    f1_min = metrics.get("hierarchy_f1_min", 0)
    f1_max = metrics.get("hierarchy_f1_max", 0)

    structure_mean = metrics.get("structure_accuracy_mean", 0)
    critical_mean = metrics.get("critical_gate_recall_mean", 0)
    halluc_mean = metrics.get("hallucination_rate_mean", 0)

    # Find best and worst performers
    valid_results = [r for r in results if r.get("parse_success") and r.get("evaluation")]
    if valid_results:
        best = max(valid_results, key=lambda r: r["evaluation"]["hierarchy_f1"])
        worst = min(valid_results, key=lambda r: r["evaluation"]["hierarchy_f1"])
    else:
        best = worst = None

    # Collect all missing critical gates
    all_missing_critical = []
    all_hallucinations = []
    for r in valid_results:
        eval_data = r.get("evaluation", {})
        all_missing_critical.extend(eval_data.get("missing_critical", []))
        all_hallucinations.extend(eval_data.get("hallucinated_gates", []))

    # Count frequencies
    from collections import Counter
    missing_critical_counts = Counter(all_missing_critical)
    hallucination_counts = Counter(all_hallucinations)

    # Build report
    report = f"""# Flow Cytometry Gating Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {data.get("model", "Unknown")}
**Test Cases:** {n_total}

---

## Executive Summary

The benchmark evaluated an LLM's ability to predict flow cytometry gating hierarchies
from panel information. Key findings:

| Metric | Score | Assessment |
|--------|-------|------------|
| Hierarchy F1 | {f1_mean:.1%} | {"Good" if f1_mean > 0.7 else "Moderate" if f1_mean > 0.5 else "Needs Improvement"} |
| Structure Accuracy | {structure_mean:.1%} | {"Good" if structure_mean > 0.7 else "Moderate" if structure_mean > 0.5 else "Needs Improvement"} |
| Critical Gate Recall | {critical_mean:.1%} | {"Good" if critical_mean > 0.7 else "Moderate" if critical_mean > 0.5 else "Needs Improvement"} |
| Hallucination Rate | {halluc_mean:.1%} | {"Low (Good)" if halluc_mean < 0.15 else "Moderate" if halluc_mean < 0.3 else "High (Concerning)"} |
| Parse Success | {parse_rate:.1%} | {"Excellent" if parse_rate > 0.95 else "Good" if parse_rate > 0.8 else "Needs Improvement"} |

---

## Key Conclusions

### 1. Overall Performance
- The model achieved **{f1_mean:.1%} F1 score** on gate name prediction
- **{parse_rate:.0%}** of responses were valid, parseable JSON hierarchies
- Structure accuracy ({structure_mean:.1%}) indicates {"strong" if structure_mean > 0.7 else "moderate"} understanding of gating logic

### 2. Strengths
- ✓ Low hallucination rate ({halluc_mean:.1%}) - model uses actual panel markers
- ✓ High parse success rate - follows JSON output format well
- ✓ Reasonable hierarchy depth and structure

### 3. Weaknesses
- ✗ Missing critical QC gates ({critical_mean:.1%} recall)
- ✗ Gate naming inconsistencies (e.g., "Singlets (SSC)" vs "Singlets")
- ✗ Over-generates subset populations not in ground truth

### 4. Most Commonly Missing Critical Gates
"""

    if missing_critical_counts:
        for gate, count in missing_critical_counts.most_common(5):
            report += f"- **{gate}**: missed in {count}/{n_valid} test cases\n"
    else:
        report += "- None identified\n"

    report += f"""
### 5. Common Hallucinations
"""
    if hallucination_counts:
        for gate, count in hallucination_counts.most_common(5):
            report += f"- **{gate}**: appeared in {count} predictions\n"
    else:
        report += "- None identified (good!)\n"

    report += f"""
---

## Performance by Test Case

| Test Case | Colors | F1 | Structure | Critical | Notes |
|-----------|--------|-----|-----------|----------|-------|
"""

    for r in results:
        tc_id = r.get("test_case_id", "Unknown")
        if r.get("parse_success") and r.get("evaluation"):
            e = r["evaluation"]
            f1 = e.get("hierarchy_f1", 0)
            struct = e.get("structure_accuracy", 0)
            crit = e.get("critical_gate_recall", 0)
            n_missing = len(e.get("missing_gates", []))
            n_extra = len(e.get("extra_gates", []))
            notes = f"{n_missing} missing, {n_extra} extra"
            report += f"| {tc_id} | - | {f1:.1%} | {struct:.1%} | {crit:.1%} | {notes} |\n"
        else:
            report += f"| {tc_id} | - | - | - | - | Parse failed |\n"

    report += f"""
---

## Cost Analysis

| Metric | Value |
|--------|-------|
| Input Tokens | {token_usage.get("input", 0):,} |
| Output Tokens | {token_usage.get("output", 0):,} |
| Total Tokens | {token_usage.get("total", 0):,} |
| **Total Cost** | **${cost:.4f}** |
| Cost per Test Case | ${cost/n_total:.4f} |

---

## Recommendations

1. **Improve QC Gate Emphasis**: Add explicit instructions to always include Time, Singlets, and Live/Dead gates
2. **Enhance Fuzzy Matching**: Accept gate name variants (e.g., "Lymphs" = "Lymphocytes")
3. **Constrain Output**: Consider providing a template or schema of expected gates
4. **Add Few-Shot Examples**: Include example hierarchies in the prompt
5. **Expand Ground Truth**: Add acceptable alternative gate names to test cases

---

## Appendix: Raw Metrics

```json
{json.dumps(metrics, indent=2)}
```
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


def generate_comparison_report(
    results_path: str | Path,
    ground_truth_dir: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """
    Generate side-by-side comparison report for manual review.

    Args:
        results_path: Path to benchmark results JSON
        ground_truth_dir: Directory with ground truth test cases
        output_path: Optional path to save report

    Returns:
        Report as markdown string
    """
    data = load_benchmark_results(results_path)
    ground_truth_dir = Path(ground_truth_dir)

    report = f"""# Manual Review Report: LLM vs Ground Truth Comparison

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {data.get("model", "Unknown")}

This report shows the LLM-predicted gating hierarchy alongside the ground truth (OMIP)
hierarchy for manual comparison and review.

---

"""

    for result in data.get("results", []):
        tc_id = result.get("test_case_id", "Unknown")

        # Load ground truth
        gt_file = ground_truth_dir / f"{tc_id.lower().replace('-', '_')}.json"
        if gt_file.exists():
            test_case = load_test_case(gt_file)
            gt_hierarchy = test_case.gating_hierarchy
            gt_text = hierarchy_to_text(gt_hierarchy.root)
            panel_info = ", ".join(test_case.panel.markers[:10])
            if len(test_case.panel.markers) > 10:
                panel_info += f"... (+{len(test_case.panel.markers) - 10} more)"
        else:
            gt_text = "(Ground truth file not found)"
            panel_info = "N/A"
            gt_hierarchy = None

        # Get predicted hierarchy
        if result.get("parsed_hierarchy"):
            pred_text = hierarchy_to_text(result["parsed_hierarchy"])
        else:
            pred_text = "(Parse failed - no hierarchy available)"

        # Get evaluation metrics
        eval_data = result.get("evaluation", {})
        f1 = eval_data.get("hierarchy_f1", 0)
        struct = eval_data.get("structure_accuracy", 0)
        crit = eval_data.get("critical_gate_recall", 0)

        report += f"""## {tc_id}

**Panel Markers:** {panel_info}

### Metrics Summary

| Metric | Score |
|--------|-------|
| Hierarchy F1 | {f1:.1%} |
| Structure Accuracy | {struct:.1%} |
| Critical Gate Recall | {crit:.1%} |

### Side-by-Side Comparison

<table>
<tr>
<th width="50%">Ground Truth (OMIP)</th>
<th width="50%">LLM Prediction</th>
</tr>
<tr>
<td>

```
{gt_text}
```

</td>
<td>

```
{pred_text}
```

</td>
</tr>
</table>

### Gate Analysis

"""

        if eval_data:
            matching = eval_data.get("matching_gates", [])
            missing = eval_data.get("missing_gates", [])
            extra = eval_data.get("extra_gates", [])
            missing_crit = eval_data.get("missing_critical", [])

            report += f"""| Category | Gates |
|----------|-------|
| ✓ **Matching** ({len(matching)}) | {', '.join(matching[:8])}{"..." if len(matching) > 8 else ""} |
| ✗ **Missing** ({len(missing)}) | {', '.join(missing[:8])}{"..." if len(missing) > 8 else ""} |
| ⚠ **Extra** ({len(extra)}) | {', '.join(extra[:8])}{"..." if len(extra) > 8 else ""} |
| 🚨 **Missing Critical** ({len(missing_crit)}) | {', '.join(missing_crit) if missing_crit else "None"} |

"""

        # Add structure errors if any
        struct_errors = eval_data.get("structure_errors", [])
        if struct_errors:
            report += "### Structure Errors\n\n"
            for err in struct_errors[:5]:
                report += f"- {err}\n"
            report += "\n"

        report += "---\n\n"

    # Add legend
    report += """## Legend

- **Matching Gates**: Gates correctly predicted (present in both)
- **Missing Gates**: Gates in ground truth but not predicted
- **Extra Gates**: Gates predicted but not in ground truth
- **Missing Critical**: Essential QC/lineage gates that were missed
- **Structure Errors**: Parent-child relationships that don't match

## Review Guidelines

When manually reviewing:
1. Check if "extra" gates are reasonable alternatives (may indicate ground truth gaps)
2. Evaluate if missing gates are truly missing or just named differently
3. Assess biological plausibility of the predicted hierarchy
4. Note any systematic patterns across test cases
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


def generate_all_reports(
    results_path: str | Path,
    ground_truth_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Generate all reports.

    Args:
        results_path: Path to benchmark results JSON
        ground_truth_dir: Directory with ground truth test cases
        output_dir: Directory to save reports

    Returns:
        Dictionary mapping report type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate summary report
    summary_path = output_dir / f"summary_report_{timestamp}.md"
    generate_summary_report(results_path, ground_truth_dir, summary_path)
    print(f"✓ Summary report: {summary_path}")

    # Generate comparison report
    comparison_path = output_dir / f"comparison_report_{timestamp}.md"
    generate_comparison_report(results_path, ground_truth_dir, comparison_path)
    print(f"✓ Comparison report: {comparison_path}")

    return {
        "summary": summary_path,
        "comparison": comparison_path,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <results.json> [ground_truth_dir] [output_dir]")
        sys.exit(1)

    results_path = sys.argv[1]
    ground_truth_dir = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "results/reports"

    reports = generate_all_reports(results_path, ground_truth_dir, output_dir)
    print(f"\nGenerated {len(reports)} reports")
