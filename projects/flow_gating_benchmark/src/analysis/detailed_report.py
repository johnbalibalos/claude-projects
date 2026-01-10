"""
Detailed per-test report generator.

Outputs comprehensive comparison between LLM predictions and gold standard
after each test case, enabling real-time analysis during benchmark runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from curation.schemas import TestCase, GateNode
from evaluation.scorer import ScoringResult


def gate_to_tree_string(node: dict | GateNode, indent: int = 0, is_last: bool = True, prefix: str = "") -> list[str]:
    """Convert a gate node to tree-style string representation."""
    lines = []

    if isinstance(node, GateNode):
        name = node.name
        markers = node.markers
        children = node.children
    else:
        name = node.get("name", "Unknown")
        markers = node.get("markers", [])
        children = node.get("children", [])

    # Build connector
    connector = "└── " if is_last else "├── "

    # Format markers
    marker_str = f" [{', '.join(markers[:3])}{'...' if len(markers) > 3 else ''}]" if markers else ""

    lines.append(f"{prefix}{connector}{name}{marker_str}")

    # Update prefix for children
    child_prefix = prefix + ("    " if is_last else "│   ")

    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        lines.extend(gate_to_tree_string(child, indent + 1, is_last_child, child_prefix))

    return lines


def extract_all_gates(node: dict | GateNode, parent: str = None) -> list[tuple[str, str | None, list[str]]]:
    """Extract all gates as (name, parent, markers) tuples."""
    gates = []

    if isinstance(node, GateNode):
        name = node.name
        markers = node.markers
        children = node.children
    else:
        name = node.get("name", "Unknown")
        markers = node.get("markers", [])
        children = node.get("children", [])

    gates.append((name, parent, markers))

    for child in children:
        gates.extend(extract_all_gates(child, name))

    return gates


def generate_per_test_report(
    result: ScoringResult,
    test_case: TestCase,
    raw_response: str,
    output_dir: Path | str | None = None,
) -> str:
    """
    Generate detailed report for a single test case.

    Args:
        result: Scoring result from the test
        test_case: Original test case with ground truth
        raw_response: Raw LLM response text
        output_dir: Optional directory to save report

    Returns:
        Report as string
    """
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append(f"DETAILED TEST REPORT: {test_case.test_case_id}")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Test case info
    report_lines.append("## TEST CASE INFORMATION")
    report_lines.append("-" * 40)
    report_lines.append(f"ID:          {test_case.test_case_id}")
    report_lines.append(f"Species:     {test_case.context.species}")
    report_lines.append(f"Sample Type: {test_case.context.sample_type}")
    report_lines.append(f"Colors:      {test_case.n_colors}")
    report_lines.append(f"Source:      {test_case.source_type.value}")
    report_lines.append("")

    # Panel markers
    report_lines.append("## PANEL MARKERS")
    report_lines.append("-" * 40)
    markers = test_case.panel.markers
    for i in range(0, len(markers), 5):
        report_lines.append("  " + ", ".join(markers[i:i+5]))
    report_lines.append("")

    # Scoring summary
    report_lines.append("## SCORING SUMMARY")
    report_lines.append("-" * 40)
    if result.parse_success and result.evaluation:
        eval_data = result.evaluation
        report_lines.append(f"Parse Status:        SUCCESS ({result.parse_format})")
        report_lines.append(f"Hierarchy F1:        {eval_data.hierarchy_f1:.1%}")
        report_lines.append(f"  - Precision:       {eval_data.hierarchy_precision:.1%}")
        report_lines.append(f"  - Recall:          {eval_data.hierarchy_recall:.1%}")
        report_lines.append(f"Structure Accuracy:  {eval_data.structure_accuracy:.1%}")
        report_lines.append(f"Critical Gate Recall:{eval_data.critical_gate_recall:.1%}")
        report_lines.append(f"Hallucination Rate:  {eval_data.hallucination_rate:.1%}")
        report_lines.append(f"Depth Accuracy:      {eval_data.depth_accuracy:.1%}")
    else:
        report_lines.append(f"Parse Status:        FAILED")
        report_lines.append(f"Error:               {result.parse_error}")
    report_lines.append("")

    # Side-by-side hierarchy comparison
    report_lines.append("## HIERARCHY COMPARISON")
    report_lines.append("-" * 40)
    report_lines.append("")

    # Gold standard
    report_lines.append("### Gold Standard (OMIP)")
    gt_lines = gate_to_tree_string(test_case.gating_hierarchy.root)
    for line in gt_lines:
        report_lines.append(f"  {line}")
    report_lines.append("")

    # LLM prediction
    report_lines.append("### LLM Prediction")
    if result.parsed_hierarchy:
        pred_lines = gate_to_tree_string(result.parsed_hierarchy)
        for line in pred_lines:
            report_lines.append(f"  {line}")
    else:
        report_lines.append("  (No hierarchy - parse failed)")
    report_lines.append("")

    # Gate-by-gate analysis
    if result.parse_success and result.evaluation:
        eval_data = result.evaluation

        report_lines.append("## GATE-BY-GATE ANALYSIS")
        report_lines.append("-" * 40)

        # Matching gates
        report_lines.append("")
        report_lines.append(f"### Matching Gates ({len(eval_data.matching_gates)})")
        if eval_data.matching_gates:
            for gate in sorted(eval_data.matching_gates):
                report_lines.append(f"  [OK] {gate}")
        else:
            report_lines.append("  (None)")

        # Missing gates
        report_lines.append("")
        report_lines.append(f"### Missing Gates ({len(eval_data.missing_gates)})")
        if eval_data.missing_gates:
            for gate in sorted(eval_data.missing_gates):
                is_critical = gate in eval_data.missing_critical
                marker = "[CRITICAL]" if is_critical else "[MISSING]"
                report_lines.append(f"  {marker} {gate}")
        else:
            report_lines.append("  (None - all gates present)")

        # Extra gates (hallucinations)
        report_lines.append("")
        report_lines.append(f"### Extra Gates / Hallucinations ({len(eval_data.extra_gates)})")
        if eval_data.extra_gates:
            for gate in sorted(eval_data.extra_gates):
                is_halluc = gate in eval_data.hallucinated_gates
                marker = "[HALLUC]" if is_halluc else "[EXTRA]"
                report_lines.append(f"  {marker} {gate}")
        else:
            report_lines.append("  (None - no extra gates)")

        # Structure errors
        report_lines.append("")
        report_lines.append(f"### Structure Errors ({len(eval_data.structure_errors)})")
        if eval_data.structure_errors:
            for error in eval_data.structure_errors:
                report_lines.append(f"  [ERR] {error}")
        else:
            report_lines.append("  (None - structure correct)")

        report_lines.append("")

    # Critical gates checklist
    report_lines.append("## CRITICAL GATES CHECKLIST")
    report_lines.append("-" * 40)

    critical_gates = ["Time", "Singlets", "Live", "CD45+", "Lymphocytes"]
    gt_gates = set(g[0].lower() for g in extract_all_gates(test_case.gating_hierarchy.root))

    if result.parsed_hierarchy:
        pred_gates = set(g[0].lower() for g in extract_all_gates(result.parsed_hierarchy))
    else:
        pred_gates = set()

    report_lines.append("")
    report_lines.append(f"{'Gate':<20} {'In Gold Std':<15} {'In Prediction':<15} {'Status'}")
    report_lines.append("-" * 65)

    for gate in critical_gates:
        in_gt = any(gate.lower() in g for g in gt_gates)
        in_pred = any(gate.lower() in g for g in pred_gates)

        gt_mark = "YES" if in_gt else "NO"
        pred_mark = "YES" if in_pred else "NO"

        if in_gt and in_pred:
            status = "OK"
        elif in_gt and not in_pred:
            status = "MISSED"
        elif not in_gt and in_pred:
            status = "EXTRA"
        else:
            status = "N/A"

        report_lines.append(f"{gate:<20} {gt_mark:<15} {pred_mark:<15} {status}")

    report_lines.append("")

    # Strategy comparison placeholder
    report_lines.append("## STRATEGY NOTES")
    report_lines.append("-" * 40)
    report_lines.append(f"Condition: {result.condition}")
    report_lines.append(f"Model:     {result.model}")
    report_lines.append("")

    # Recommendations
    report_lines.append("## RECOMMENDATIONS")
    report_lines.append("-" * 40)

    if result.parse_success and result.evaluation:
        eval_data = result.evaluation

        if eval_data.critical_gate_recall < 0.5:
            report_lines.append("- Add explicit QC gate instructions (Time, Singlets, Live/Dead)")

        if eval_data.hallucination_rate > 0.2:
            report_lines.append("- Constrain output to panel markers only")

        if eval_data.structure_accuracy < 0.5:
            report_lines.append("- Provide example hierarchies for reference")

        if len(eval_data.extra_gates) > 10:
            report_lines.append("- Reduce over-generation of subset populations")

        if eval_data.hierarchy_f1 > 0.6:
            report_lines.append("- Good performance - consider as baseline")
    else:
        report_lines.append("- Fix parsing issues before evaluating content")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("")

    report = "\n".join(report_lines)

    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_case.test_case_id.lower().replace('-', '_')}_{timestamp}.txt"

        with open(output_dir / filename, "w") as f:
            f.write(report)

    return report


def generate_comparison_table(
    results: list[ScoringResult],
    test_cases: dict[str, TestCase],
) -> str:
    """
    Generate a comparison table showing LLM vs Gold Standard for all tests.

    Args:
        results: List of scoring results
        test_cases: Dictionary mapping test_case_id to TestCase

    Returns:
        Markdown table as string
    """
    lines = []

    lines.append("# LLM vs Gold Standard Comparison Table")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("| Test Case | Colors | F1 | Struct | Critical | Match | Miss | Extra | Best Gate | Worst Issue |")
    lines.append("|-----------|--------|-----|--------|----------|-------|------|-------|-----------|-------------|")

    for result in sorted(results, key=lambda r: r.test_case_id):
        tc_id = result.test_case_id
        tc = test_cases.get(tc_id)
        colors = tc.n_colors if tc else "?"

        if result.parse_success and result.evaluation:
            e = result.evaluation
            f1 = f"{e.hierarchy_f1:.0%}"
            struct = f"{e.structure_accuracy:.0%}"
            crit = f"{e.critical_gate_recall:.0%}"
            match = len(e.matching_gates)
            miss = len(e.missing_gates)
            extra = len(e.extra_gates)

            # Find best matching gate
            best = e.matching_gates[0] if e.matching_gates else "None"

            # Find worst issue
            if e.missing_critical:
                worst = f"Missing: {e.missing_critical[0]}"
            elif e.missing_gates:
                worst = f"Missing: {e.missing_gates[0]}"
            elif e.hallucinated_gates:
                worst = f"Halluc: {e.hallucinated_gates[0]}"
            else:
                worst = "None"

            lines.append(f"| {tc_id} | {colors} | {f1} | {struct} | {crit} | {match} | {miss} | {extra} | {best[:15]} | {worst[:20]} |")
        else:
            lines.append(f"| {tc_id} | {colors} | - | - | - | - | - | - | - | Parse failed |")

    lines.append("")

    return "\n".join(lines)


def generate_strategy_comparison_report(
    results_by_strategy: dict[str, list[ScoringResult]],
    output_path: Path | str | None = None,
) -> str:
    """
    Generate report comparing different prompting strategies.

    Args:
        results_by_strategy: Dictionary mapping strategy name to results
        output_path: Optional path to save report

    Returns:
        Report as markdown string
    """
    lines = []

    lines.append("# Strategy Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    lines.append("## Summary Metrics by Strategy")
    lines.append("")
    lines.append("| Strategy | N | Parse% | F1 | Struct | Critical | Halluc | Cost |")
    lines.append("|----------|---|--------|-----|--------|----------|--------|------|")

    strategy_metrics = {}

    for strategy, results in results_by_strategy.items():
        valid = [r for r in results if r.parse_success and r.evaluation]
        n = len(results)

        if valid:
            parse_rate = len(valid) / n
            f1_mean = sum(r.evaluation.hierarchy_f1 for r in valid) / len(valid)
            struct_mean = sum(r.evaluation.structure_accuracy for r in valid) / len(valid)
            crit_mean = sum(r.evaluation.critical_gate_recall for r in valid) / len(valid)
            halluc_mean = sum(r.evaluation.hallucination_rate for r in valid) / len(valid)

            strategy_metrics[strategy] = {
                "n": n,
                "parse_rate": parse_rate,
                "f1": f1_mean,
                "structure": struct_mean,
                "critical": crit_mean,
                "hallucination": halluc_mean,
            }

            lines.append(f"| {strategy} | {n} | {parse_rate:.0%} | {f1_mean:.1%} | {struct_mean:.1%} | {crit_mean:.1%} | {halluc_mean:.1%} | - |")
        else:
            lines.append(f"| {strategy} | {n} | 0% | - | - | - | - | - |")

    lines.append("")

    # Best strategy analysis
    if strategy_metrics:
        best_f1 = max(strategy_metrics.items(), key=lambda x: x[1].get("f1", 0))
        best_struct = max(strategy_metrics.items(), key=lambda x: x[1].get("structure", 0))
        best_crit = max(strategy_metrics.items(), key=lambda x: x[1].get("critical", 0))

        lines.append("## Best Performing Strategies")
        lines.append("")
        lines.append(f"- **Best F1 Score:** {best_f1[0]} ({best_f1[1]['f1']:.1%})")
        lines.append(f"- **Best Structure:** {best_struct[0]} ({best_struct[1]['structure']:.1%})")
        lines.append(f"- **Best Critical Gate Recall:** {best_crit[0]} ({best_crit[1]['critical']:.1%})")
        lines.append("")

    # Detailed breakdown per strategy
    lines.append("## Detailed Results by Strategy")
    lines.append("")

    for strategy, results in results_by_strategy.items():
        lines.append(f"### {strategy}")
        lines.append("")

        valid = [r for r in results if r.parse_success and r.evaluation]

        if valid:
            # Best and worst performers
            best = max(valid, key=lambda r: r.evaluation.hierarchy_f1)
            worst = min(valid, key=lambda r: r.evaluation.hierarchy_f1)

            lines.append(f"- Best: {best.test_case_id} (F1={best.evaluation.hierarchy_f1:.1%})")
            lines.append(f"- Worst: {worst.test_case_id} (F1={worst.evaluation.hierarchy_f1:.1%})")

            # Common issues
            all_missing = []
            all_halluc = []
            for r in valid:
                all_missing.extend(r.evaluation.missing_critical)
                all_halluc.extend(r.evaluation.hallucinated_gates)

            if all_missing:
                from collections import Counter
                common_missing = Counter(all_missing).most_common(3)
                lines.append(f"- Most missed: {', '.join(f'{g}({c})' for g, c in common_missing)}")

            if all_halluc:
                from collections import Counter
                common_halluc = Counter(all_halluc).most_common(3)
                lines.append(f"- Most hallucinated: {', '.join(f'{g}({c})' for g, c in common_halluc)}")
        else:
            lines.append("- No valid results")

        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report


class LiveReportWriter:
    """
    Writer that outputs detailed reports after each test case.

    Supports multiple output modes:
    - console: Print to stdout
    - file: Write individual report files
    - aggregate: Accumulate for final report
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        console_output: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize live report writer.

        Args:
            output_dir: Directory for report files
            console_output: Whether to print to console
            verbose: If True, print full reports; if False, print summary only
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.console_output = console_output
        self.verbose = verbose
        self.results: list[ScoringResult] = []
        self.test_cases: dict[str, TestCase] = {}

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_result(
        self,
        result: ScoringResult,
        test_case: TestCase,
        raw_response: str,
    ) -> str:
        """
        Record a test result and generate report.

        Args:
            result: Scoring result
            test_case: Test case
            raw_response: Raw LLM response

        Returns:
            Report string
        """
        self.results.append(result)
        self.test_cases[test_case.test_case_id] = test_case

        # Generate detailed report
        report = generate_per_test_report(
            result=result,
            test_case=test_case,
            raw_response=raw_response,
            output_dir=self.output_dir / "per_test" if self.output_dir else None,
        )

        # Console output
        if self.console_output:
            if self.verbose:
                print(report)
            else:
                self._print_summary(result, test_case)

        return report

    def _print_summary(self, result: ScoringResult, test_case: TestCase):
        """Print condensed summary to console."""
        if result.parse_success and result.evaluation:
            e = result.evaluation
            print(f"  ├─ F1={e.hierarchy_f1:.1%} | Struct={e.structure_accuracy:.1%} | Crit={e.critical_gate_recall:.1%}")
            print(f"  ├─ Matched: {len(e.matching_gates)} | Missing: {len(e.missing_gates)} | Extra: {len(e.extra_gates)}")
            if e.missing_critical:
                print(f"  └─ Missing critical: {', '.join(e.missing_critical[:3])}")
            else:
                print(f"  └─ All critical gates present")
        else:
            print(f"  └─ Parse FAILED: {result.parse_error}")

    def finalize(self) -> dict[str, str]:
        """
        Generate final aggregate reports.

        Returns:
            Dictionary mapping report type to content
        """
        reports = {}

        if not self.results:
            return reports

        # Comparison table
        comparison = generate_comparison_table(self.results, self.test_cases)
        reports["comparison_table"] = comparison

        if self.output_dir:
            with open(self.output_dir / "comparison_table.md", "w") as f:
                f.write(comparison)

        # Strategy comparison if multiple conditions
        conditions = set(r.condition for r in self.results)
        if len(conditions) > 1:
            by_condition = {c: [r for r in self.results if r.condition == c] for c in conditions}
            strategy_report = generate_strategy_comparison_report(
                by_condition,
                self.output_dir / "strategy_comparison.md" if self.output_dir else None,
            )
            reports["strategy_comparison"] = strategy_report

        return reports
