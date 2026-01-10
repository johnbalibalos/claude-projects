"""
Manual review report generator for gating hierarchy predictions.

Generates side-by-side comparison reports for manual verification
of LLM predictions against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from curation.schemas import GateNode, TestCase
from evaluation.scorer import ScoringResult
from evaluation.task_failure import TaskFailureType

ReportLevel = Literal["summary", "outliers", "full"]


@dataclass
class OutlierThresholds:
    """Thresholds for identifying outlier results."""
    min_f1: float = 0.3
    max_hallucination: float = 0.2
    min_critical_recall: float = 0.5


def format_hierarchy_tree(node: dict | GateNode, indent: int = 0) -> str:
    """Format hierarchy as ASCII tree with markers."""
    lines = []

    if isinstance(node, GateNode):
        name = node.name
        markers = node.markers or []
        children = node.children
    else:
        name = node.get("name", "Unknown")
        markers = node.get("markers", [])
        children = node.get("children", [])

    prefix = "  " * indent + "├─ " if indent > 0 else "├─ "
    marker_str = f" [{', '.join(markers)}]" if markers else ""
    lines.append(f"{prefix}{name}{marker_str}")

    for child in children:
        lines.append(format_hierarchy_tree(child, indent + 1))

    return "\n".join(lines)


def is_outlier(result: ScoringResult, thresholds: OutlierThresholds) -> bool:
    """Check if a result is an outlier based on thresholds."""
    if not result.parse_success or not result.evaluation:
        return True  # Parse failures are always outliers

    # Task failures are always outliers
    if result.is_task_failure:
        return True

    return (
        result.hierarchy_f1 < thresholds.min_f1 or
        result.hallucination_rate > thresholds.max_hallucination or
        result.critical_gate_recall < thresholds.min_critical_recall
    )


def get_outlier_reason(result: ScoringResult, thresholds: OutlierThresholds) -> str:
    """Get reason why result is an outlier."""
    if not result.parse_success:
        return "Parse failure"
    if not result.evaluation:
        return "No evaluation"

    reasons = []

    # Check for task failure
    if result.is_task_failure:
        failure_type = result.task_failure_type
        type_names = {
            TaskFailureType.META_QUESTIONS: "Asked questions instead of predicting",
            TaskFailureType.REFUSAL: "Refused to predict",
            TaskFailureType.INSTRUCTIONS: "Gave instructions instead of prediction",
            TaskFailureType.EMPTY: "Empty response",
            TaskFailureType.MALFORMED: "Malformed response",
        }
        reason = type_names.get(failure_type, f"Task failure ({failure_type.value})")
        reasons.append(reason)

    if result.hierarchy_f1 < thresholds.min_f1:
        reasons.append(f"Low F1 ({result.hierarchy_f1:.1%})")
    if result.hallucination_rate > thresholds.max_hallucination:
        reasons.append(f"High hallucination ({result.hallucination_rate:.1%})")
    if result.critical_gate_recall < thresholds.min_critical_recall:
        reasons.append(f"Low critical recall ({result.critical_gate_recall:.1%})")

    return ", ".join(reasons) if reasons else "Unknown"


class ManualReviewReportGenerator:
    """Generates manual review reports for experiment results."""

    def __init__(
        self,
        test_cases: dict[str, TestCase],
        thresholds: OutlierThresholds | None = None,
    ):
        self.test_cases = test_cases
        self.thresholds = thresholds or OutlierThresholds()

    def generate(
        self,
        results: list[ScoringResult],
        level: ReportLevel = "summary",
        model: str | None = None,
        condition: str | None = None,
    ) -> str:
        """
        Generate manual review report.

        Args:
            results: List of ScoringResults to include
            level: Report detail level (summary, outliers, full)
            model: Optional model filter
            condition: Optional condition filter

        Returns:
            Markdown report string
        """
        # Filter results
        filtered = results
        if model:
            filtered = [r for r in filtered if r.model == model]
        if condition:
            filtered = [r for r in filtered if r.condition == condition]

        if not filtered:
            return "# Manual Review Report\n\nNo results to display."

        # Generate report sections
        lines = [
            "# Manual Review Report: LLM vs Ground Truth Comparison",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Results:** {len(filtered)} evaluations",
        ]

        if model:
            lines.append(f"**Model:** {model}")
        if condition:
            lines.append(f"**Condition:** {condition}")

        lines.extend(["", "---", ""])

        # Summary section
        lines.extend(self._generate_summary_section(filtered))

        # Detailed sections based on level
        if level == "full":
            lines.extend(self._generate_detailed_sections(filtered))
        elif level == "outliers":
            outliers = [r for r in filtered if is_outlier(r, self.thresholds)]
            if outliers:
                lines.extend([
                    "",
                    "## Outlier Details",
                    "",
                    f"Showing {len(outliers)} outlier(s) based on thresholds:",
                    f"- F1 < {self.thresholds.min_f1:.0%}",
                    f"- Hallucination > {self.thresholds.max_hallucination:.0%}",
                    f"- Critical Recall < {self.thresholds.min_critical_recall:.0%}",
                    "",
                ])
                lines.extend(self._generate_detailed_sections(outliers))
            else:
                lines.extend(["", "No outliers found.", ""])

        # Legend
        lines.extend(self._generate_legend())

        return "\n".join(lines)

    def _generate_summary_section(self, results: list[ScoringResult]) -> list[str]:
        """Generate summary metrics table."""
        lines = [
            "## Summary",
            "",
            "### Overall Metrics",
            "",
            "| Metric | Mean | Min | Max |",
            "|--------|------|-----|-----|",
        ]

        valid = [r for r in results if r.parse_success and r.evaluation]
        if not valid:
            lines.append("| No valid results | - | - | - |")
            return lines

        metrics = [
            ("Hierarchy F1", [r.hierarchy_f1 for r in valid]),
            ("Structure Accuracy", [r.structure_accuracy for r in valid]),
            ("Critical Gate Recall", [r.critical_gate_recall for r in valid]),
            ("Hallucination Rate", [r.hallucination_rate for r in valid]),
        ]

        for name, values in metrics:
            mean = sum(values) / len(values)
            lines.append(f"| {name} | {mean:.1%} | {min(values):.1%} | {max(values):.1%} |")

        # Parse success rate
        parse_rate = len(valid) / len(results)
        lines.append(f"| Parse Success | {parse_rate:.1%} | - | - |")

        # Task failure metrics
        task_failures = [r for r in results if r.is_task_failure]
        task_failure_rate = len(task_failures) / len(results) if results else 0
        lines.append(f"| Task Failure Rate | {task_failure_rate:.1%} | - | - |")

        # Task failure breakdown if any failures
        if task_failures:
            lines.extend([
                "",
                "### Task Failures by Type",
                "",
                "| Type | Count |",
                "|------|-------|",
            ])
            failure_counts: dict[TaskFailureType, int] = {}
            for r in task_failures:
                ft = r.task_failure_type
                failure_counts[ft] = failure_counts.get(ft, 0) + 1

            type_names = {
                TaskFailureType.META_QUESTIONS: "Meta-questions",
                TaskFailureType.REFUSAL: "Refusals",
                TaskFailureType.INSTRUCTIONS: "Instructions",
                TaskFailureType.EMPTY: "Empty",
                TaskFailureType.MALFORMED: "Malformed",
            }
            for ft, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                name = type_names.get(ft, ft.value)
                lines.append(f"| {name} | {count} |")

        # Results by test case
        lines.extend([
            "",
            "### Results by Test Case",
            "",
            "| Test Case | F1 | Structure | Critical | Halluc | Status |",
            "|-----------|-------|-----------|----------|--------|--------|",
        ])

        # Group by test case
        by_tc: dict[str, list[ScoringResult]] = {}
        for r in results:
            if r.test_case_id not in by_tc:
                by_tc[r.test_case_id] = []
            by_tc[r.test_case_id].append(r)

        for tc_id in sorted(by_tc.keys()):
            tc_results = by_tc[tc_id]
            tc_valid = [r for r in tc_results if r.parse_success and r.evaluation]

            if not tc_valid:
                lines.append(f"| {tc_id} | - | - | - | - | ⚠ Parse fail |")
                continue

            # Use best result for this test case
            best = max(tc_valid, key=lambda r: r.hierarchy_f1)

            status = "✓" if not is_outlier(best, self.thresholds) else "⚠ Outlier"
            lines.append(
                f"| {tc_id} | {best.hierarchy_f1:.1%} | {best.structure_accuracy:.1%} | "
                f"{best.critical_gate_recall:.1%} | {best.hallucination_rate:.1%} | {status} |"
            )

        return lines

    def _generate_detailed_sections(self, results: list[ScoringResult]) -> list[str]:
        """Generate detailed comparison sections for each result."""
        lines = []

        # Group by test case for organization
        by_tc: dict[str, list[ScoringResult]] = {}
        for r in results:
            if r.test_case_id not in by_tc:
                by_tc[r.test_case_id] = []
            by_tc[r.test_case_id].append(r)

        for tc_id in sorted(by_tc.keys()):
            tc = self.test_cases.get(tc_id)
            if not tc:
                continue

            for result in by_tc[tc_id]:
                lines.extend(self._generate_single_comparison(result, tc))

        return lines

    def _generate_single_comparison(
        self,
        result: ScoringResult,
        test_case: TestCase,
    ) -> list[str]:
        """Generate detailed comparison for a single result."""
        lines = [
            "",
            "---",
            "",
            f"## {result.test_case_id}",
            "",
        ]

        # Show condition if multiple
        if result.condition:
            lines.append(f"**Condition:** {result.condition}")

        # Panel markers
        markers = test_case.panel.markers[:15]
        marker_str = ", ".join(markers)
        if len(test_case.panel.markers) > 15:
            marker_str += f"... (+{len(test_case.panel.markers) - 15} more)"
        lines.extend([
            f"**Panel Markers:** {marker_str}",
            "",
        ])

        # Handle parse failures
        if not result.parse_success:
            lines.extend([
                "### ⚠ Parse Failure",
                "",
                f"**Error:** {result.parse_error}",
                "",
            ])
            return lines

        # Metrics summary
        lines.extend([
            "### Metrics Summary",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Hierarchy F1 | {result.hierarchy_f1:.1%} |",
            f"| Structure Accuracy | {result.structure_accuracy:.1%} |",
            f"| Critical Gate Recall | {result.critical_gate_recall:.1%} |",
            f"| Hallucination Rate | {result.hallucination_rate:.1%} |",
        ])

        # Add task failure info if detected
        if result.is_task_failure:
            lines.append(f"| Task Failure | {result.task_failure_type.value} |")

        lines.append("")

        # Task failure details if present
        if result.is_task_failure and result.task_failure:
            lines.extend([
                "### Task Failure Detected",
                "",
                f"**Type:** {result.task_failure_type.value}",
                f"**Confidence:** {result.task_failure.confidence:.1%}",
                "",
            ])
            if result.task_failure.evidence:
                lines.append("**Evidence:**")
                for ev in result.task_failure.evidence[:5]:
                    lines.append(f"- {ev}")
                lines.append("")

        # Outlier reason if applicable
        if is_outlier(result, self.thresholds):
            reason = get_outlier_reason(result, self.thresholds)
            lines.extend([
                f"**Outlier Reason:** {reason}",
                "",
            ])

        # Side-by-side comparison
        gt_tree = format_hierarchy_tree(test_case.gating_hierarchy.root)
        pred_tree = format_hierarchy_tree(result.parsed_hierarchy) if result.parsed_hierarchy else "(No hierarchy)"

        lines.extend([
            "### Side-by-Side Comparison",
            "",
            "<table>",
            "<tr>",
            '<th width="50%">Ground Truth (OMIP)</th>',
            '<th width="50%">LLM Prediction</th>',
            "</tr>",
            "<tr>",
            "<td>",
            "",
            "```",
            gt_tree,
            "```",
            "",
            "</td>",
            "<td>",
            "",
            "```",
            pred_tree,
            "```",
            "",
            "</td>",
            "</tr>",
            "</table>",
            "",
        ])

        # Gate analysis
        if result.evaluation:
            eval_result = result.evaluation

            matching = eval_result.matching_gates[:10]
            missing = eval_result.missing_gates[:10]
            extra = eval_result.extra_gates[:10]
            missing_critical = eval_result.missing_critical

            lines.extend([
                "### Gate Analysis",
                "",
                "| Category | Gates |",
                "|----------|-------|",
            ])

            matching_str = ", ".join(matching)
            if len(eval_result.matching_gates) > 10:
                matching_str += "..."
            lines.append(f"| ✓ **Matching** ({len(eval_result.matching_gates)}) | {matching_str} |")

            missing_str = ", ".join(missing)
            if len(eval_result.missing_gates) > 10:
                missing_str += "..."
            lines.append(f"| ✗ **Missing** ({len(eval_result.missing_gates)}) | {missing_str} |")

            extra_str = ", ".join(extra)
            if len(eval_result.extra_gates) > 10:
                extra_str += "..."
            lines.append(f"| ⚠ **Extra** ({len(eval_result.extra_gates)}) | {extra_str} |")

            if missing_critical:
                lines.append(f"| 🚨 **Missing Critical** ({len(missing_critical)}) | {', '.join(missing_critical)} |")

            lines.append("")

            # Structure errors
            if eval_result.structure_errors:
                lines.extend([
                    "### Structure Errors",
                    "",
                ])
                for error in eval_result.structure_errors[:5]:
                    lines.append(f"- {error}")
                if len(eval_result.structure_errors) > 5:
                    lines.append(f"- ... and {len(eval_result.structure_errors) - 5} more")
                lines.append("")

        return lines

    def _generate_legend(self) -> list[str]:
        """Generate report legend and review guidelines."""
        return [
            "",
            "---",
            "",
            "## Legend",
            "",
            "- **Matching Gates**: Gates correctly predicted (present in both)",
            "- **Missing Gates**: Gates in ground truth but not predicted",
            "- **Extra Gates**: Gates predicted but not in ground truth",
            "- **Missing Critical**: Essential QC/lineage gates that were missed",
            "- **Structure Errors**: Parent-child relationships that don't match",
            "",
            "## Review Guidelines",
            "",
            "When manually reviewing:",
            "1. Check if \"extra\" gates are reasonable alternatives (may indicate ground truth gaps)",
            "2. Evaluate if missing gates are truly missing or just named differently",
            "3. Assess biological plausibility of the predicted hierarchy",
            "4. Note any systematic patterns across test cases",
            "",
        ]


def generate_manual_review_report(
    results: list[ScoringResult],
    test_cases: dict[str, TestCase],
    level: ReportLevel = "summary",
    output_path: Path | None = None,
    model: str | None = None,
    condition: str | None = None,
    thresholds: OutlierThresholds | None = None,
) -> str:
    """
    Convenience function to generate a manual review report.

    Args:
        results: List of ScoringResults
        test_cases: Dict mapping test case ID to TestCase
        level: Report detail level
        output_path: Optional path to save report
        model: Optional model filter
        condition: Optional condition filter
        thresholds: Optional outlier thresholds

    Returns:
        Report markdown string
    """
    generator = ManualReviewReportGenerator(test_cases, thresholds)
    report = generator.generate(results, level, model, condition)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report
