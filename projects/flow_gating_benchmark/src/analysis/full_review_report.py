"""
Full review report generator for manual inspection of benchmark results.

Shows complete data flow for each prediction:
1. Prompt sent to benchmark model
2. Raw LLM response
3. Flattened version (what judge sees)
4. Parsed hierarchy (what scorer uses)
5. Scoring metrics
6. Judge prompt, response, and scores
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from curation.schemas import GateNode, TestCase
from evaluation.response_parser import parse_llm_response
from experiments.llm_judge import format_prediction_for_judge


@dataclass
class FullReviewEntry:
    """All data for a single prediction for review."""

    # Identity
    test_case_id: str
    model: str
    condition: str
    bootstrap_run: int

    # Test case context
    sample_type: str
    species: str
    panel_markers: list[str]

    # Benchmark model
    prompt: str
    raw_response: str

    # Transformations
    flattened_for_judge: str  # What judge sees
    parsed_hierarchy: dict | None  # What scorer uses
    parse_success: bool
    parse_error: str | None

    # Scoring results
    hierarchy_f1: float
    structure_accuracy: float
    critical_gate_recall: float
    hallucination_rate: float

    # Judge results (optional)
    judge_prompt: str | None = None
    judge_raw_response: str | None = None
    judge_completeness: float | None = None
    judge_accuracy: float | None = None
    judge_scientific: float | None = None
    judge_overall: float | None = None
    judge_issues: str | None = None
    judge_summary: str | None = None


def format_hierarchy_tree(node: dict | GateNode, indent: int = 0) -> str:
    """Format hierarchy as ASCII tree."""
    lines = []

    if isinstance(node, GateNode):
        name = node.name
        children = node.children
    else:
        name = node.get("name", "Unknown")
        children = node.get("children", [])

    prefix = "  " * indent + "├─ " if indent > 0 else ""
    lines.append(f"{prefix}{name}")

    for child in children:
        lines.append(format_hierarchy_tree(child, indent + 1))

    return "\n".join(lines)


def build_full_review_entry(
    prediction: Any,
    scoring_result: Any,
    test_case: TestCase,
    judge_result: Any | None = None,
) -> FullReviewEntry:
    """Build a full review entry from prediction, scoring, and judge results."""
    # Get flattened version (what judge sees)
    flattened = format_prediction_for_judge(prediction.raw_response)

    # Get parsed hierarchy (what scorer uses)
    parse_result = parse_llm_response(prediction.raw_response)

    return FullReviewEntry(
        # Identity
        test_case_id=prediction.test_case_id,
        model=prediction.model,
        condition=prediction.condition,
        bootstrap_run=prediction.bootstrap_run,
        # Test case context
        sample_type=test_case.context.sample_type,
        species=test_case.context.species,
        panel_markers=test_case.panel.markers,
        # Benchmark model
        prompt=getattr(prediction, "prompt", ""),
        raw_response=prediction.raw_response,
        # Transformations
        flattened_for_judge=flattened,
        parsed_hierarchy=parse_result.hierarchy if parse_result.success else None,
        parse_success=parse_result.success,
        parse_error=parse_result.error,
        # Scoring results
        hierarchy_f1=scoring_result.hierarchy_f1,
        structure_accuracy=scoring_result.structure_accuracy,
        critical_gate_recall=scoring_result.critical_gate_recall,
        hallucination_rate=getattr(scoring_result, "hallucination_rate", 0.0),
        # Judge results
        judge_prompt=getattr(judge_result, "judge_prompt", None) if judge_result else None,
        judge_raw_response=getattr(judge_result, "judge_raw_response", None) if judge_result else None,
        judge_completeness=judge_result.completeness if judge_result else None,
        judge_accuracy=judge_result.accuracy if judge_result else None,
        judge_scientific=judge_result.scientific if judge_result else None,
        judge_overall=judge_result.overall if judge_result else None,
        judge_issues=judge_result.issues if judge_result else None,
        judge_summary=judge_result.summary if judge_result else None,
    )


class FullReviewReportGenerator:
    """Generates comprehensive review reports showing full data flow."""

    def __init__(self, test_cases: dict[str, TestCase]):
        self.test_cases = test_cases

    def generate(
        self,
        entries: list[FullReviewEntry],
        include_judge: bool = True,
        max_response_length: int = 2000,
    ) -> str:
        """Generate full review report as markdown."""
        lines = [
            "# Full Review Report: Benchmark Data Flow",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Entries:** {len(entries)}",
            "",
            "This report shows the complete data flow for each prediction:",
            "1. **Prompt** → sent to benchmark model",
            "2. **Raw Response** → LLM output text",
            "3. **Flattened** → arrow notation sent to judge",
            "4. **Parsed Hierarchy** → tree structure used for scoring",
            "5. **Scores** → automated metrics",
            "6. **Judge** → LLM judge evaluation (if available)",
            "",
            "---",
            "",
        ]

        # Summary table
        lines.extend(self._generate_summary_table(entries))

        # Detailed entries
        for entry in entries:
            lines.extend(
                self._generate_entry_section(entry, include_judge, max_response_length)
            )

        return "\n".join(lines)

    def _generate_summary_table(self, entries: list[FullReviewEntry]) -> list[str]:
        """Generate summary table of all entries."""
        lines = [
            "## Summary",
            "",
            "| Test Case | Model | Condition | F1 | Structure | Parse | Judge |",
            "|-----------|-------|-----------|-----|-----------|-------|-------|",
        ]

        for e in entries:
            parse_status = "✓" if e.parse_success else "✗"
            judge_score = f"{e.judge_overall:.0%}" if e.judge_overall is not None else "-"
            lines.append(
                f"| {e.test_case_id} | {e.model} | {e.condition} | "
                f"{e.hierarchy_f1:.0%} | {e.structure_accuracy:.0%} | "
                f"{parse_status} | {judge_score} |"
            )

        lines.extend(["", "---", ""])
        return lines

    def _generate_entry_section(
        self,
        entry: FullReviewEntry,
        include_judge: bool,
        max_length: int,
    ) -> list[str]:
        """Generate detailed section for a single entry."""
        lines = [
            f"## {entry.test_case_id} | {entry.model} | {entry.condition}",
            "",
            f"**Sample:** {entry.sample_type} ({entry.species})",
            f"**Panel:** {', '.join(entry.panel_markers[:10])}{'...' if len(entry.panel_markers) > 10 else ''}",
            f"**Bootstrap Run:** {entry.bootstrap_run}",
            "",
        ]

        # Metrics summary
        lines.extend([
            "### Metrics",
            "",
            "| Metric | Value | Source |",
            "|--------|-------|--------|",
            f"| Hierarchy F1 | {entry.hierarchy_f1:.1%} | Scorer |",
            f"| Structure Accuracy | {entry.structure_accuracy:.1%} | Scorer |",
            f"| Critical Gate Recall | {entry.critical_gate_recall:.1%} | Scorer |",
            f"| Hallucination Rate | {entry.hallucination_rate:.1%} | Scorer |",
            f"| Parse Success | {'Yes' if entry.parse_success else 'No'} | Parser |",
        ])

        if entry.judge_overall is not None:
            lines.extend([
                f"| Judge Overall | {entry.judge_overall:.1%} | LLM Judge |",
                f"| Judge Completeness | {entry.judge_completeness:.1%} | LLM Judge |",
                f"| Judge Accuracy | {entry.judge_accuracy:.1%} | LLM Judge |",
                f"| Judge Scientific | {entry.judge_scientific:.1%} | LLM Judge |",
            ])

        lines.append("")

        # Prompt sent to model
        lines.extend([
            "### 1. Prompt Sent to Benchmark Model",
            "",
            "<details>",
            "<summary>Click to expand prompt</summary>",
            "",
            "```",
            self._truncate(entry.prompt, max_length) if entry.prompt else "(prompt not saved)",
            "```",
            "",
            "</details>",
            "",
        ])

        # Side-by-side: Raw response vs Flattened
        lines.extend([
            "### 2. Model Output Comparison",
            "",
            "<table>",
            "<tr>",
            '<th width="50%">Raw LLM Response</th>',
            '<th width="50%">Flattened for Judge</th>',
            "</tr>",
            "<tr>",
            "<td>",
            "",
            "```",
            self._truncate(entry.raw_response, max_length),
            "```",
            "",
            "</td>",
            "<td>",
            "",
            "```",
            entry.flattened_for_judge,
            "```",
            "",
            "</td>",
            "</tr>",
            "</table>",
            "",
        ])

        # Parsed hierarchy (used for scoring)
        lines.extend([
            "### 3. Parsed Hierarchy (Used for Scoring)",
            "",
        ])

        if entry.parse_success and entry.parsed_hierarchy:
            tree = format_hierarchy_tree(entry.parsed_hierarchy)
            lines.extend([
                "```",
                tree,
                "```",
                "",
            ])
        else:
            lines.extend([
                f"**Parse Failed:** {entry.parse_error or 'Unknown error'}",
                "",
            ])

        # Judge section
        if include_judge and entry.judge_prompt:
            lines.extend([
                "### 4. LLM Judge Evaluation",
                "",
                "#### Judge Prompt",
                "",
                "<details>",
                "<summary>Click to expand judge prompt</summary>",
                "",
                "```",
                self._truncate(entry.judge_prompt, max_length),
                "```",
                "",
                "</details>",
                "",
                "#### Judge Response",
                "",
                "```",
                entry.judge_raw_response or "(no response)",
                "```",
                "",
            ])

            if entry.judge_issues:
                lines.extend([
                    f"**Issues:** {entry.judge_issues}",
                    "",
                ])

            if entry.judge_summary:
                lines.extend([
                    f"**Summary:** {entry.judge_summary}",
                    "",
                ])

        lines.extend(["---", ""])
        return lines

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text with indicator."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + f"\n... [truncated, {len(text) - max_length} more chars]"


def generate_full_review_report(
    predictions: list[Any],
    scoring_results: list[Any],
    test_cases: dict[str, TestCase],
    judge_results: list[Any] | None = None,
    output_path: Path | None = None,
    include_judge: bool = True,
    max_response_length: int = 2000,
) -> str:
    """
    Convenience function to generate a full review report.

    Args:
        predictions: List of Prediction objects
        scoring_results: List of ScoringResult objects
        test_cases: Dict mapping test case ID to TestCase
        judge_results: Optional list of JudgeResult objects
        output_path: Optional path to save report
        include_judge: Whether to include judge sections
        max_response_length: Max chars for truncated sections

    Returns:
        Report markdown string
    """
    # Build lookup maps
    scoring_map = {
        (r.test_case_id, r.model, r.condition, getattr(r, "bootstrap_run", 0)): r
        for r in scoring_results
    }

    judge_map = {}
    if judge_results:
        judge_map = {
            (r.test_case_id, r.model, r.condition, r.bootstrap_run): r
            for r in judge_results
        }

    # Build entries
    entries = []
    for pred in predictions:
        tc = test_cases.get(pred.test_case_id)
        if not tc:
            continue

        key = (pred.test_case_id, pred.model, pred.condition, pred.bootstrap_run)
        scoring = scoring_map.get(key)
        if not scoring:
            continue

        judge = judge_map.get(key)

        entry = build_full_review_entry(pred, scoring, tc, judge)
        entries.append(entry)

    # Generate report
    generator = FullReviewReportGenerator(test_cases)
    report = generator.generate(entries, include_judge, max_response_length)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report
