#!/usr/bin/env python3
"""
Format Ablation Test.

Tests whether model failures are idiosyncratic to prose style or robust
to the structure of the task.

Formats tested:
- Prose: Natural language description (mimics OMIP papers)
- Table: Structured markdown tables
- Pseudocode: Programming logic style
- JSON: Explicit specification format
- Bullets: Hierarchical bullet points

Results:
- If model fails all formats → robust reasoning deficit
- If model solves structured but fails prose → extraction issue, not reasoning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PromptFormat(Enum):
    """Available prompt formats for ablation testing."""
    PROSE = "prose"
    TABLE = "table"
    PSEUDOCODE = "pseudocode"
    JSON_SPEC = "json_spec"
    BULLET_POINTS = "bullet_points"


@dataclass
class FormattedPrompt:
    """A prompt formatted in a specific style."""
    format: PromptFormat
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FormatAblationResult:
    """Result from testing a single format."""
    format: PromptFormat
    f1_score: float
    structure_accuracy: float
    parse_success: bool
    task_failure: bool
    raw_response: str | None = None


@dataclass
class FormatAblationAnalysis:
    """Complete analysis of format ablation test."""
    test_case_id: str
    results: dict[PromptFormat, FormatAblationResult]
    best_format: PromptFormat
    worst_format: PromptFormat
    format_variance: float
    is_robust_failure: bool
    is_extraction_issue: bool
    interpretation: str

    @property
    def f1_by_format(self) -> dict[str, float]:
        """Get F1 scores indexed by format name."""
        return {fmt.value: result.f1_score for fmt, result in self.results.items()}


class FormatAblationTest:
    """
    Format ablation test implementation.

    Tests whether the model's failures are due to:
    1. Information extraction difficulties (fails prose, succeeds structured)
    2. Genuine reasoning deficits (fails all formats)
    """

    def __init__(self, formats: list[PromptFormat] | None = None):
        """
        Initialize format ablation test.

        Args:
            formats: List of formats to test (default: prose, table, pseudocode)
        """
        self.formats = formats or [
            PromptFormat.PROSE,
            PromptFormat.TABLE,
            PromptFormat.PSEUDOCODE,
        ]

    def format_panel_as_prose(
        self,
        panel: list[dict],
        context: dict,
    ) -> FormattedPrompt:
        """Format panel as natural language prose."""
        markers = [e.get("marker", "") for e in panel]
        sample_type = context.get("sample_type", "sample")
        species = context.get("species", "")
        application = context.get("application", "immunophenotyping")

        lines = [
            f"## Gating Strategy for {application}",
            "",
            f"This {len(panel)}-color flow cytometry panel is designed for "
            f"{sample_type} analysis{' in ' + species if species else ''}.",
            "",
            "### Panel Markers",
            "",
            f"The panel includes: {', '.join(markers)}.",
            "",
            "### Expected Approach",
            "",
            "Begin with quality control gates (singlets, viability). "
            "Then identify major immune lineages based on lineage markers. "
            "Gate subpopulations as appropriate for the panel design.",
            "",
        ]

        return FormattedPrompt(
            format=PromptFormat.PROSE,
            content="\n".join(lines),
        )

    def format_panel_as_table(
        self,
        panel: list[dict],
        context: dict,
    ) -> FormattedPrompt:
        """Format panel as structured markdown tables."""
        lines = [
            "## Panel Information",
            "",
            "| Marker | Fluorophore | Clone |",
            "|--------|-------------|-------|",
        ]

        for entry in panel:
            marker = entry.get("marker", "")
            fluor = entry.get("fluorophore", "")
            clone = entry.get("clone", "-") or "-"
            lines.append(f"| {marker} | {fluor} | {clone} |")

        lines.extend([
            "",
            "## Context",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Sample | {context.get('sample_type', 'Unknown')} |",
            f"| Species | {context.get('species', 'Unknown')} |",
            f"| Application | {context.get('application', 'Unknown')} |",
            "",
        ])

        return FormattedPrompt(
            format=PromptFormat.TABLE,
            content="\n".join(lines),
        )

    def format_panel_as_pseudocode(
        self,
        panel: list[dict],
        context: dict,
    ) -> FormattedPrompt:
        """Format panel as programming pseudocode."""
        markers = [e.get("marker", "") for e in panel]

        lines = [
            "```python",
            "# Flow Cytometry Gating Logic",
            f"# Panel: {len(panel)}-color {context.get('sample_type', 'sample')}",
            "",
            f"MARKERS = {markers}",
            "",
            "def gate_sample(events):",
            '    """Gate flow cytometry events hierarchically."""',
            "    ",
            "    # Level 1: Quality Control",
            "    singlets = gate_singlets(events)",
            "    ",
            "    # Level 2: Viability",
            "    live = gate_live_cells(singlets)",
            "    ",
            "    # Level 3+: Lineage and subset gating",
            "    # Gate based on available markers",
            "    ",
            "    return build_hierarchy()",
            "```",
            "",
            "Construct the complete gating hierarchy based on the markers.",
            "",
        ]

        return FormattedPrompt(
            format=PromptFormat.PSEUDOCODE,
            content="\n".join(lines),
        )

    def format_panel_as_json(
        self,
        panel: list[dict],
        context: dict,
    ) -> FormattedPrompt:
        """Format panel as explicit JSON specification."""
        import json

        spec = {
            "sample_type": context.get("sample_type", "Unknown"),
            "species": context.get("species", "Unknown"),
            "application": context.get("application", "Unknown"),
            "panel": [
                {
                    "marker": e.get("marker", ""),
                    "fluorophore": e.get("fluorophore", ""),
                    "clone": e.get("clone"),
                }
                for e in panel
            ],
        }

        lines = [
            "## Input Specification",
            "",
            "```json",
            json.dumps(spec, indent=2),
            "```",
            "",
            "## Output Format",
            "",
            "Provide a JSON hierarchy with: `{name, markers, children}`",
            "",
        ]

        return FormattedPrompt(
            format=PromptFormat.JSON_SPEC,
            content="\n".join(lines),
        )

    def format_panel_as_bullets(
        self,
        panel: list[dict],
        context: dict,
    ) -> FormattedPrompt:
        """Format panel as hierarchical bullet points."""
        lines = [
            "## Panel Markers",
            "",
        ]

        for entry in panel:
            marker = entry.get("marker", "")
            fluor = entry.get("fluorophore", "")
            clone = entry.get("clone", "")
            clone_info = f" (clone: {clone})" if clone else ""
            lines.append(f"- {marker}: {fluor}{clone_info}")

        lines.extend([
            "",
            "## Context",
            "",
            f"- Sample: {context.get('sample_type', 'Unknown')}",
            f"- Species: {context.get('species', 'Unknown')}",
            f"- Application: {context.get('application', 'Unknown')}",
            "",
            "## Task",
            "",
            "Construct the gating hierarchy for this panel.",
            "",
        ])

        return FormattedPrompt(
            format=PromptFormat.BULLET_POINTS,
            content="\n".join(lines),
        )

    def generate_all_formats(
        self,
        panel: list[dict],
        context: dict,
    ) -> dict[PromptFormat, FormattedPrompt]:
        """Generate prompts in all configured formats."""
        formatters = {
            PromptFormat.PROSE: self.format_panel_as_prose,
            PromptFormat.TABLE: self.format_panel_as_table,
            PromptFormat.PSEUDOCODE: self.format_panel_as_pseudocode,
            PromptFormat.JSON_SPEC: self.format_panel_as_json,
            PromptFormat.BULLET_POINTS: self.format_panel_as_bullets,
        }

        results = {}
        for fmt in self.formats:
            formatter = formatters.get(fmt)
            if formatter:
                results[fmt] = formatter(panel, context)

        return results

    def analyze_results(
        self,
        results: dict[PromptFormat, FormatAblationResult],
        test_case_id: str,
    ) -> FormatAblationAnalysis:
        """Analyze results across all formats."""
        if not results:
            return FormatAblationAnalysis(
                test_case_id=test_case_id,
                results={},
                best_format=PromptFormat.PROSE,
                worst_format=PromptFormat.PROSE,
                format_variance=0.0,
                is_robust_failure=True,
                is_extraction_issue=False,
                interpretation="No results to analyze",
            )

        # Find best and worst
        sorted_formats = sorted(
            results.items(),
            key=lambda x: x[1].f1_score,
            reverse=True,
        )
        best_format, best_result = sorted_formats[0]
        worst_format, worst_result = sorted_formats[-1]

        # Calculate variance
        f1_scores = [r.f1_score for r in results.values()]
        mean_f1 = sum(f1_scores) / len(f1_scores)
        variance = sum((s - mean_f1) ** 2 for s in f1_scores) / len(f1_scores)

        # Determine interpretation
        f1_range = best_result.f1_score - worst_result.f1_score

        # Check if structured formats are better
        structured_formats = [PromptFormat.PSEUDOCODE, PromptFormat.TABLE, PromptFormat.JSON_SPEC]
        structured_f1 = max(
            (results[f].f1_score for f in structured_formats if f in results),
            default=0.0,
        )
        prose_f1 = results.get(PromptFormat.PROSE, FormatAblationResult(
            format=PromptFormat.PROSE, f1_score=0, structure_accuracy=0,
            parse_success=False, task_failure=True
        )).f1_score

        is_extraction_issue = structured_f1 > prose_f1 + 0.15
        is_robust_failure = best_result.f1_score < 0.5 and f1_range < 0.15

        # Generate interpretation
        if is_robust_failure:
            interpretation = (
                f"ROBUST FAILURE (range={f1_range:.3f}, max={best_result.f1_score:.3f}): "
                "Model fails regardless of format → reasoning deficit."
            )
        elif is_extraction_issue:
            interpretation = (
                f"EXTRACTION ISSUE (structured={structured_f1:.3f} vs prose={prose_f1:.3f}): "
                "Model succeeds with structured formats → parsing problem, not reasoning."
            )
        elif f1_range > 0.2:
            interpretation = (
                f"FORMAT-SENSITIVE (range={f1_range:.3f}): "
                f"Best: {best_format.value}, Worst: {worst_format.value}."
            )
        else:
            interpretation = (
                f"FORMAT-ROBUST (range={f1_range:.3f}): "
                "Consistent performance across formats."
            )

        return FormatAblationAnalysis(
            test_case_id=test_case_id,
            results=results,
            best_format=best_format,
            worst_format=worst_format,
            format_variance=variance,
            is_robust_failure=is_robust_failure,
            is_extraction_issue=is_extraction_issue,
            interpretation=interpretation,
        )


def run_format_ablation_example():
    """Example usage of format ablation test."""
    panel = [
        {"marker": "CD3", "fluorophore": "BUV395", "clone": "UCHT1"},
        {"marker": "CD4", "fluorophore": "BUV496", "clone": "SK3"},
        {"marker": "CD8", "fluorophore": "BUV661", "clone": "SK1"},
        {"marker": "CD19", "fluorophore": "BV605", "clone": "SJ25C1"},
    ]
    context = {
        "sample_type": "Human PBMC",
        "species": "human",
        "application": "T and B cell immunophenotyping",
    }

    test = FormatAblationTest()
    prompts = test.generate_all_formats(panel, context)

    print("=== FORMAT ABLATION TEST EXAMPLE ===\n")
    for fmt, prompt in prompts.items():
        print(f"--- {fmt.value.upper()} ---")
        print(prompt.content[:300] + "..." if len(prompt.content) > 300 else prompt.content)
        print()


if __name__ == "__main__":
    run_format_ablation_example()
