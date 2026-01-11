"""
Format Ablation Hypothesis Test.

Tests whether model failures are idiosyncratic to prose style or robust
to the structure of the task.

Hypotheses:
- H_A: Failure is idiosyncratic to the prose style of OMIP papers
- H_B: Failure is robust to the structure of the task

Test:
Take one failed OMIP and rewrite the prompt context in three different styles:
1. Prose: (Original) "The cells are then gated for..."
2. Structured Table: "Population | Markers | Parent"
3. Pseudocode: "if CD3 and CD4: return T_Cell"

Result Interpretation:
- If model fails across ALL formats: Reasoning deficit is robust
- If model solves Pseudocode but fails Prose: Issue is Information Extraction, not Reasoning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from curation.schemas import GateNode, GatingHierarchy, TestCase


class PromptFormat(Enum):
    """Available prompt formats for ablation testing."""

    PROSE = "prose"
    TABLE = "table"
    PSEUDOCODE = "pseudocode"
    JSON_SPEC = "json_spec"  # Explicit JSON specification
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

    # Computed metrics
    best_format: PromptFormat
    worst_format: PromptFormat
    format_variance: float

    # Interpretation
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
            formats: List of formats to test (default: all formats)
        """
        self.formats = formats or [
            PromptFormat.PROSE,
            PromptFormat.TABLE,
            PromptFormat.PSEUDOCODE,
        ]

    def format_as_prose(self, test_case: TestCase) -> FormattedPrompt:
        """
        Format test case as natural language prose.

        This mimics how OMIP papers describe gating strategies.
        """
        lines = [
            f"## Gating Strategy for {test_case.context.application}",
            "",
            f"This {test_case.panel.n_colors}-color flow cytometry panel is designed for "
            f"{test_case.context.sample_type} analysis in {test_case.context.species}.",
            "",
            "### Panel Markers",
            "",
        ]

        # Describe markers in prose
        marker_groups = self._group_markers_by_function(test_case)
        for group_name, markers in marker_groups.items():
            marker_list = ", ".join(f"{m.marker} ({m.fluorophore})" for m in markers)
            lines.append(f"**{group_name}:** {marker_list}")
        lines.append("")

        # Describe expected gating in prose
        lines.extend([
            "### Expected Gating Approach",
            "",
            "Begin by excluding debris and doublets using FSC and SSC parameters. "
            "Viability staining should be used to identify live cells. "
            "Subsequently, identify major immune lineages based on the available markers. "
            "The hierarchical structure should reflect the biological relationships "
            "between cell populations.",
            "",
        ])

        return FormattedPrompt(
            format=PromptFormat.PROSE,
            content="\n".join(lines),
        )

    def format_as_table(self, test_case: TestCase) -> FormattedPrompt:
        """
        Format test case as structured markdown tables.
        """
        lines = [
            "## Panel Information",
            "",
            "| Marker | Fluorophore | Clone |",
            "|--------|-------------|-------|",
        ]

        for entry in test_case.panel.entries:
            clone = entry.clone or "-"
            lines.append(f"| {entry.marker} | {entry.fluorophore} | {clone} |")

        lines.extend([
            "",
            "## Sample Context",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Sample Type | {test_case.context.sample_type} |",
            f"| Species | {test_case.context.species} |",
            f"| Application | {test_case.context.application} |",
            f"| Panel Size | {test_case.panel.n_colors} colors |",
            "",
            "## Expected Hierarchy Structure",
            "",
            "| Level | Gate Type | Typical Markers |",
            "|-------|-----------|-----------------|",
            "| 1 | Quality Control | FSC, SSC, Time |",
            "| 2 | Viability | Live/Dead stain |",
            "| 3 | Major Lineage | Lineage markers (CD3, CD19, CD14, etc.) |",
            "| 4+ | Subpopulations | Subset markers |",
            "",
        ])

        return FormattedPrompt(
            format=PromptFormat.TABLE,
            content="\n".join(lines),
        )

    def format_as_pseudocode(self, test_case: TestCase) -> FormattedPrompt:
        """
        Format test case as pseudocode/programming logic.

        This explicitly encodes the logical structure of gating.
        """
        lines = [
            "```python",
            "# Flow Cytometry Gating Logic",
            f"# Panel: {test_case.panel.n_colors}-color {test_case.context.sample_type}",
            "",
            "# Available markers:",
            f"MARKERS = {test_case.panel.markers}",
            "",
            "def gate_sample(events):",
            '    """Gate flow cytometry events hierarchically."""',
            "    ",
            "    # Level 1: Quality Control",
            "    qc_events = apply_time_gate(events)  # Remove acquisition artifacts",
            "    singlets = gate_singlets(qc_events, params=['FSC-A', 'FSC-H'])",
            "    ",
            "    # Level 2: Viability",
        ]

        # Find viability marker
        viability_markers = [m for m in test_case.panel.markers
                           if any(v in m.lower() for v in ["live", "dead", "zombie", "viability"])]
        if viability_markers:
            lines.append(f"    live = gate_negative(singlets, marker='{viability_markers[0]}')")
        else:
            lines.append("    live = gate_viability(singlets)  # Use available viability marker")

        lines.extend([
            "    ",
            "    # Level 3: Major Lineages",
            "    # Define populations based on marker logic:",
        ])

        # Add marker-based gating logic
        lineage_markers = self._identify_lineage_markers(test_case)
        for population, logic in lineage_markers.items():
            lines.append(f"    # {population} = {logic}")

        lines.extend([
            "    ",
            "    # Return hierarchical structure as nested dict",
            "    return build_hierarchy()",
            "```",
            "",
            "## Task",
            "Based on the markers and logic above, construct the complete gating hierarchy.",
            "",
        ])

        return FormattedPrompt(
            format=PromptFormat.PSEUDOCODE,
            content="\n".join(lines),
        )

    def format_as_json_spec(self, test_case: TestCase) -> FormattedPrompt:
        """
        Format as explicit JSON specification with marker mappings.
        """
        lines = [
            "## Input Specification",
            "",
            "```json",
            "{",
            f'  "sample_type": "{test_case.context.sample_type}",',
            f'  "species": "{test_case.context.species}",',
            f'  "application": "{test_case.context.application}",',
            '  "panel": [',
        ]

        for i, entry in enumerate(test_case.panel.entries):
            comma = "," if i < len(test_case.panel.entries) - 1 else ""
            clone_str = f'"{entry.clone}"' if entry.clone else "null"
            lines.append(
                f'    {{"marker": "{entry.marker}", "fluorophore": "{entry.fluorophore}", "clone": {clone_str}}}{comma}'
            )

        lines.extend([
            "  ]",
            "}",
            "```",
            "",
            "## Gating Rules",
            "",
            "1. Start with 'All Events' as root",
            "2. Apply quality control gates (Time, Singlets)",
            "3. Gate on viability (Live cells)",
            "4. Identify major lineages using lineage markers",
            "5. Gate subpopulations based on available subset markers",
            "",
            "## Expected Output Format",
            "",
            "Provide a JSON hierarchy where each node has: `name`, `markers`, `children`",
            "",
        ])

        return FormattedPrompt(
            format=PromptFormat.JSON_SPEC,
            content="\n".join(lines),
        )

    def format_as_bullet_points(self, test_case: TestCase) -> FormattedPrompt:
        """
        Format as hierarchical bullet points.
        """
        lines = [
            "## Panel Markers",
            "",
        ]

        for entry in test_case.panel.entries:
            clone_info = f" (clone: {entry.clone})" if entry.clone else ""
            lines.append(f"- {entry.marker}: {entry.fluorophore}{clone_info}")

        lines.extend([
            "",
            "## Sample Information",
            "",
            f"- Sample Type: {test_case.context.sample_type}",
            f"- Species: {test_case.context.species}",
            f"- Application: {test_case.context.application}",
            f"- Panel Size: {test_case.panel.n_colors} colors",
            "",
            "## Gating Strategy Guidelines",
            "",
            "- Start with quality control gates",
            "  - Time gate to exclude acquisition artifacts",
            "  - Singlet gate using FSC-A vs FSC-H",
            "- Apply viability gating",
            "  - Live cells are viability dye negative",
            "- Identify major lineages",
            "  - Use lineage markers (CD3, CD19, CD14, etc.)",
            "  - Consider marker combinations for specificity",
            "- Gate subpopulations",
            "  - Based on available subset markers",
            "",
        ])

        return FormattedPrompt(
            format=PromptFormat.BULLET_POINTS,
            content="\n".join(lines),
        )

    def generate_all_formats(self, test_case: TestCase) -> dict[PromptFormat, FormattedPrompt]:
        """
        Generate prompts in all configured formats.

        Args:
            test_case: Test case to format

        Returns:
            Dict mapping format type to formatted prompt
        """
        formatters = {
            PromptFormat.PROSE: self.format_as_prose,
            PromptFormat.TABLE: self.format_as_table,
            PromptFormat.PSEUDOCODE: self.format_as_pseudocode,
            PromptFormat.JSON_SPEC: self.format_as_json_spec,
            PromptFormat.BULLET_POINTS: self.format_as_bullet_points,
        }

        results = {}
        for fmt in self.formats:
            formatter = formatters.get(fmt)
            if formatter:
                results[fmt] = formatter(test_case)

        return results

    def analyze_results(
        self,
        results: dict[PromptFormat, FormatAblationResult],
        test_case_id: str,
    ) -> FormatAblationAnalysis:
        """
        Analyze results across all formats.

        Args:
            results: Dict mapping format to result
            test_case_id: ID of the test case

        Returns:
            FormatAblationAnalysis with interpretation
        """
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

        # Find best and worst formats
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
        structured_f1 = max(
            results.get(PromptFormat.PSEUDOCODE, FormatAblationResult(
                format=PromptFormat.PSEUDOCODE, f1_score=0, structure_accuracy=0,
                parse_success=False, task_failure=True
            )).f1_score,
            results.get(PromptFormat.TABLE, FormatAblationResult(
                format=PromptFormat.TABLE, f1_score=0, structure_accuracy=0,
                parse_success=False, task_failure=True
            )).f1_score,
        )
        prose_f1 = results.get(PromptFormat.PROSE, FormatAblationResult(
            format=PromptFormat.PROSE, f1_score=0, structure_accuracy=0,
            parse_success=False, task_failure=True
        )).f1_score

        # Is this an extraction issue or reasoning issue?
        is_extraction_issue = (
            structured_f1 > prose_f1 + 0.15  # Structured significantly better than prose
        )
        is_robust_failure = (
            best_result.f1_score < 0.5 and f1_range < 0.15  # All formats fail
        )

        # Generate interpretation
        if is_robust_failure:
            interpretation = (
                f"ROBUST FAILURE ACROSS FORMATS (range={f1_range:.3f}, max_f1={best_result.f1_score:.3f}): "
                "Model fails regardless of prompt format, suggesting a fundamental "
                "reasoning deficit rather than information extraction issues."
            )
        elif is_extraction_issue:
            interpretation = (
                f"EXTRACTION ISSUE DETECTED (structured_f1={structured_f1:.3f} vs prose_f1={prose_f1:.3f}): "
                "Model performs better with structured formats (pseudocode/tables) than prose. "
                "This suggests the issue is information extraction from natural language, "
                "not fundamental reasoning capability."
            )
        elif f1_range > 0.2:
            interpretation = (
                f"FORMAT-SENSITIVE (range={f1_range:.3f}): "
                f"Performance varies significantly by format. Best: {best_format.value} "
                f"(F1={best_result.f1_score:.3f}), Worst: {worst_format.value} "
                f"(F1={worst_result.f1_score:.3f}). Consider optimizing prompt format."
            )
        else:
            interpretation = (
                f"FORMAT-ROBUST (range={f1_range:.3f}): "
                "Performance is consistent across formats, suggesting the model's "
                "ability (or inability) is stable regardless of presentation style."
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

    def _group_markers_by_function(self, test_case: TestCase) -> dict[str, list]:
        """Group panel markers by their likely function."""
        groups: dict[str, list] = {
            "Lineage Markers": [],
            "Activation/Function": [],
            "Viability": [],
            "Other": [],
        }

        lineage_keywords = ["cd3", "cd4", "cd8", "cd19", "cd20", "cd14", "cd16", "cd56", "cd45"]
        activation_keywords = ["hla", "cd25", "cd69", "cd38", "pd-1", "tim-3", "lag-3"]
        viability_keywords = ["live", "dead", "zombie", "viability"]

        for entry in test_case.panel.entries:
            marker_lower = entry.marker.lower()
            if any(kw in marker_lower for kw in viability_keywords):
                groups["Viability"].append(entry)
            elif any(kw in marker_lower for kw in lineage_keywords):
                groups["Lineage Markers"].append(entry)
            elif any(kw in marker_lower for kw in activation_keywords):
                groups["Activation/Function"].append(entry)
            else:
                groups["Other"].append(entry)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _identify_lineage_markers(self, test_case: TestCase) -> dict[str, str]:
        """Identify potential lineage definitions from markers."""
        markers = [m.lower() for m in test_case.panel.markers]
        lineages = {}

        if "cd3" in markers:
            if "cd19" in markers or "cd20" in markers:
                lineages["T_cells"] = "CD3+ and (CD19- or CD20-)"
                lineages["B_cells"] = "CD3- and (CD19+ or CD20+)"
            else:
                lineages["T_cells"] = "CD3+"

            if "cd4" in markers and "cd8" in markers:
                lineages["CD4_T_cells"] = "CD3+ and CD4+ and CD8-"
                lineages["CD8_T_cells"] = "CD3+ and CD4- and CD8+"

        if "cd56" in markers:
            if "cd3" in markers:
                lineages["NK_cells"] = "CD3- and CD56+"
                lineages["NKT_cells"] = "CD3+ and CD56+"
            else:
                lineages["NK_cells"] = "CD56+"

        if "cd14" in markers:
            lineages["Monocytes"] = "CD14+"

        return lineages
