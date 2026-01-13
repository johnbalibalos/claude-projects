"""
Data schemas for ground truth test cases.

These Pydantic models define the structure of benchmark test cases,
including panel information, gating hierarchies, and validation metadata.
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Complexity(str, Enum):
    """Test case complexity level based on panel size."""

    SIMPLE = "simple"  # ≤15 colors
    MEDIUM = "medium"  # 16-25 colors
    COMPLEX = "complex"  # 26+ colors


class Difficulty(str, Enum):
    """Test case difficulty based on gating hierarchy size."""

    EASY = "easy"  # ≤10 gates
    MEDIUM = "medium"  # 11-18 gates
    HARD = "hard"  # >18 gates


class SourceType(str, Enum):
    """Source of the ground truth data."""

    OMIP_PAPER = "omip_paper"
    FLOWREPOSITORY_WSP = "flowrepository_wsp"
    EXPERT_ANNOTATED = "expert_annotated"
    SYNTHETIC = "synthetic"


class GateType(str, Enum):
    """Types of gates in flow cytometry."""

    RECTANGLE = "RectangleGate"
    POLYGON = "PolygonGate"
    ELLIPSE = "EllipseGate"
    QUADRANT = "QuadrantGate"
    BOOLEAN = "BooleanGate"
    RANGE = "RangeGate"
    UNKNOWN = "Unknown"


class PanelEntry(BaseModel):
    """A single marker in the flow cytometry panel."""

    marker: str = Field(..., description="Target marker (e.g., CD3, CD4)")
    fluorophore: str | None = Field(None, description="Fluorophore conjugate (e.g., BUV395, PE)")
    clone: str | None = Field(None, description="Antibody clone (e.g., UCHT1)")
    channel: str | None = Field(None, description="Detector channel name")
    vendor: str | None = Field(None, description="Antibody vendor")
    cat_number: str | None = Field(None, description="Catalog number")


class Panel(BaseModel):
    """Complete flow cytometry panel definition."""

    entries: list[PanelEntry] = Field(..., description="List of panel markers")

    @property
    def n_colors(self) -> int:
        """Number of fluorescent markers (excluding scatter)."""
        return len([e for e in self.entries if e.fluorophore and e.fluorophore not in ["FSC", "SSC"]])

    @property
    def markers(self) -> list[str]:
        """List of marker names."""
        return [e.marker for e in self.entries]

    @property
    def complexity(self) -> Complexity:
        """Determine complexity based on color count."""
        n = self.n_colors
        if n <= 15:
            return Complexity.SIMPLE
        elif n <= 25:
            return Complexity.MEDIUM
        else:
            return Complexity.COMPLEX


class MarkerExpression(BaseModel):
    """A marker expression with positive/negative status."""

    marker: str = Field(..., description="Marker name (e.g., 'CD3', 'CD19')")
    positive: bool = Field(True, description="True for positive (+), False for negative (-)")
    level: str | None = Field(
        None,
        description="Expression level modifier (e.g., 'bright', 'dim', 'high', 'low')",
    )

    def __str__(self) -> str:
        """Return string representation like 'CD3+' or 'CD19-'."""
        suffix = "+" if self.positive else "-"
        if self.level:
            suffix = self.level
        return f"{self.marker}{suffix}"


def parse_marker_logic_string(value: str) -> list[MarkerExpression]:
    """
    Parse a marker logic string into MarkerExpression objects.

    Examples:
        "CD3+" -> [MarkerExpression(marker="CD3", positive=True)]
        "CD3+ CD19-" -> [MarkerExpression(marker="CD3", positive=True),
                        MarkerExpression(marker="CD19", positive=False)]
        "CD45RA+ CCR7+ CD28+ CD95-" -> [...]
        "FSC-A vs FSC-H" -> [] (scatter gates, not marker logic)
        "CD56dim" -> [MarkerExpression(marker="CD56", positive=True, level="dim")]
    """
    if not value or not isinstance(value, str):
        return []

    # Skip scatter/morphology gates (not marker logic)
    scatter_patterns = ["FSC", "SSC", " vs ", "Time", "morphology", "scatter"]
    if any(p.lower() in value.lower() for p in scatter_patterns):
        return []

    expressions = []

    # Pattern to match marker expressions like CD3+, CD19-, CD56bright, CD4+/low
    # Handles: CD3+, CD19-, CD56dim, CD56bright, CD45RAhigh, HLA-DR+, γδTCR+
    pattern = r'([A-Za-z0-9γδαβ/_-]+?)(bright|dim|high|low|\+/?low|\+|-)'

    for match in re.finditer(pattern, value):
        marker = match.group(1)
        suffix = match.group(2).lower()

        # Skip if marker looks like a scatter parameter
        if marker.upper() in ("FSC", "SSC", "FSC-A", "FSC-H", "SSC-A", "SSC-H"):
            continue

        if suffix == '+' or suffix == '+/low':
            expressions.append(MarkerExpression(marker=marker, positive=True, level=None))
        elif suffix == '-':
            expressions.append(MarkerExpression(marker=marker, positive=False, level=None))
        elif suffix in ('bright', 'high'):
            expressions.append(MarkerExpression(marker=marker, positive=True, level=suffix))
        elif suffix in ('dim', 'low'):
            expressions.append(MarkerExpression(marker=marker, positive=True, level=suffix))

    return expressions


class GateNode(BaseModel):
    """A single gate in the hierarchy."""

    name: str = Field(..., description="Gate name (e.g., 'Live', 'CD3+')")
    markers: list[str] = Field(
        default_factory=list,
        description="Markers/dimensions used for this gate (legacy, use marker_logic for new gates)",
    )
    marker_logic: list[MarkerExpression] = Field(
        default_factory=list,
        description="Marker expressions defining this population (e.g., CD3+ CD19-)",
    )

    @field_validator("marker_logic", mode="before")
    @classmethod
    def parse_marker_logic(cls, v):
        """Accept string format and convert to MarkerExpression list."""
        if v is None:
            return []
        if isinstance(v, str):
            return parse_marker_logic_string(v)
        if isinstance(v, list):
            # Could be list of dicts or list of MarkerExpression
            return v
        return []

    gate_type: GateType | str = Field(
        default=GateType.UNKNOWN,
        description="Type of gate (rectangle, polygon, etc.)",
    )
    children: list[GateNode] = Field(
        default_factory=list,
        description="Child gates in the hierarchy",
    )
    is_critical: bool = Field(
        default=False,
        description="Whether this is a critical gate that must be present",
    )
    notes: str | None = Field(None, description="Additional notes about this gate")

    @property
    def marker_logic_str(self) -> str:
        """Return string representation of marker logic (e.g., 'CD3+ CD19-')."""
        return " ".join(str(m) for m in self.marker_logic)


class GatingHierarchy(BaseModel):
    """Complete gating hierarchy as a tree structure."""

    root: GateNode = Field(..., description="Root of the gating tree")

    def get_all_gates(self) -> list[str]:
        """Get flat list of all gate names."""
        gates = []

        def traverse(node: GateNode):
            gates.append(node.name)
            for child in node.children:
                traverse(child)

        traverse(self.root)
        return gates

    def get_critical_gates(self) -> list[str]:
        """Get list of critical gate names."""
        gates = []

        def traverse(node: GateNode):
            if node.is_critical:
                gates.append(node.name)
            for child in node.children:
                traverse(child)

        traverse(self.root)
        return gates

    def get_parent_map(self) -> dict[str, str | None]:
        """Get mapping of gate name to parent name."""
        parent_map: dict[str, str | None] = {self.root.name: None}

        def traverse(node: GateNode):
            for child in node.children:
                parent_map[child.name] = node.name
                traverse(child)

        traverse(self.root)
        return parent_map

    def count_gates(self) -> int:
        """Count total gates excluding root 'All Events'."""
        count = 0

        def traverse(node: GateNode):
            nonlocal count
            count += 1
            for child in node.children:
                traverse(child)

        traverse(self.root)
        return count - 1  # Exclude root

    def get_max_depth(self) -> int:
        """Get maximum depth of the hierarchy."""

        def traverse(node: GateNode, depth: int) -> int:
            if not node.children:
                return depth
            return max(traverse(child, depth + 1) for child in node.children)

        return traverse(self.root, 0)

    def compute_difficulty(self) -> Difficulty:
        """Compute difficulty based on gate count."""
        n_gates = self.count_gates()
        if n_gates <= 10:
            return Difficulty.EASY
        elif n_gates <= 18:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD


class ExperimentContext(BaseModel):
    """Experimental context for the test case."""

    sample_type: str = Field(..., description="Sample type (e.g., 'Human PBMC')")
    species: str = Field(..., description="Species (e.g., 'human', 'mouse')")
    application: str = Field(..., description="Experimental application/goal")
    tissue: str | None = Field(None, description="Tissue source")
    disease_state: str | None = Field(None, description="Disease state if applicable")
    additional_notes: str | None = Field(None, description="Additional context")


class ValidationInfo(BaseModel):
    """Validation and curation metadata."""

    paper_source: str | None = Field(None, description="Source in paper (e.g., 'Figure 2')")
    wsp_extraction_match: bool | None = Field(
        None,
        description="Whether paper and .wsp hierarchies match",
    )
    discrepancies: list[str] = Field(
        default_factory=list,
        description="List of discrepancies between sources",
    )
    curator_notes: str | None = Field(None, description="Notes from curator")
    validation_date: date | None = Field(None, description="Date of validation")


class CurationMetadata(BaseModel):
    """Metadata about the curation process."""

    curation_date: date = Field(..., description="Date of curation")
    curator: str = Field(..., description="Who curated this test case")
    flowkit_version: str | None = Field(None, description="flowkit version used")
    last_updated: date | None = Field(None, description="Last update date")


class TestCase(BaseModel):
    """Complete test case for the gating benchmark."""

    # Identifiers
    test_case_id: str = Field(..., description="Unique identifier for this test case")
    source_type: SourceType = Field(..., description="Source of ground truth")

    # OMIP reference (if applicable)
    omip_id: str | None = Field(None, description="OMIP paper ID (e.g., 'OMIP-069')")
    doi: str | None = Field(None, description="DOI of the source paper")

    # FlowRepository reference (if applicable)
    flowrepository_id: str | None = Field(
        None,
        description="FlowRepository ID (e.g., 'FR-FCM-Z7YM')",
    )
    has_wsp: bool = Field(False, description="Whether .wsp file is available")
    wsp_validated: bool = Field(False, description="Whether .wsp was validated")

    # Core data
    context: ExperimentContext = Field(..., description="Experimental context")
    panel: Panel = Field(..., description="Flow cytometry panel")
    gating_hierarchy: GatingHierarchy = Field(..., description="Ground truth hierarchy")

    # Difficulty (optional - computed from hierarchy if not set)
    difficulty: Difficulty | None = Field(
        None,
        description="Test case difficulty based on hierarchy size (easy/medium/hard)",
    )

    # Validation and metadata
    validation: ValidationInfo = Field(
        default_factory=lambda: ValidationInfo(
            paper_source=None,
            wsp_extraction_match=None,
            curator_notes=None,
            validation_date=None,
        ),
        description="Validation information",
    )
    metadata: CurationMetadata = Field(..., description="Curation metadata")

    def get_difficulty(self) -> Difficulty:
        """Get difficulty - use stored value or compute from hierarchy."""
        if self.difficulty is not None:
            return self.difficulty
        return self.gating_hierarchy.compute_difficulty()

    @property
    def complexity(self) -> Complexity:
        """Get complexity level from panel."""
        return self.panel.complexity

    @property
    def n_colors(self) -> int:
        """Get color count from panel."""
        return self.panel.n_colors

    def to_prompt_context(self, level: Literal["minimal", "standard", "rich"] = "standard") -> str:
        """
        Generate context string for LLM prompting.

        Args:
            level: Amount of context to include
                - minimal: Just marker list
                - standard: Markers + sample type + experiment goal
                - rich: Standard + full context

        Returns:
            Formatted context string
        """
        if level == "minimal":
            markers = ", ".join(self.panel.markers)
            return f"Markers: {markers}"

        lines = [
            f"Sample Type: {self.context.sample_type}",
            f"Species: {self.context.species}",
            f"Application: {self.context.application}",
            "",
            "Panel:",
        ]

        for entry in self.panel.entries:
            fluor = entry.fluorophore or "unknown"
            line = f"  - {entry.marker}: {fluor}"
            if entry.clone:
                line += f" (clone: {entry.clone})"
            lines.append(line)

        if level == "rich" and self.context.additional_notes:
            lines.extend(["", f"Notes: {self.context.additional_notes}"])

        return "\n".join(lines)


# Example test case for reference
# Uses HIPC-standardized gating logic:
# - No FSC/SSC lymphocyte gate; go directly to CD3+ from CD45+
# - Negative markers included (T cells = CD3+ CD19-, B cells = CD3- CD19+)
# Reference: https://www.nature.com/articles/srep20686
EXAMPLE_TEST_CASE = TestCase(
    test_case_id="OMIP-069",
    source_type=SourceType.OMIP_PAPER,
    omip_id="OMIP-069",
    doi="10.1002/cyto.a.24213",
    flowrepository_id="FR-FCM-Z7YM",
    has_wsp=False,
    wsp_validated=False,
    difficulty=Difficulty.EASY,
    context=ExperimentContext(
        sample_type="Human PBMC",
        species="human",
        application="Deep immunophenotyping of major immune lineages",
        tissue=None,
        disease_state=None,
        additional_notes=None,
    ),
    panel=Panel(
        entries=[
            PanelEntry(marker="CD3", fluorophore="BUV395", clone="UCHT1", channel=None, vendor=None, cat_number=None),
            PanelEntry(marker="CD4", fluorophore="BUV496", clone="SK3", channel=None, vendor=None, cat_number=None),
            PanelEntry(marker="CD8", fluorophore="BUV661", clone="SK1", channel=None, vendor=None, cat_number=None),
            PanelEntry(marker="CD19", fluorophore="BV605", clone="SJ25C1", channel=None, vendor=None, cat_number=None),
            PanelEntry(marker="CD45", fluorophore="BV421", clone="HI30", channel=None, vendor=None, cat_number=None),
            PanelEntry(marker="Live/Dead", fluorophore="Zombie NIR", clone=None, channel=None, vendor=None, cat_number=None),
        ]
    ),
    gating_hierarchy=GatingHierarchy(
        root=GateNode(
            name="All Events",
            notes=None,
            children=[
                GateNode(
                    name="Time",
                    markers=["Time"],
                    is_critical=True,
                    notes=None,
                    children=[
                        GateNode(
                            name="Singlets",
                            markers=["FSC-A", "FSC-H"],
                            is_critical=True,
                            notes=None,
                            children=[
                                GateNode(
                                    name="Live",
                                    markers=["Zombie NIR"],
                                    marker_logic=[MarkerExpression(marker="Zombie NIR", positive=False, level=None)],
                                    is_critical=True,
                                    notes=None,
                                    children=[
                                        GateNode(
                                            name="CD45+",
                                            markers=["CD45"],
                                            marker_logic=[MarkerExpression(marker="CD45", positive=True, level=None)],
                                            notes="All leukocytes; go directly to lineage markers (no FSC/SSC lymphocyte gate)",
                                            children=[
                                                # Plot CD19 vs CD3 first to separate T and B cells
                                                GateNode(
                                                    name="T cells",
                                                    markers=["CD3", "CD19"],
                                                    marker_logic=[
                                                        MarkerExpression(marker="CD3", positive=True, level=None),
                                                        MarkerExpression(marker="CD19", positive=False, level=None),
                                                    ],
                                                    notes="CD3+ CD19- T lymphocytes",
                                                ),
                                                GateNode(
                                                    name="B cells",
                                                    markers=["CD19", "CD3"],
                                                    marker_logic=[
                                                        MarkerExpression(marker="CD3", positive=False, level=None),
                                                        MarkerExpression(marker="CD19", positive=True, level=None),
                                                    ],
                                                    notes="CD3- CD19+ B lymphocytes (CD20+ also acceptable)",
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
    ),
    metadata=CurationMetadata(
        curation_date=date.today(),
        curator="John Balibalos",
        flowkit_version=None,
        last_updated=None,
    ),
)
