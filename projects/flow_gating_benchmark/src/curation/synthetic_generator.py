"""
Synthetic test case generator for contamination-resistant evaluation.

Generates test case variations from existing templates to:
1. Avoid benchmark contamination from training data
2. Test generalization beyond known OMIP papers
3. Expand test coverage with controlled variations

Usage:
    from curation.synthetic_generator import SyntheticTestCaseGenerator

    gen = SyntheticTestCaseGenerator()
    base = load_test_case("data/ground_truth/omip_074.json")

    # Create variation with marker swaps
    varied = gen.from_template(base, marker_swaps={"CD3": "CD3e"})

    # Perturb hierarchy
    perturbed = gen.perturb_hierarchy(base, remove_gates=["Time gate"])
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date

from .schemas import (
    CurationMetadata,
    GateNode,
    Panel,
    PanelEntry,
    SourceType,
    TestCase,
    ValidationInfo,
)

# Common marker name variations
MARKER_SYNONYMS: dict[str, list[str]] = {
    "CD3": ["CD3e", "CD3epsilon", "CD3-epsilon"],
    "CD4": ["L3T4", "Leu-3a", "OKT4"],
    "CD8": ["Lyt-2", "Leu-2a", "OKT8"],
    "CD19": ["B4", "Leu-12"],
    "CD20": ["B1", "Leu-16", "MS4A1"],
    "CD56": ["NKH-1", "Leu-19", "NCAM"],
    "CD14": ["Mo2", "Leu-M3"],
    "CD45": ["LCA", "T200", "Ly-5"],
    "CD45RA": ["2H4", "Leu-18"],
    "CD45RO": ["UCHL1", "Leu-45RO"],
    "CCR7": ["CD197", "BLR2", "EBI1"],
    "CXCR5": ["CD185", "BLR1"],
    "PD-1": ["CD279", "PDCD1"],
    "CTLA-4": ["CD152"],
    "FoxP3": ["FOXP3", "Foxp3", "foxp3"],
    "Ki-67": ["Ki67", "MKI67", "Mki67"],
}

# Common fluorophore alternatives (similar spectra)
FLUOROPHORE_ALTERNATIVES: dict[str, list[str]] = {
    "FITC": ["Alexa Fluor 488", "AF488", "A488"],
    "PE": ["R-PE", "R-Phycoerythrin"],
    "APC": ["Allophycocyanin"],
    "PE-Cy7": ["PE-Cyanine7"],
    "APC-Cy7": ["APC-Cyanine7", "APC-H7"],
    "BV421": ["Pacific Blue", "eFluor 450"],
    "BV510": ["eFluor 506"],
    "BV605": ["Super Bright 600"],
    "BV711": ["Super Bright 702"],
    "BV785": ["Super Bright 780"],
}

# Standard QC gates that can be added
STANDARD_QC_GATES = [
    GateNode(
        name="Time gate",
        markers=["Time", "SSC-A"],
        is_critical=False,
        notes="Exclude acquisition anomalies",
    ),
    GateNode(
        name="Doublet exclusion (FSC)",
        markers=["FSC-A", "FSC-H"],
        is_critical=True,
        notes="FSC-based singlet gate",
    ),
    GateNode(
        name="Doublet exclusion (SSC)",
        markers=["SSC-A", "SSC-H"],
        is_critical=False,
        notes="SSC-based singlet gate",
    ),
]


@dataclass
class GenerationConfig:
    """Configuration for synthetic generation."""

    seed: int | None = None
    max_marker_swaps: int = 5
    max_gate_changes: int = 3
    preserve_critical: bool = True


class SyntheticTestCaseGenerator:
    """
    Generate synthetic test case variations from templates.

    Supports:
    - Marker name swapping (CD3 -> CD3e)
    - Fluorophore swapping (FITC -> AF488)
    - Gate addition/removal
    - Hierarchy perturbation
    """

    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def from_template(
        self,
        base_case: TestCase,
        marker_swaps: dict[str, str] | None = None,
        fluorophore_swaps: dict[str, str] | None = None,
        new_id_suffix: str = "synthetic",
    ) -> TestCase:
        """
        Create a test case variation with marker/fluorophore substitutions.

        Args:
            base_case: Template test case to modify
            marker_swaps: Dict mapping original marker -> replacement
            fluorophore_swaps: Dict mapping original fluorophore -> replacement
            new_id_suffix: Suffix for new test case ID

        Returns:
            New TestCase with substitutions applied
        """
        # Deep copy to avoid modifying original
        new_case = base_case.model_copy(deep=True)

        # Apply marker swaps to panel
        if marker_swaps:
            new_entries = []
            for entry in new_case.panel.entries:
                new_marker = marker_swaps.get(entry.marker, entry.marker)
                new_entries.append(
                    PanelEntry(
                        marker=new_marker,
                        fluorophore=entry.fluorophore,
                        clone=entry.clone,
                        channel=entry.channel,
                        vendor=entry.vendor,
                        cat_number=entry.cat_number,
                    )
                )
            new_case.panel = Panel(entries=new_entries)

            # Also update hierarchy gate names that reference markers
            self._update_hierarchy_markers(new_case.gating_hierarchy.root, marker_swaps)

        # Apply fluorophore swaps
        if fluorophore_swaps:
            new_entries = []
            for entry in new_case.panel.entries:
                new_fluor = fluorophore_swaps.get(entry.fluorophore, entry.fluorophore)
                new_entries.append(
                    PanelEntry(
                        marker=entry.marker,
                        fluorophore=new_fluor,
                        clone=entry.clone,
                        channel=entry.channel,
                        vendor=entry.vendor,
                        cat_number=entry.cat_number,
                    )
                )
            new_case.panel = Panel(entries=new_entries)

        # Update metadata
        new_case.test_case_id = f"{base_case.test_case_id}_{new_id_suffix}"
        new_case.source_type = SourceType.SYNTHETIC
        new_case.metadata = CurationMetadata(
            curation_date=date.today(),
            curator="synthetic_generator",
            flowkit_version=None,
            last_updated=date.today(),
        )
        new_case.validation = ValidationInfo(
            paper_source=None,
            wsp_extraction_match=None,
            discrepancies=[],
            curator_notes=f"Synthetic variation of {base_case.test_case_id}",
            validation_date=None,
        )

        return new_case

    def _update_hierarchy_markers(
        self,
        node: GateNode,
        marker_swaps: dict[str, str],
    ) -> None:
        """Recursively update marker references in hierarchy."""
        # Update gate name if it contains a swapped marker
        for old, new in marker_swaps.items():
            if old in node.name:
                node.name = node.name.replace(old, new)

        # Update markers list
        node.markers = [marker_swaps.get(m, m) for m in node.markers]

        # Update marker_logic
        for expr in node.marker_logic:
            expr.marker = marker_swaps.get(expr.marker, expr.marker)

        # Recurse to children
        for child in node.children:
            self._update_hierarchy_markers(child, marker_swaps)

    def perturb_hierarchy(
        self,
        base_case: TestCase,
        add_gates: list[GateNode] | None = None,
        remove_gates: list[str] | None = None,
        depth_limit: int | None = None,
    ) -> TestCase:
        """
        Add or remove gates from the hierarchy.

        Args:
            base_case: Template test case
            add_gates: Gates to add (inserted at appropriate position)
            remove_gates: Gate names to remove
            depth_limit: Maximum hierarchy depth to preserve

        Returns:
            New TestCase with modified hierarchy
        """
        new_case = base_case.model_copy(deep=True)
        root = new_case.gating_hierarchy.root

        # Remove specified gates (unless critical and config says preserve)
        if remove_gates:
            self._remove_gates(root, remove_gates)

        # Add specified gates
        if add_gates:
            # Insert after first-level gates
            if root.children:
                first_child = root.children[0]
                for gate in add_gates:
                    gate_copy = gate.model_copy(deep=True)
                    first_child.children.insert(0, gate_copy)

        # Apply depth limit
        if depth_limit is not None:
            self._limit_depth(root, depth_limit, current_depth=0)

        # Update metadata
        new_case.test_case_id = f"{base_case.test_case_id}_perturbed"
        new_case.source_type = SourceType.SYNTHETIC

        return new_case

    def _remove_gates(self, node: GateNode, gates_to_remove: list[str]) -> None:
        """Recursively remove gates by name."""
        # Filter children
        new_children = []
        for child in node.children:
            if child.name in gates_to_remove:
                if child.is_critical and self.config.preserve_critical:
                    new_children.append(child)  # Keep critical gates
                # else: skip this gate (remove it)
            else:
                new_children.append(child)
                self._remove_gates(child, gates_to_remove)

        node.children = new_children

    def _limit_depth(self, node: GateNode, max_depth: int, current_depth: int) -> None:
        """Limit hierarchy to specified depth."""
        if current_depth >= max_depth:
            node.children = []
        else:
            for child in node.children:
                self._limit_depth(child, max_depth, current_depth + 1)

    def random_variation(
        self,
        base_case: TestCase,
        n_marker_swaps: int = 2,
        n_fluorophore_swaps: int = 1,
    ) -> TestCase:
        """
        Create a random variation with automatic swaps.

        Args:
            base_case: Template test case
            n_marker_swaps: Number of random marker swaps
            n_fluorophore_swaps: Number of random fluorophore swaps

        Returns:
            New TestCase with random variations
        """
        marker_swaps = {}
        fluorophore_swaps = {}

        # Find markers in base case that have synonyms
        base_markers = base_case.panel.markers
        swappable_markers = [m for m in base_markers if m in MARKER_SYNONYMS]

        if swappable_markers and n_marker_swaps > 0:
            selected = random.sample(
                swappable_markers,
                min(n_marker_swaps, len(swappable_markers)),
            )
            for marker in selected:
                alternatives = MARKER_SYNONYMS[marker]
                marker_swaps[marker] = random.choice(alternatives)

        # Find fluorophores that have alternatives
        base_fluorophores = [e.fluorophore for e in base_case.panel.entries]
        swappable_fluors = [f for f in base_fluorophores if f in FLUOROPHORE_ALTERNATIVES]

        if swappable_fluors and n_fluorophore_swaps > 0:
            selected = random.sample(
                swappable_fluors,
                min(n_fluorophore_swaps, len(swappable_fluors)),
            )
            for fluor in selected:
                alternatives = FLUOROPHORE_ALTERNATIVES[fluor]
                fluorophore_swaps[fluor] = random.choice(alternatives)

        return self.from_template(
            base_case,
            marker_swaps=marker_swaps,
            fluorophore_swaps=fluorophore_swaps,
            new_id_suffix="random",
        )

    def compose_subset(
        self,
        cases: list[TestCase],
        panel_from: int = 0,
        hierarchy_from: int = 0,
        context_from: int = 0,
    ) -> TestCase:
        """
        Compose a new test case from parts of multiple cases.

        Args:
            cases: List of source test cases
            panel_from: Index of case to take panel from
            hierarchy_from: Index of case to take hierarchy from
            context_from: Index of case to take context from

        Returns:
            New composed TestCase
        """
        if not cases:
            raise ValueError("At least one source case required")

        panel_case = cases[panel_from % len(cases)]
        hierarchy_case = cases[hierarchy_from % len(cases)]
        context_case = cases[context_from % len(cases)]

        return TestCase(
            test_case_id=f"composed_{panel_case.test_case_id}_{hierarchy_case.test_case_id}",
            source_type=SourceType.SYNTHETIC,
            omip_id=None,
            doi=None,
            flowrepository_id=None,
            has_wsp=False,
            wsp_validated=False,
            context=context_case.context.model_copy(),
            panel=panel_case.panel.model_copy(deep=True),
            gating_hierarchy=hierarchy_case.gating_hierarchy.model_copy(deep=True),
            metadata=CurationMetadata(
                curation_date=date.today(),
                curator="synthetic_generator",
                flowkit_version=None,
                last_updated=None,
            ),
            validation=ValidationInfo(
                paper_source=None,
                wsp_extraction_match=None,
                discrepancies=[],
                curator_notes=f"Composed from: panel={panel_case.test_case_id}, "
                f"hierarchy={hierarchy_case.test_case_id}, context={context_case.test_case_id}",
                validation_date=None,
            ),
        )
