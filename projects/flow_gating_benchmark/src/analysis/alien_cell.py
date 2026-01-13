#!/usr/bin/env python3
"""
Alien Cell Injection Test.

The "kill shot" test for the frequency vs reasoning debate.
Replaces real population names with nonsense tokens while preserving
marker logic to test if the model reasons from markers or memorizes names.

Example:
    Original: CD3+ CD4+ CD25+ → "Regulatory T Cells"
    Alien:    CD3+ CD4+ CD25+ → "Glorp Cells"

If model can identify "Glorp Cells" from markers alone → reasoning
If model fails on "Glorp Cells" → memorization/retrieval dependency
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from curation.schemas import TestCase


ALIEN_CELL_NAMES = [
    "Glorp Cells",
    "Blixon Population",
    "Zythroid Subset",
    "Flumox Lineage",
    "Quandrix Cells",
    "Vexil Population",
    "Droflex Subset",
    "Mortix Lineage",
    "Splynk Cells",
    "Throbin Population",
    "Krynex Cells",
    "Zolvian Subset",
    "Plexor Population",
    "Vyndra Lineage",
    "Thornix Cells",
]


@dataclass
class AlienCellMapping:
    """Mapping between real and alien cell names."""
    original_name: str
    alien_name: str
    marker_logic: str
    depth: int


@dataclass
class AlienCellTestCase:
    """A test case with alien cell names injected."""
    original_test_case: TestCase
    modified_hierarchy: dict  # Modified hierarchy as dict (for JSON serialization)
    mappings: list[AlienCellMapping]
    mapping_context: str  # Context explaining the alien names


@dataclass
class AlienCellResult:
    """Result of an alien cell test run."""
    test_case_id: str
    original_f1: float
    alien_f1: float
    delta_f1: float
    mappings: list[AlienCellMapping]
    reasoning_score: float  # 0-1, higher = more reasoning, less memorization
    interpretation: str


class AlienCellTest:
    """
    The "Alien Cell" injection test.

    Takes a real gating hierarchy and replaces population names with
    nonsense words while preserving the marker logic.
    """

    # Gates that should NOT be renamed (QC gates)
    PRESERVED_PATTERNS = [
        r"^All\s*Events$",
        r"^Time",
        r"^Singlet",
        r"^Live",
        r"^Dead",
        r"^Debris",
        r"^Doublet",
        r"^Viable",
        r"^Lymphocyte",
    ]

    def __init__(
        self,
        alien_names: list[str] | None = None,
        preserve_critical_gates: bool = True,
    ):
        """
        Initialize Alien Cell test.

        Args:
            alien_names: List of nonsense names to use
            preserve_critical_gates: Whether to preserve QC gate names
        """
        self.alien_names = alien_names or ALIEN_CELL_NAMES.copy()
        self.preserve_critical_gates = preserve_critical_gates
        self._name_index = 0

    def _should_preserve(self, gate_name: str) -> bool:
        """Check if a gate name should be preserved."""
        if not self.preserve_critical_gates:
            return False
        for pattern in self.PRESERVED_PATTERNS:
            if re.match(pattern, gate_name, re.IGNORECASE):
                return True
        return False

    def _get_next_alien_name(self) -> str:
        """Get the next alien name from the pool."""
        if self._name_index >= len(self.alien_names):
            suffix = self._name_index // len(ALIEN_CELL_NAMES) + 1
            base_name = ALIEN_CELL_NAMES[self._name_index % len(ALIEN_CELL_NAMES)]
            name = f"{base_name} Type-{suffix}"
        else:
            name = self.alien_names[self._name_index]
        self._name_index += 1
        return name

    def create_alien_hierarchy(
        self,
        hierarchy_dict: dict,
        include_mapping_context: bool = True,
    ) -> tuple[dict, list[AlienCellMapping], str]:
        """
        Create an alien cell version of a hierarchy.

        Args:
            hierarchy_dict: Original hierarchy as dict
            include_mapping_context: Whether to generate mapping context

        Returns:
            Tuple of (modified_hierarchy, mappings, mapping_context)
        """
        self._name_index = 0
        mappings: list[AlienCellMapping] = []

        def transform_gate(gate: dict, depth: int = 0) -> dict:
            """Recursively transform gate names."""
            new_gate = gate.copy()
            name = gate.get("name", "")

            if self._should_preserve(name):
                pass  # Keep original name
            else:
                alien_name = self._get_next_alien_name()
                marker_logic = gate.get("marker_logic_str", "")
                if not marker_logic:
                    markers = gate.get("markers", [])
                    marker_logic = " ".join(markers) if markers else ""

                mappings.append(AlienCellMapping(
                    original_name=name,
                    alien_name=alien_name,
                    marker_logic=marker_logic,
                    depth=depth,
                ))
                new_gate["name"] = alien_name

            # Transform children
            if "children" in gate:
                new_gate["children"] = [
                    transform_gate(child, depth + 1)
                    for child in gate["children"]
                ]

            return new_gate

        # Handle root structure
        if "root" in hierarchy_dict:
            new_hierarchy = {"root": transform_gate(hierarchy_dict["root"])}
        else:
            new_hierarchy = transform_gate(hierarchy_dict)

        # Build mapping context
        mapping_context = ""
        if include_mapping_context:
            mapping_context = self._build_mapping_context(mappings)

        return new_hierarchy, mappings, mapping_context

    def _build_mapping_context(self, mappings: list[AlienCellMapping]) -> str:
        """Build a context string explaining the alien cell mappings."""
        lines = [
            "## Cell Population Definitions",
            "",
            "The following populations are defined by their marker phenotypes:",
            "",
        ]

        for mapping in mappings:
            if mapping.marker_logic:
                lines.append(f"- **{mapping.alien_name}**: {mapping.marker_logic}")
            else:
                lines.append(f"- **{mapping.alien_name}**: (defined by gating position)")

        lines.extend([
            "",
            "Use these marker definitions to construct the gating hierarchy.",
            "",
        ])

        return "\n".join(lines)

    def analyze_result(
        self,
        original_f1: float,
        alien_f1: float,
        mappings: list[AlienCellMapping],
        test_case_id: str,
    ) -> AlienCellResult:
        """
        Analyze the difference between original and alien test results.

        Args:
            original_f1: F1 on original test case
            alien_f1: F1 on alien cell test case
            mappings: The alien cell mappings used
            test_case_id: ID of the test case

        Returns:
            AlienCellResult with analysis
        """
        delta_f1 = original_f1 - alien_f1

        # Calculate reasoning score
        if original_f1 > 0:
            reasoning_score = alien_f1 / original_f1
        else:
            reasoning_score = 1.0 if alien_f1 == 0 else 0.0

        # Clamp to [0, 1]
        reasoning_score = max(0.0, min(1.0, reasoning_score))

        # Generate interpretation
        if delta_f1 < 0.05:
            interpretation = (
                f"REASONING SUPPORTED (Δ={delta_f1:.3f}): "
                "Model performs equally well with alien names, suggesting "
                "it reasons from marker logic rather than population name tokens."
            )
        elif delta_f1 < 0.20:
            interpretation = (
                f"MIXED EVIDENCE (Δ={delta_f1:.3f}): "
                "Moderate performance drop with alien names suggests "
                "partial reliance on both reasoning and token associations."
            )
        else:
            interpretation = (
                f"MEMORIZATION INDICATED (Δ={delta_f1:.3f}): "
                "Large performance drop with alien names suggests "
                "model relies on population name tokens rather than marker logic."
            )

        return AlienCellResult(
            test_case_id=test_case_id,
            original_f1=original_f1,
            alien_f1=alien_f1,
            delta_f1=delta_f1,
            mappings=mappings,
            reasoning_score=reasoning_score,
            interpretation=interpretation,
        )


def run_alien_cell_example():
    """Example usage of the Alien Cell test."""
    # Example hierarchy
    hierarchy = {
        "name": "All Events",
        "children": [
            {
                "name": "Singlets",
                "children": [
                    {
                        "name": "Live cells",
                        "children": [
                            {
                                "name": "T cells",
                                "marker_logic_str": "CD3+ CD19-",
                                "children": [
                                    {"name": "CD4+ T cells", "marker_logic_str": "CD4+ CD8-"},
                                    {"name": "CD8+ T cells", "marker_logic_str": "CD4- CD8+"},
                                ],
                            },
                            {"name": "B cells", "marker_logic_str": "CD3- CD19+"},
                        ],
                    }
                ],
            }
        ],
    }

    test = AlienCellTest()
    new_hierarchy, mappings, context = test.create_alien_hierarchy(hierarchy)

    print("=== ALIEN CELL TEST EXAMPLE ===\n")
    print("MAPPINGS:")
    for m in mappings:
        print(f"  {m.original_name} → {m.alien_name} ({m.marker_logic})")

    print("\nCONTEXT FOR PROMPT:")
    print(context)

    # Simulate results
    result = test.analyze_result(
        original_f1=0.75,
        alien_f1=0.45,
        mappings=mappings,
        test_case_id="OMIP-TEST",
    )
    print("\nRESULT:")
    print(f"  Original F1: {result.original_f1:.3f}")
    print(f"  Alien F1: {result.alien_f1:.3f}")
    print(f"  Delta: {result.delta_f1:.3f}")
    print(f"  Reasoning Score: {result.reasoning_score:.3f}")
    print(f"  {result.interpretation}")


if __name__ == "__main__":
    run_alien_cell_example()
