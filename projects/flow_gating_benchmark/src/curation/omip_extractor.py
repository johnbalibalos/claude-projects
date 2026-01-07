"""
OMIP paper ground truth extraction.

This module provides tools to create ground truth test cases from
OMIP (Optimized Multicolor Immunofluorescence Panel) papers.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .schemas import (
    TestCase,
    SourceType,
    ExperimentContext,
    Panel,
    PanelEntry,
    GatingHierarchy,
    GateNode,
    CurationMetadata,
    ValidationInfo,
)


# Known OMIP papers with key information for curation
KNOWN_OMIPS = {
    "OMIP-069": {
        "title": "40-Color Full Spectrum Flow Cytometry Panel for Deep Immunophenotyping",
        "doi": "10.1002/cyto.a.24213",
        "flowrepository_id": "FR-FCM-Z7YM",
        "sample_type": "Human PBMC",
        "species": "human",
        "application": "Deep immunophenotyping of major immune lineages",
        "n_colors": 40,
        "complexity": "complex",
    },
    "OMIP-058": {
        "title": "30-Color Phenotyping Panel for T, NK, and iNKT Cell Immunomonitoring",
        "doi": "10.1002/cyto.a.23968",
        "flowrepository_id": "FR-FCM-ZYRN",
        "sample_type": "Human PBMC",
        "species": "human",
        "application": "T cell, NK cell, and iNKT cell phenotyping",
        "n_colors": 30,
        "complexity": "complex",
    },
    "OMIP-044": {
        "title": "28-Color Immunophenotyping of the Human Dendritic Cell Compartment",
        "doi": "10.1002/cyto.a.23732",
        "flowrepository_id": "FR-FCM-ZYC2",
        "sample_type": "Human PBMC",
        "species": "human",
        "application": "Dendritic cell phenotyping",
        "n_colors": 28,
        "complexity": "complex",
    },
    "OMIP-043": {
        "title": "25-Color Flow Cytometry Panel for Immunophenotyping of Antibody-Secreting Cells",
        "doi": "10.1002/cyto.a.23700",
        "flowrepository_id": "FR-FCM-ZYBP",
        "sample_type": "Human PBMC",
        "species": "human",
        "application": "Antibody-secreting cell phenotyping",
        "n_colors": 25,
        "complexity": "medium",
    },
    "OMIP-023": {
        "title": "10-Color Leukocyte Immunophenotyping",
        "doi": "10.1002/cyto.a.22591",
        "flowrepository_id": "FR-FCM-ZZ74",
        "sample_type": "Human PBMC",
        "species": "human",
        "application": "Basic leukocyte phenotyping",
        "n_colors": 10,
        "complexity": "simple",
    },
    "OMIP-021": {
        "title": "Innate-like T Cell Immunophenotyping",
        "doi": "10.1002/cyto.a.22555",
        "flowrepository_id": "FR-FCM-ZZ9H",
        "sample_type": "Human PBMC",
        "species": "human",
        "application": "Innate-like T cell phenotyping",
        "n_colors": 12,
        "complexity": "simple",
    },
}


def create_test_case_template(omip_id: str) -> TestCase | None:
    """
    Create a test case template from known OMIP information.

    This provides a starting point that needs to be completed with
    panel and hierarchy information extracted from the paper.

    Args:
        omip_id: OMIP identifier (e.g., "OMIP-069")

    Returns:
        Partial TestCase or None if OMIP not found
    """
    if omip_id not in KNOWN_OMIPS:
        return None

    info = KNOWN_OMIPS[omip_id]

    return TestCase(
        test_case_id=omip_id,
        source_type=SourceType.OMIP_PAPER,
        omip_id=omip_id,
        doi=info["doi"],
        flowrepository_id=info.get("flowrepository_id"),
        has_wsp=False,
        wsp_validated=False,
        context=ExperimentContext(
            sample_type=info["sample_type"],
            species=info["species"],
            application=info["application"],
        ),
        panel=Panel(entries=[]),  # To be filled from paper
        gating_hierarchy=GatingHierarchy(
            root=GateNode(name="All Events", children=[])  # To be filled from paper
        ),
        metadata=CurationMetadata(
            curation_date=date.today(),
            curator="",  # To be filled
        ),
    )


def extract_omip_ground_truth(
    omip_id: str,
    panel_data: list[dict[str, str]],
    hierarchy_data: dict[str, Any],
    curator: str,
    paper_source: str | None = None,
    notes: str | None = None,
) -> TestCase:
    """
    Create a complete test case from OMIP paper data.

    Args:
        omip_id: OMIP identifier
        panel_data: List of panel entries as dicts with marker, fluorophore, clone
        hierarchy_data: Gating hierarchy as nested dict
        curator: Name of the person curating
        paper_source: Source in paper (e.g., "Figure 2")
        notes: Additional curator notes

    Returns:
        Complete TestCase

    Example:
        >>> panel = [
        ...     {"marker": "CD3", "fluorophore": "BUV395", "clone": "UCHT1"},
        ...     {"marker": "CD4", "fluorophore": "BUV496", "clone": "SK3"},
        ... ]
        >>> hierarchy = {
        ...     "name": "All Events",
        ...     "children": [
        ...         {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "children": [...]}
        ...     ]
        ... }
        >>> test_case = extract_omip_ground_truth("OMIP-069", panel, hierarchy, "John")
    """
    # Get template with known OMIP info
    template = create_test_case_template(omip_id)

    if template is None:
        # Create from scratch if not in known list
        template = TestCase(
            test_case_id=omip_id,
            source_type=SourceType.OMIP_PAPER,
            omip_id=omip_id,
            context=ExperimentContext(
                sample_type="Unknown",
                species="unknown",
                application="Unknown",
            ),
            panel=Panel(entries=[]),
            gating_hierarchy=GatingHierarchy(root=GateNode(name="All Events")),
            metadata=CurationMetadata(
                curation_date=date.today(),
                curator=curator,
            ),
        )

    # Convert panel data
    panel_entries = [
        PanelEntry(
            marker=entry["marker"],
            fluorophore=entry["fluorophore"],
            clone=entry.get("clone"),
            channel=entry.get("channel"),
            vendor=entry.get("vendor"),
            cat_number=entry.get("cat_number"),
        )
        for entry in panel_data
    ]

    # Convert hierarchy data
    def dict_to_gate_node(data: dict) -> GateNode:
        return GateNode(
            name=data["name"],
            markers=data.get("markers", []),
            gate_type=data.get("gate_type", "Unknown"),
            is_critical=data.get("is_critical", False),
            notes=data.get("notes"),
            children=[dict_to_gate_node(child) for child in data.get("children", [])],
        )

    hierarchy = GatingHierarchy(root=dict_to_gate_node(hierarchy_data))

    # Update template with extracted data
    return TestCase(
        test_case_id=template.test_case_id,
        source_type=template.source_type,
        omip_id=template.omip_id,
        doi=template.doi,
        flowrepository_id=template.flowrepository_id,
        has_wsp=template.has_wsp,
        wsp_validated=template.wsp_validated,
        context=template.context,
        panel=Panel(entries=panel_entries),
        gating_hierarchy=hierarchy,
        validation=ValidationInfo(
            paper_source=paper_source,
            curator_notes=notes,
        ),
        metadata=CurationMetadata(
            curation_date=date.today(),
            curator=curator,
        ),
    )


def save_test_case(test_case: TestCase, output_dir: str | Path) -> Path:
    """
    Save a test case to JSON file.

    Args:
        test_case: TestCase to save
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{test_case.test_case_id.lower().replace('-', '_')}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(test_case.model_dump(mode="json"), f, indent=2, default=str)

    return output_path


def load_test_case(path: str | Path) -> TestCase:
    """
    Load a test case from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        TestCase object
    """
    with open(path) as f:
        data = json.load(f)

    return TestCase.model_validate(data)


def load_all_test_cases(directory: str | Path) -> list[TestCase]:
    """
    Load all test cases from a directory.

    Args:
        directory: Directory containing test case JSON files

    Returns:
        List of TestCase objects
    """
    directory = Path(directory)
    test_cases = []

    for path in directory.glob("*.json"):
        try:
            test_case = load_test_case(path)
            test_cases.append(test_case)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    return test_cases


def list_known_omips() -> None:
    """Print list of known OMIP papers for curation."""
    print("Known OMIP Papers for Curation")
    print("=" * 60)

    for omip_id, info in KNOWN_OMIPS.items():
        print(f"\n{omip_id}")
        print(f"  Title: {info['title']}")
        print(f"  DOI: {info['doi']}")
        print(f"  FlowRepository: {info.get('flowrepository_id', 'N/A')}")
        print(f"  Complexity: {info['complexity']} ({info['n_colors']} colors)")


if __name__ == "__main__":
    list_known_omips()
