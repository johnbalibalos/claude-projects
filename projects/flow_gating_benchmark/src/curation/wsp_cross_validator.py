"""
WSP cross-validation with OMIP paper ground truth.

This module validates flowkit-extracted hierarchies against
paper-based ground truth to ensure consistency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import TestCase, GatingHierarchy, GateNode, ValidationInfo

try:
    import flowkit as fk
except ImportError:
    fk = None


@dataclass
class CrossValidationResult:
    """Result of cross-validating WSP against paper ground truth."""

    wsp_path: str
    test_case_id: str
    extraction_success: bool
    extraction_error: str | None = None

    # Gate comparison
    paper_gates: set[str] | None = None
    wsp_gates: set[str] | None = None
    matching_gates: set[str] | None = None
    missing_in_wsp: set[str] | None = None
    extra_in_wsp: set[str] | None = None

    # Structure comparison
    structure_matches: bool | None = None
    structure_differences: list[str] | None = None

    # Summary
    overall_match: bool = False
    match_score: float = 0.0
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "wsp_path": self.wsp_path,
            "test_case_id": self.test_case_id,
            "extraction_success": self.extraction_success,
            "extraction_error": self.extraction_error,
            "paper_gates": list(self.paper_gates) if self.paper_gates else None,
            "wsp_gates": list(self.wsp_gates) if self.wsp_gates else None,
            "matching_gates": list(self.matching_gates) if self.matching_gates else None,
            "missing_in_wsp": list(self.missing_in_wsp) if self.missing_in_wsp else None,
            "extra_in_wsp": list(self.extra_in_wsp) if self.extra_in_wsp else None,
            "structure_matches": self.structure_matches,
            "structure_differences": self.structure_differences,
            "overall_match": self.overall_match,
            "match_score": self.match_score,
            "notes": self.notes,
        }


def extract_wsp_hierarchy(wsp_path: str | Path) -> tuple[dict[str, dict], str | None]:
    """
    Extract gating hierarchy from a WSP file.

    Args:
        wsp_path: Path to the workspace file

    Returns:
        Tuple of (hierarchy dict, error message or None)
    """
    if fk is None:
        return {}, "flowkit not installed"

    try:
        ws = fk.Workspace(str(wsp_path), ignore_missing_files=True)
        sample_ids = ws.get_sample_ids()

        if not sample_ids:
            return {}, "No samples found in workspace"

        # Use first sample
        sample_id = sample_ids[0]
        gate_ids = ws.get_gate_ids(sample_id)

        hierarchy = {}
        for gate_id in gate_ids:
            try:
                gate = ws.get_gate(sample_id, gate_id)
                dimensions = []

                if hasattr(gate, "dimensions"):
                    for dim in gate.dimensions:
                        if hasattr(dim, "id"):
                            dimensions.append(dim.id)

                hierarchy[gate_id] = {
                    "parent": getattr(gate, "parent", None),
                    "markers": dimensions,
                    "gate_type": type(gate).__name__,
                }
            except Exception as e:
                hierarchy[gate_id] = {"error": str(e)}

        return hierarchy, None

    except Exception as e:
        return {}, str(e)


def hierarchy_to_flat(hierarchy: GatingHierarchy) -> dict[str, dict]:
    """
    Convert a GatingHierarchy to flat dict format for comparison.

    Args:
        hierarchy: GatingHierarchy object

    Returns:
        Flat dictionary mapping gate names to properties
    """
    flat: dict[str, dict] = {}

    def traverse(node: GateNode, parent: str | None = None):
        flat[node.name] = {
            "parent": parent,
            "markers": node.markers,
            "gate_type": str(node.gate_type),
        }
        for child in node.children:
            traverse(child, node.name)

    traverse(hierarchy.root)
    return flat


def normalize_gate_name(name: str) -> str:
    """
    Normalize gate name for comparison.

    Handles common variations like:
    - "CD3+" vs "CD3 positive" vs "CD3pos"
    - Case differences
    - Whitespace differences
    """
    normalized = name.lower().strip()

    # Common replacements
    replacements = [
        (" positive", "+"),
        ("positive", "+"),
        (" negative", "-"),
        ("negative", "-"),
        (" pos", "+"),
        (" neg", "-"),
        ("lymphocytes", "lymphs"),
        ("monocytes", "monos"),
    ]

    for old, new in replacements:
        normalized = normalized.replace(old, new)

    # Remove extra whitespace
    normalized = " ".join(normalized.split())

    return normalized


def compare_gate_sets(
    paper_gates: set[str],
    wsp_gates: set[str],
    fuzzy_match: bool = True,
) -> tuple[set[str], set[str], set[str]]:
    """
    Compare two sets of gate names.

    Args:
        paper_gates: Gates from paper ground truth
        wsp_gates: Gates extracted from WSP
        fuzzy_match: Whether to use fuzzy name matching

    Returns:
        Tuple of (matching, missing_in_wsp, extra_in_wsp)
    """
    if not fuzzy_match:
        matching = paper_gates & wsp_gates
        missing = paper_gates - wsp_gates
        extra = wsp_gates - paper_gates
        return matching, missing, extra

    # Fuzzy matching
    paper_normalized = {normalize_gate_name(g): g for g in paper_gates}
    wsp_normalized = {normalize_gate_name(g): g for g in wsp_gates}

    matching_normalized = set(paper_normalized.keys()) & set(wsp_normalized.keys())
    matching = {paper_normalized[n] for n in matching_normalized}

    missing_normalized = set(paper_normalized.keys()) - set(wsp_normalized.keys())
    missing = {paper_normalized[n] for n in missing_normalized}

    extra_normalized = set(wsp_normalized.keys()) - set(paper_normalized.keys())
    extra = {wsp_normalized[n] for n in extra_normalized}

    return matching, missing, extra


def compare_structure(
    paper_hierarchy: dict[str, dict],
    wsp_hierarchy: dict[str, dict],
) -> tuple[bool, list[str]]:
    """
    Compare hierarchical structure (parent-child relationships).

    Args:
        paper_hierarchy: Flat hierarchy from paper
        wsp_hierarchy: Flat hierarchy from WSP

    Returns:
        Tuple of (structures_match, list of differences)
    """
    differences = []

    # Find common gates
    common_gates = set(paper_hierarchy.keys()) & set(wsp_hierarchy.keys())

    for gate in common_gates:
        paper_parent = paper_hierarchy[gate].get("parent")
        wsp_parent = wsp_hierarchy[gate].get("parent")

        if paper_parent != wsp_parent:
            differences.append(
                f"Gate '{gate}': paper parent='{paper_parent}', wsp parent='{wsp_parent}'"
            )

    return len(differences) == 0, differences


def cross_validate_with_wsp(
    test_case: TestCase,
    wsp_path: str | Path,
) -> CrossValidationResult:
    """
    Cross-validate a test case's ground truth with a WSP file.

    Args:
        test_case: TestCase with paper-based ground truth
        wsp_path: Path to WSP file

    Returns:
        CrossValidationResult with comparison details
    """
    result = CrossValidationResult(
        wsp_path=str(wsp_path),
        test_case_id=test_case.test_case_id,
        extraction_success=False,
    )

    # Extract hierarchy from WSP
    wsp_hierarchy, error = extract_wsp_hierarchy(wsp_path)

    if error:
        result.extraction_error = error
        return result

    result.extraction_success = True

    # Get paper hierarchy as flat dict
    paper_hierarchy = hierarchy_to_flat(test_case.gating_hierarchy)

    # Compare gate sets
    paper_gates = set(paper_hierarchy.keys())
    wsp_gates = set(wsp_hierarchy.keys())

    matching, missing, extra = compare_gate_sets(paper_gates, wsp_gates)

    result.paper_gates = paper_gates
    result.wsp_gates = wsp_gates
    result.matching_gates = matching
    result.missing_in_wsp = missing
    result.extra_in_wsp = extra

    # Compare structure
    structure_matches, differences = compare_structure(paper_hierarchy, wsp_hierarchy)
    result.structure_matches = structure_matches
    result.structure_differences = differences

    # Calculate match score
    if paper_gates:
        result.match_score = len(matching) / len(paper_gates)

    # Determine overall match
    result.overall_match = (
        result.match_score >= 0.9
        and structure_matches
        and len(missing) <= 2  # Allow small discrepancies
    )

    # Add notes
    notes = []
    if missing:
        notes.append(f"Missing gates in WSP: {', '.join(sorted(missing)[:5])}")
    if extra:
        notes.append(f"Extra gates in WSP: {', '.join(sorted(extra)[:5])}")
    if differences:
        notes.append(f"Structure differences: {len(differences)}")

    result.notes = notes if notes else None

    return result


def update_test_case_validation(
    test_case: TestCase,
    validation_result: CrossValidationResult,
) -> TestCase:
    """
    Update a test case with WSP validation results.

    Args:
        test_case: Original test case
        validation_result: Cross-validation result

    Returns:
        Updated test case
    """
    # Create updated validation info
    validation = ValidationInfo(
        paper_source=test_case.validation.paper_source,
        wsp_extraction_match=validation_result.overall_match,
        discrepancies=(
            validation_result.notes if validation_result.notes else []
        ),
        curator_notes=test_case.validation.curator_notes,
        validation_date=test_case.validation.validation_date,
    )

    # Return updated test case
    return TestCase(
        test_case_id=test_case.test_case_id,
        source_type=test_case.source_type,
        omip_id=test_case.omip_id,
        doi=test_case.doi,
        flowrepository_id=test_case.flowrepository_id,
        has_wsp=True,
        wsp_validated=validation_result.extraction_success,
        context=test_case.context,
        panel=test_case.panel,
        gating_hierarchy=test_case.gating_hierarchy,
        validation=validation,
        metadata=test_case.metadata,
    )


def batch_cross_validate(
    test_cases_dir: str | Path,
    wsp_dir: str | Path,
    output_path: str | Path | None = None,
) -> list[CrossValidationResult]:
    """
    Cross-validate all test cases against matching WSP files.

    Matches test cases to WSP files by test_case_id or omip_id.

    Args:
        test_cases_dir: Directory containing test case JSON files
        wsp_dir: Directory containing WSP files
        output_path: Optional path to save results

    Returns:
        List of CrossValidationResult
    """
    from .omip_extractor import load_all_test_cases

    test_cases = load_all_test_cases(test_cases_dir)
    wsp_dir = Path(wsp_dir)

    results = []

    for test_case in test_cases:
        # Look for matching WSP file
        possible_names = [
            f"{test_case.test_case_id}.wsp",
            f"{test_case.test_case_id.lower()}.wsp",
            f"{test_case.omip_id}.wsp" if test_case.omip_id else None,
        ]

        wsp_path = None
        for name in possible_names:
            if name and (wsp_dir / name).exists():
                wsp_path = wsp_dir / name
                break

        if wsp_path:
            result = cross_validate_with_wsp(test_case, wsp_path)
            results.append(result)
            print(f"✓ Validated {test_case.test_case_id}: "
                  f"match_score={result.match_score:.2f}")
        else:
            print(f"✗ No WSP found for {test_case.test_case_id}")

    # Save results if output path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python wsp_cross_validator.py <test_case.json> <workspace.wsp>")
        sys.exit(1)

    from .omip_extractor import load_test_case

    test_case = load_test_case(sys.argv[1])
    result = cross_validate_with_wsp(test_case, sys.argv[2])

    print("\nCross-Validation Result")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2))
