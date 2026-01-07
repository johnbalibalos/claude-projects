"""
Flowkit .wsp parsing validation.

This module validates that flowkit can correctly parse FlowRepository workspace files
and extract gating hierarchies suitable for benchmark ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import flowkit as fk
except ImportError:
    fk = None


def check_flowkit_available() -> bool:
    """Check if flowkit is installed and importable."""
    return fk is not None


def validate_wsp_parsing(wsp_path: str | Path) -> dict[str, Any]:
    """
    Validate that a .wsp file can be parsed by flowkit.

    Args:
        wsp_path: Path to the workspace file

    Returns:
        Dictionary with parsing results and metadata
    """
    if not check_flowkit_available():
        return {
            "success": False,
            "error": "flowkit not installed. Run: pip install flowkit"
        }

    wsp_path = Path(wsp_path)
    if not wsp_path.exists():
        return {
            "success": False,
            "error": f"File not found: {wsp_path}"
        }

    try:
        # Method 1: Direct parsing with parse_wsp
        wsp_dict = fk.parse_wsp(str(wsp_path))

        result = {
            "success": True,
            "method": "parse_wsp",
            "n_samples": len(wsp_dict.get("samples", {})),
            "sample_names": list(wsp_dict.get("samples", {}).keys())[:5],
            "has_groups": "groups" in wsp_dict,
        }

        # Try to extract hierarchy from first sample
        if result["n_samples"] > 0:
            sample_name = list(wsp_dict["samples"].keys())[0]
            sample_data = wsp_dict["samples"][sample_name]

            if "gating_strategy" in sample_data:
                hierarchy = extract_hierarchy_from_dict(sample_data["gating_strategy"])
                result["hierarchy_extracted"] = True
                result["n_gates"] = len(hierarchy)
                result["sample_hierarchy"] = hierarchy
            else:
                result["hierarchy_extracted"] = False
                result["note"] = "No gating_strategy in sample data"

        return result

    except fk.exceptions.FlowJoWSPParsingError as e:
        return {
            "success": False,
            "error": f"FlowJo parsing error: {e}",
            "error_type": "FlowJoWSPParsingError"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def validate_wsp_with_workspace(wsp_path: str | Path) -> dict[str, Any]:
    """
    Validate using flowkit's Workspace class (more features).

    Args:
        wsp_path: Path to the workspace file

    Returns:
        Dictionary with parsing results
    """
    if not check_flowkit_available():
        return {
            "success": False,
            "error": "flowkit not installed"
        }

    try:
        ws = fk.Workspace(str(wsp_path), ignore_missing_files=True)
        sample_ids = ws.get_sample_ids()

        result = {
            "success": True,
            "method": "Workspace",
            "n_samples": len(sample_ids),
            "sample_ids": sample_ids[:5],
        }

        if sample_ids:
            first_sample = sample_ids[0]
            gate_ids = ws.get_gate_ids(first_sample)
            result["n_gates"] = len(gate_ids)
            result["gate_ids"] = gate_ids

            # Extract full hierarchy
            hierarchy = extract_hierarchy(ws, first_sample)
            result["hierarchy"] = hierarchy

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def extract_hierarchy_from_dict(gating_strategy: dict) -> dict[str, dict]:
    """
    Convert flowkit gate dict to simple hierarchy.

    Args:
        gating_strategy: Raw gating strategy from parse_wsp

    Returns:
        Dictionary mapping gate names to their properties
    """
    hierarchy = {}

    for gate_id, gate in gating_strategy.items():
        gate_name = getattr(gate, "gate_name", str(gate_id))
        parent = getattr(gate, "parent", None)

        # Extract dimensions (marker names)
        dimensions = []
        if hasattr(gate, "dimensions"):
            for dim in gate.dimensions:
                if hasattr(dim, "id"):
                    dimensions.append(dim.id)
                elif hasattr(dim, "compensation_ref"):
                    dimensions.append(dim.compensation_ref)

        hierarchy[gate_name] = {
            "parent": parent,
            "markers": dimensions,
            "gate_type": type(gate).__name__
        }

    return hierarchy


def extract_hierarchy(workspace: "fk.Workspace", sample_id: str) -> dict[str, dict]:
    """
    Extract gating hierarchy from a Workspace object.

    Args:
        workspace: flowkit Workspace object
        sample_id: Sample identifier

    Returns:
        Dictionary representing the gating hierarchy
    """
    hierarchy = {}
    gate_ids = workspace.get_gate_ids(sample_id)

    for gate_id in gate_ids:
        try:
            gate = workspace.get_gate(sample_id, gate_id)

            dimensions = []
            if hasattr(gate, "dimensions"):
                for dim in gate.dimensions:
                    if hasattr(dim, "id"):
                        dimensions.append(dim.id)

            hierarchy[gate_id] = {
                "parent": getattr(gate, "parent", None),
                "markers": dimensions,
                "gate_type": type(gate).__name__
            }
        except Exception as e:
            hierarchy[gate_id] = {
                "error": str(e)
            }

    return hierarchy


def hierarchy_to_tree(hierarchy: dict[str, dict]) -> dict:
    """
    Convert flat hierarchy dict to nested tree structure.

    Args:
        hierarchy: Flat dictionary from extract_hierarchy

    Returns:
        Nested tree structure suitable for comparison
    """
    # Build parent-child relationships
    children: dict[str | None, list[str]] = {}
    for gate_name, props in hierarchy.items():
        parent = props.get("parent")
        if parent not in children:
            children[parent] = []
        children[parent].append(gate_name)

    def build_tree(gate_name: str | None) -> dict | None:
        if gate_name is None:
            # Root level - return list of root gates as tree
            if None in children:
                return {
                    "name": "root",
                    "children": [build_tree(child) for child in children[None]]
                }
            return None

        node = {
            "name": gate_name,
            "markers": hierarchy.get(gate_name, {}).get("markers", []),
            "gate_type": hierarchy.get(gate_name, {}).get("gate_type"),
        }

        if gate_name in children:
            node["children"] = [build_tree(child) for child in children[gate_name]]

        return node

    return build_tree(None) or {"name": "root", "children": []}


def run_validation(wsp_path: str | Path) -> None:
    """Run validation and print results."""
    print(f"Validating: {wsp_path}")
    print("=" * 60)

    # Test 1: parse_wsp
    print("\n1. Testing parse_wsp...")
    result1 = validate_wsp_parsing(wsp_path)
    print(json.dumps(result1, indent=2, default=str))

    # Test 2: Workspace class
    print("\n2. Testing Workspace class...")
    result2 = validate_wsp_with_workspace(wsp_path)
    print(json.dumps(result2, indent=2, default=str))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"parse_wsp: {'SUCCESS' if result1.get('success') else 'FAILED'}")
    print(f"Workspace: {'SUCCESS' if result2.get('success') else 'FAILED'}")

    if result1.get("success") or result2.get("success"):
        print("\n✓ flowkit can parse this .wsp file - proceed to Phase 1")
    else:
        print("\n✗ flowkit failed to parse - try another file or use Plan B")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python flowkit_validator.py <path_to_wsp_file>")
        print("\nExample:")
        print("  python flowkit_validator.py data/raw/test_workspace.wsp")
        sys.exit(1)

    run_validation(sys.argv[1])
