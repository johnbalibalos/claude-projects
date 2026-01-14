#!/usr/bin/env python3
"""
Update ground truth test cases to reflect HIPC-standardized gating logic.

Changes:
1. Remove FSC/SSC lymphocyte gate - go directly to lineage markers from CD45+
2. Add marker_logic with positive and negative markers
3. T cells = CD3+ CD19- (when CD19 in panel)
4. B cells = CD3- CD19+ (or CD3- CD20+)
5. NK cells = CD3- CD56+ and/or CD16+

Reference: https://www.nature.com/articles/srep20686
"""

import json
from pathlib import Path
from typing import Any


def has_marker_in_panel(panel: dict, marker: str) -> bool:
    """Check if a marker is present in the panel."""
    entries = panel.get("entries", [])
    marker_lower = marker.lower()
    for entry in entries:
        if entry.get("marker", "").lower() == marker_lower:
            return True
    return False


def create_marker_logic(positive: list[str], negative: list[str] = None) -> list[dict]:
    """Create marker_logic array with positive and negative markers."""
    logic = []
    for marker in positive:
        logic.append({"marker": marker, "positive": True, "level": None})
    if negative:
        for marker in negative:
            logic.append({"marker": marker, "positive": False, "level": None})
    return logic


def update_gate_node(node: dict, panel: dict, parent_name: str | None = None) -> dict:
    """Update a gate node with new marker_logic and restructure if needed."""
    name = node.get("name", "")
    name_lower = name.lower()
    markers = node.get("markers", [])

    # Add marker_logic if not present
    if "marker_logic" not in node:
        node["marker_logic"] = []

    # Update specific populations with proper marker logic
    has_cd19 = has_marker_in_panel(panel, "CD19")
    has_cd20 = has_marker_in_panel(panel, "CD20")
    has_cd3 = has_marker_in_panel(panel, "CD3")
    has_cd56 = has_marker_in_panel(panel, "CD56")
    has_cd16 = has_marker_in_panel(panel, "CD16")

    # T cells: CD3+ and CD19- (if CD19 in panel)
    if "t cell" in name_lower or name_lower == "t cells" or name_lower == "cd3+ t cells":
        positive = ["CD3"]
        negative = []
        if has_cd19:
            negative.append("CD19")
        node["marker_logic"] = create_marker_logic(positive, negative if negative else None)
        if has_cd19 and "CD19" not in markers:
            node["markers"] = ["CD3", "CD19"]
        node["notes"] = "CD3+ T lymphocytes" + (" (CD19- when plotting CD19 vs CD3)" if has_cd19 else "")

    # CD4+ T cells: CD3+ CD4+ CD8-
    elif "cd4+ t" in name_lower or "cd4 t" in name_lower or name_lower == "cd4+ t cells":
        node["marker_logic"] = create_marker_logic(["CD3", "CD4"], ["CD8"])
        node["notes"] = "CD3+ CD4+ CD8- helper T cells"

    # CD8+ T cells: CD3+ CD4- CD8+
    elif "cd8+ t" in name_lower or "cd8 t" in name_lower or name_lower == "cd8+ t cells":
        node["marker_logic"] = create_marker_logic(["CD3", "CD8"], ["CD4"])
        node["notes"] = "CD3+ CD4- CD8+ cytotoxic T cells"

    # B cells: CD3- CD19+ (or CD20+)
    elif "b cell" in name_lower or name_lower == "b cells":
        if has_cd19:
            node["marker_logic"] = create_marker_logic(["CD19"], ["CD3"] if has_cd3 else None)
            node["markers"] = ["CD19", "CD3"] if has_cd3 else ["CD19"]
            node["notes"] = "CD3- CD19+ B lymphocytes" + (" (CD20+ also acceptable)" if has_cd20 else "")
        elif has_cd20:
            node["marker_logic"] = create_marker_logic(["CD20"], ["CD3"] if has_cd3 else None)
            node["markers"] = ["CD20", "CD3"] if has_cd3 else ["CD20"]
            node["notes"] = "CD3- CD20+ B lymphocytes"

    # NK cells: CD3- CD56+ and/or CD16+
    elif "nk cell" in name_lower or name_lower == "nk cells":
        positive = []
        if has_cd56:
            positive.append("CD56")
        if has_cd16:
            positive.append("CD16")
        negative = ["CD3"] if has_cd3 else []
        if positive:
            node["marker_logic"] = create_marker_logic(positive, negative if negative else None)
            node["notes"] = "CD3- " + "/".join(f"{m}+" for m in positive) + " NK cells"

    # CD56bright NK: CD3- CD56bright CD16dim/-
    elif "cd56bright" in name_lower:
        node["marker_logic"] = [
            {"marker": "CD3", "positive": False, "level": None},
            {"marker": "CD56", "positive": True, "level": "bright"},
            {"marker": "CD16", "positive": False, "level": "dim"},
        ]
        node["notes"] = "CD3- CD56bright CD16dim/- regulatory NK cells"

    # CD56dim NK: CD3- CD56dim CD16+
    elif "cd56dim" in name_lower:
        node["marker_logic"] = [
            {"marker": "CD3", "positive": False, "level": None},
            {"marker": "CD56", "positive": True, "level": "dim"},
            {"marker": "CD16", "positive": True, "level": None},
        ]
        node["notes"] = "CD3- CD56dim CD16+ cytotoxic NK cells"

    # Monocytes: CD14+ CD3-
    elif "monocyte" in name_lower or name_lower == "monocytes":
        if has_marker_in_panel(panel, "CD14"):
            negative = ["CD3"] if has_cd3 else []
            node["marker_logic"] = create_marker_logic(["CD14"], negative if negative else None)
            node["notes"] = "CD14+ monocytes" + (" (CD3-)" if has_cd3 else "")

    # Classical monocytes: CD14++ CD16-
    elif "classical monocyte" in name_lower:
        node["marker_logic"] = [
            {"marker": "CD14", "positive": True, "level": "high"},
            {"marker": "CD16", "positive": False, "level": None},
        ]
        node["notes"] = "CD14++ CD16- classical monocytes"

    # Non-classical monocytes: CD14dim CD16++
    elif "non-classical" in name_lower or "nonclassical" in name_lower:
        node["marker_logic"] = [
            {"marker": "CD14", "positive": True, "level": "dim"},
            {"marker": "CD16", "positive": True, "level": "high"},
        ]
        node["notes"] = "CD14dim CD16++ non-classical monocytes"

    # Live cells - negative for viability dye
    elif name_lower == "live" or "live cell" in name_lower:
        # Find the viability dye in markers
        for marker in markers:
            marker_lower = marker.lower()
            if any(dye in marker_lower for dye in ["zombie", "7-aad", "live/dead", "viability", "aqua", "fixable"]):
                node["marker_logic"] = [{"marker": marker, "positive": False, "level": None}]
                node["notes"] = f"{marker} negative = live cells"
                break

    # CD45+ leukocytes
    elif "cd45+" in name_lower or name_lower == "cd45+":
        node["marker_logic"] = create_marker_logic(["CD45"])
        node["notes"] = "All leukocytes; go directly to lineage markers (no FSC/SSC lymphocyte gate per HIPC)"

    # Recursively update children
    new_children = []
    for child in node.get("children", []):
        updated_child = update_gate_node(child, panel, name)
        new_children.append(updated_child)
    node["children"] = new_children

    return node


def remove_lymphocyte_gate(node: dict, panel: dict) -> dict:
    """
    Remove FSC/SSC lymphocyte gate and restructure hierarchy.

    Per HIPC: Don't gate lymphocytes by FSC/SSC; go directly to CD3+.
    Move children of Lymphocytes gate to parent (CD45+ or Live).
    """
    name = node.get("name", "")
    name_lower = name.lower()
    children = node.get("children", [])

    new_children = []
    for child in children:
        child_name = child.get("name", "").lower()
        child_markers = child.get("markers", [])

        # Check if this is an FSC/SSC lymphocyte gate
        is_lymphocyte_gate = (
            "lymphocyte" in child_name and
            any(m.lower() in ["fsc-a", "ssc-a", "fsc", "ssc"] for m in child_markers)
        )

        if is_lymphocyte_gate:
            # Promote children of lymphocyte gate to this level
            grandchildren = child.get("children", [])
            for grandchild in grandchildren:
                # Recursively process and add to new_children
                processed = remove_lymphocyte_gate(grandchild, panel)
                new_children.append(processed)
        else:
            # Keep this child, but recursively process its descendants
            processed = remove_lymphocyte_gate(child, panel)
            new_children.append(processed)

    node["children"] = new_children
    return node


def update_test_case(test_case: dict) -> dict:
    """Update a complete test case with new gating logic."""
    panel = test_case.get("panel", {})
    hierarchy = test_case.get("gating_hierarchy", {})
    root = hierarchy.get("root", {})

    # Step 1: Remove FSC/SSC lymphocyte gates
    root = remove_lymphocyte_gate(root, panel)

    # Step 2: Update marker logic for all gates
    root = update_gate_node(root, panel)

    hierarchy["root"] = root
    test_case["gating_hierarchy"] = hierarchy

    # Add note about HIPC standardization
    validation = test_case.get("validation", {})
    existing_notes = validation.get("curator_notes", "")
    if "HIPC" not in existing_notes:
        validation["curator_notes"] = (
            existing_notes +
            " Updated to HIPC-standardized gating logic (no FSC/SSC lymphocyte gate, negative markers included)."
        ).strip()
    test_case["validation"] = validation

    return test_case


def main():
    """Update all ground truth test cases."""
    ground_truth_dir = Path(__file__).parent.parent / "data" / "verified"

    if not ground_truth_dir.exists():
        print(f"Ground truth directory not found: {ground_truth_dir}")
        return

    json_files = list(ground_truth_dir.glob("*.json"))
    print(f"Found {len(json_files)} test case files")

    for json_file in sorted(json_files):
        print(f"Processing {json_file.name}...")

        with open(json_file, "r") as f:
            test_case = json.load(f)

        updated = update_test_case(test_case)

        with open(json_file, "w") as f:
            json.dump(updated, f, indent=2)

        print(f"  Updated {json_file.name}")

    print(f"\nDone! Updated {len(json_files)} test cases.")


if __name__ == "__main__":
    main()
