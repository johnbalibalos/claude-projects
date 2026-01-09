#!/usr/bin/env python3
"""
Convert claude-extracted data to ground truth format for experiments.

Takes multi-method extraction JSON files and converts them to the
TestCase format expected by the experiment runner.
"""

import json
from datetime import date
from pathlib import Path


def convert_hierarchy_to_gate_node(hierarchy: dict) -> dict:
    """Convert LLM hierarchy format to GateNode format."""
    if not hierarchy:
        return {
            "name": "All Events",
            "markers": [],
            "marker_logic": [],
            "gate_type": "Unknown",
            "children": [],
            "is_critical": False,
        }

    # Parse marker_logic string into structured format
    marker_logic = []
    marker_logic_str = hierarchy.get("marker_logic", "")
    if marker_logic_str:
        # Parse expressions like "CD3+ CD19-" or "FSC-A vs FSC-H"
        # For now, store as markers list
        pass

    children = []
    for child in hierarchy.get("children", []):
        children.append(convert_hierarchy_to_gate_node(child))

    name = hierarchy.get("name", "Unknown")
    is_critical = name.lower() in ["live cells", "singlets", "lymphocytes", "all events"]

    return {
        "name": name,
        "markers": [],
        "marker_logic": marker_logic,
        "gate_type": "Unknown",
        "children": children,
        "is_critical": is_critical,
    }


def convert_extraction_to_test_case(extraction_data: dict) -> dict:
    """Convert a multi-method extraction to TestCase format."""

    # Get metadata
    omip_id = extraction_data["omip_id"]
    title = extraction_data.get("title", "")

    # Extract species and sample type from title if possible
    title_lower = title.lower()
    if "mouse" in title_lower or "murine" in title_lower:
        species = "mouse"
    elif "human" in title_lower:
        species = "human"
    else:
        species = "unknown"

    if "pbmc" in title_lower:
        sample_type = "PBMC"
    elif "whole blood" in title_lower or "blood" in title_lower:
        sample_type = "whole blood"
    elif "spleen" in title_lower:
        sample_type = "spleen"
    else:
        sample_type = "unknown"

    # Get best panel extraction
    extractions = extraction_data.get("extractions", {})
    panel_extractions = extractions.get("panel", {})

    # Prefer XML, fall back to LLM
    best_panel_method = extraction_data.get("best_extraction", {}).get("panel", "xml")
    if best_panel_method in panel_extractions:
        panel_data = panel_extractions[best_panel_method]["data"]
    elif "xml" in panel_extractions:
        panel_data = panel_extractions["xml"]["data"]
    elif "llm" in panel_extractions:
        panel_data = panel_extractions["llm"]["data"]
    else:
        panel_data = {"entries": []}

    # Convert panel entries to PanelEntry format
    panel_entries = []
    for entry in panel_data.get("entries", []):
        panel_entries.append({
            "marker": entry.get("marker", ""),
            "fluorophore": entry.get("fluorophore", ""),
            "clone": entry.get("clone"),
            "channel": None,
            "vendor": entry.get("vendor"),
            "cat_number": entry.get("cat_number"),
        })

    # Get best hierarchy extraction
    hierarchy_extractions = extractions.get("gating_hierarchy", {})
    best_hierarchy_method = extraction_data.get("best_extraction", {}).get("gating_hierarchy", "llm")

    if best_hierarchy_method in hierarchy_extractions:
        hierarchy_data = hierarchy_extractions[best_hierarchy_method]["data"]
    elif "llm" in hierarchy_extractions:
        hierarchy_data = hierarchy_extractions["llm"]["data"]
    elif "xml" in hierarchy_extractions:
        hierarchy_data = hierarchy_extractions["xml"]["data"]
    else:
        hierarchy_data = {}

    # Convert hierarchy to GatingHierarchy format
    raw_hierarchy = hierarchy_data.get("hierarchy", {"name": "All Events", "children": []})
    root_node = convert_hierarchy_to_gate_node(raw_hierarchy)

    # Build TestCase
    return {
        "test_case_id": omip_id,
        "source_type": "omip_paper",
        "omip_id": omip_id,
        "doi": extraction_data.get("doi"),
        "flowrepository_id": None,
        "has_wsp": False,
        "wsp_validated": False,
        "context": {
            "sample_type": sample_type,
            "species": species,
            "application": title,
            "tissue": None,
            "disease_state": None,
            "additional_notes": None,
        },
        "panel": {
            "entries": panel_entries,
        },
        "gating_hierarchy": {
            "root": root_node,
        },
        "validation": {
            "paper_source": None,
            "wsp_extraction_match": None,
            "discrepancies": [],
            "curator_notes": None,
            "validation_date": None,
        },
        "metadata": {
            "curation_date": str(date.today()),
            "curator": "claude-extraction",
            "flowkit_version": None,
            "last_updated": str(date.today()),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert extractions to ground truth")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "claude-extracted",
        help="Directory with extraction JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "ground_truth",
        help="Directory for ground truth files"
    )
    parser.add_argument(
        "--min-concordance",
        type=float,
        default=0.95,
        help="Minimum panel concordance to include (default: 0.95)"
    )
    parser.add_argument(
        "--omip-ids",
        nargs="+",
        help="Specific OMIP IDs to convert"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find extraction files
    extraction_files = list(args.input_dir.glob("omip_*.json"))

    # Load summary to get concordance scores
    summary_path = args.input_dir / "extraction_summary.json"
    concordance_scores = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            for item in summary.get("successful", []):
                omip_id = item.get("omip_id", "")
                panel_conc = item.get("panel_concordance", [])
                if panel_conc:
                    concordance_scores[omip_id] = panel_conc[0].get("score", 0)

    converted = []
    skipped = []

    for extraction_file in sorted(extraction_files):
        if extraction_file.name == "extraction_summary.json":
            continue

        with open(extraction_file) as f:
            extraction_data = json.load(f)

        omip_id = extraction_data.get("omip_id", "")

        # Check if specific IDs requested
        if args.omip_ids and omip_id not in args.omip_ids:
            continue

        # Check concordance threshold
        concordance = concordance_scores.get(omip_id, 0)

        # Skip if no panel or hierarchy
        extractions = extraction_data.get("extractions", {})
        has_panel = bool(extractions.get("panel", {}))
        has_hierarchy = bool(extractions.get("gating_hierarchy", {}))

        if not has_panel or not has_hierarchy:
            skipped.append(f"{omip_id}: no panel or hierarchy")
            continue

        # Skip if below concordance threshold (unless only LLM extraction)
        panel_methods = list(extractions.get("panel", {}).keys())
        if len(panel_methods) > 1 and concordance < args.min_concordance:
            skipped.append(f"{omip_id}: concordance {concordance:.2f} < {args.min_concordance}")
            continue

        # Convert to TestCase format
        test_case = convert_extraction_to_test_case(extraction_data)

        # Write output
        safe_id = omip_id.lower().replace("-", "_").replace(" ", "_")
        output_path = args.output_dir / f"{safe_id}.json"

        with open(output_path, "w") as f:
            json.dump(test_case, f, indent=2)

        converted.append(f"{omip_id} -> {output_path.name}")

    print(f"Converted {len(converted)} files to {args.output_dir}")
    for item in converted:
        print(f"  ✓ {item}")

    if skipped:
        print(f"\nSkipped {len(skipped)} files:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
