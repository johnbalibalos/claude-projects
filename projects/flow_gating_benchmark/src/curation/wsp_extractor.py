"""
Extract gating hierarchies from FlowJo workspace (.wsp) files using FlowKit.

This module converts WSP files to benchmark test case JSON format,
creating ground truth from actual gating strategies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

try:
    import flowkit as fk
except ImportError:
    fk = None


@dataclass
class ExtractionResult:
    """Result of extracting hierarchy from a WSP file."""

    wsp_path: str
    success: bool
    error: str | None = None
    n_samples: int = 0
    sample_used: str | None = None
    n_gates: int = 0
    gate_names: list[str] = field(default_factory=list)
    hierarchy: dict | None = None
    panel_markers: list[str] = field(default_factory=list)


def extract_hierarchy_from_wsp(wsp_path: str | Path) -> ExtractionResult:
    """
    Extract gating hierarchy from a FlowJo workspace file.

    Args:
        wsp_path: Path to the .wsp file

    Returns:
        ExtractionResult with hierarchy data
    """
    if fk is None:
        return ExtractionResult(
            wsp_path=str(wsp_path),
            success=False,
            error="FlowKit not installed. Run: pip install flowkit"
        )

    wsp_path = Path(wsp_path)
    result = ExtractionResult(wsp_path=str(wsp_path), success=False)

    if not wsp_path.exists():
        result.error = f"File not found: {wsp_path}"
        return result

    try:
        # Use parse_wsp to get gating strategies without loading FCS data
        wsp_data = fk.parse_wsp(str(wsp_path))
        samples = wsp_data.get('samples', {})

        result.n_samples = len(samples)

        if not samples:
            result.error = "No samples found in workspace"
            return result

        # Use first sample
        sample_name = list(samples.keys())[0]
        result.sample_used = sample_name

        sample_data = samples[sample_name]
        gating_strategy = sample_data.get('gating_strategy')

        if gating_strategy is None:
            result.error = "No gating strategy found in sample"
            return result

        # Get gate IDs - returns list of (gate_name, gate_path_tuple)
        gate_id_tuples = gating_strategy.get_gate_ids()
        result.n_gates = len(gate_id_tuples)
        result.gate_names = [g[0] for g in gate_id_tuples]

        # Build hierarchy structure from gate paths
        gates_data = {}
        all_markers = set()

        for gate_name, gate_path in gate_id_tuples:
            try:
                gate = gating_strategy.get_gate(gate_name, gate_path)

                # Extract dimensions (markers used for gating)
                dimensions = []
                if hasattr(gate, 'dimensions'):
                    for dim in gate.dimensions:
                        dim_id = getattr(dim, 'id', None)
                        if dim_id:
                            dimensions.append(dim_id)
                            all_markers.add(dim_id)

                # Parent is second-to-last in path (last is 'root')
                parent = gate_path[-1] if len(gate_path) > 1 and gate_path[-1] != 'root' else None

                # Create unique key using full path to handle duplicate names
                gate_key = f"{gate_name}|{'/'.join(gate_path)}"

                gates_data[gate_key] = {
                    "name": gate_name,
                    "parent_path": gate_path,
                    "parent": parent,
                    "markers": dimensions,
                    "gate_type": type(gate).__name__,
                }
            except Exception as e:
                gate_key = f"{gate_name}|error"
                gates_data[gate_key] = {
                    "name": gate_name,
                    "parent": None,
                    "markers": [],
                    "gate_type": "Unknown",
                    "error": str(e)
                }

        # Convert to nested tree using gate paths
        result.hierarchy = build_hierarchy_from_paths(gate_id_tuples, gates_data)
        result.panel_markers = sorted(all_markers)

        result.success = True
        return result

    except Exception as e:
        result.error = f"Failed to parse WSP: {str(e)}"
        return result


def build_hierarchy_from_paths(
    gate_id_tuples: list[tuple[str, tuple]],
    gates_data: dict[str, dict]
) -> dict:
    """
    Build nested hierarchy tree from gate paths.

    Args:
        gate_id_tuples: List of (gate_name, gate_path_tuple) from FlowKit
        gates_data: Dict mapping gate_key to gate data

    Returns:
        Nested tree structure with root node
    """
    # Build parent-child relationships from paths
    # Path example: ('root', 'Time', 'Singlets', 'aAmine-', 'CD3+')
    # This means CD3+ is child of aAmine-, which is child of Singlets, etc.

    # Create nodes dict keyed by full path
    nodes_by_path: dict[tuple, dict] = {}

    for gate_name, gate_path in gate_id_tuples:
        full_path = gate_path + (gate_name,)
        gate_key = f"{gate_name}|{'/'.join(gate_path)}"
        data = gates_data.get(gate_key, {})

        node = {
            "name": gate_name,
            "markers": data.get("markers", []),
            "gate_type": data.get("gate_type", "Unknown"),
            "is_critical": is_critical_gate(gate_name),
            "notes": None,
            "children": [],
            "marker_logic": infer_marker_logic(gate_name, data.get("markers", []))
        }
        nodes_by_path[full_path] = node

    # Now build the tree by connecting parents to children
    for gate_name, gate_path in gate_id_tuples:
        full_path = gate_path + (gate_name,)
        node = nodes_by_path[full_path]

        # Parent path is all but the last element
        if len(gate_path) > 1:  # Has a non-root parent
            parent_path = gate_path
            if parent_path in nodes_by_path:
                nodes_by_path[parent_path]["children"].append(node)

    # Find root-level gates (those whose parent is just ('root',))
    root_children = []
    for gate_name, gate_path in gate_id_tuples:
        if gate_path == ('root',):
            full_path = gate_path + (gate_name,)
            root_children.append(nodes_by_path[full_path])

    # Create root node
    root = {
        "name": "All Events",
        "markers": [],
        "gate_type": "Unknown",
        "is_critical": False,
        "notes": None,
        "children": root_children,
        "marker_logic": []
    }

    return {"root": root}


def build_hierarchy_tree(gates_data: dict[str, dict]) -> dict:
    """
    Convert flat gate dict to nested tree structure (legacy).

    Args:
        gates_data: Dict mapping gate_id to {parent, markers, gate_type}

    Returns:
        Nested tree structure with root node
    """
    # Find children for each gate
    children_map: dict[str | None, list[str]] = {None: []}
    for gate_id, data in gates_data.items():
        parent = data.get("parent")
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(gate_id)

    def build_node(gate_id: str) -> dict:
        """Build a node and its children recursively."""
        data = gates_data.get(gate_id, {})
        node = {
            "name": gate_id,
            "markers": data.get("markers", []),
            "gate_type": data.get("gate_type", "Unknown"),
            "is_critical": is_critical_gate(gate_id),
            "notes": None,
            "children": [],
            "marker_logic": infer_marker_logic(gate_id, data.get("markers", []))
        }

        # Add children
        if gate_id in children_map:
            for child_id in sorted(children_map[gate_id]):
                node["children"].append(build_node(child_id))

        return node

    # Build tree starting from root gates (those with no parent)
    root_gates = children_map.get(None, [])

    if len(root_gates) == 1:
        # Single root - use it directly
        return {"root": build_node(root_gates[0])}
    else:
        # Multiple roots - create synthetic "All Events" root
        root = {
            "name": "All Events",
            "markers": [],
            "gate_type": "Unknown",
            "is_critical": False,
            "notes": None,
            "children": [build_node(g) for g in sorted(root_gates)],
            "marker_logic": []
        }
        return {"root": root}


def is_critical_gate(gate_name: str) -> bool:
    """
    Determine if a gate is critical (must be present).

    Critical gates include: Time, Singlets, Live, CD45+
    """
    name_lower = gate_name.lower()
    critical_patterns = [
        "time",
        "singlet",
        "live",
        "cd45",
        "leukocyte",
    ]
    return any(p in name_lower for p in critical_patterns)


def infer_marker_logic(gate_name: str, markers: list[str]) -> list[dict]:
    """
    Infer marker logic from gate name.

    Examples:
        "CD3+" -> [{"marker": "CD3", "positive": True}]
        "CD3- CD19+" -> [{"marker": "CD3", "positive": False}, {"marker": "CD19", "positive": True}]
        "Live" with "Zombie NIR" -> [{"marker": "Zombie NIR", "positive": False}]
    """
    logic = []

    # Pattern for marker expressions like "CD3+" or "CD19-"
    pattern = r'(CD\d+[a-z]?|HLA-DR|TCR[gd]+|NK\d*\.?\d*|Va\d+|Ja\d+)([+-]|bright|dim|high|low)?'
    matches = re.findall(pattern, gate_name, re.IGNORECASE)

    for marker, modifier in matches:
        if modifier == "-":
            logic.append({"marker": marker, "positive": False, "level": None})
        elif modifier in ["bright", "high"]:
            logic.append({"marker": marker, "positive": True, "level": modifier.lower()})
        elif modifier in ["dim", "low"]:
            logic.append({"marker": marker, "positive": True, "level": modifier.lower()})
        else:
            logic.append({"marker": marker, "positive": True, "level": None})

    # Handle Live/Dead gates
    if "live" in gate_name.lower():
        for marker in markers:
            if any(ld in marker.lower() for ld in ["zombie", "live", "dead", "aqua", "nir", "7-aad", "pi"]):
                logic.append({"marker": marker, "positive": False, "level": None})
                break

    return logic


def create_test_case_json(
    extraction_result: ExtractionResult,
    omip_id: str,
    doi: str,
    flowrepository_id: str,
    context: dict,
    panel_entries: list[dict] | None = None,
) -> dict:
    """
    Create a test case JSON from extraction result.

    Args:
        extraction_result: Result from extract_hierarchy_from_wsp
        omip_id: OMIP paper ID (e.g., "OMIP-069")
        doi: DOI of the paper
        flowrepository_id: FlowRepository ID (e.g., "FR-FCM-Z7YM")
        context: Experiment context dict with sample_type, species, application
        panel_entries: Optional list of panel entries. If None, inferred from markers

    Returns:
        Test case dict ready to be saved as JSON
    """
    if not extraction_result.success or not extraction_result.hierarchy:
        raise ValueError(f"Cannot create test case: {extraction_result.error}")

    # Create panel entries from markers if not provided
    if panel_entries is None:
        panel_entries = []
        for marker in extraction_result.panel_markers:
            # Skip scatter parameters
            if marker.startswith(("FSC", "SSC", "Time")):
                continue
            panel_entries.append({
                "marker": extract_marker_name(marker),
                "fluorophore": extract_fluorophore(marker),
                "clone": None
            })

    test_case = {
        "test_case_id": omip_id,
        "source_type": "flowrepository_wsp",
        "omip_id": omip_id,
        "doi": doi,
        "flowrepository_id": flowrepository_id,
        "has_wsp": True,
        "wsp_validated": True,
        "context": {
            "sample_type": context.get("sample_type", "Unknown"),
            "species": context.get("species", "human"),
            "application": context.get("application", "Flow cytometry analysis"),
            "tissue": context.get("tissue"),
            "disease_state": context.get("disease_state"),
            "additional_notes": f"Extracted from FlowRepository {flowrepository_id}"
        },
        "panel": {
            "entries": panel_entries
        },
        "gating_hierarchy": extraction_result.hierarchy,
        "validation": {
            "paper_source": "FlowRepository WSP",
            "wsp_extraction_match": True,
            "discrepancies": [],
            "curator_notes": f"Automatically extracted using FlowKit from {extraction_result.wsp_path}"
        },
        "metadata": {
            "curation_date": date.today().isoformat(),
            "curator": "FlowKit auto-extraction",
            "flowkit_version": fk.__version__ if fk else None
        }
    }

    return test_case


def extract_marker_name(channel_name: str) -> str:
    """
    Extract marker name from channel name.

    Examples:
        "CD3-BUV395-A" -> "CD3"
        "BV421-A :: CD45" -> "CD45"
        "Comp-BV605-A" -> infer from fluorophore
    """
    # Pattern: marker name followed by fluorophore
    pattern1 = r'^(CD\d+[a-z]?|HLA-DR|TCR[gd]+|NK\d*\.?\d*|FoxP3|Live|Dead)'
    match = re.match(pattern1, channel_name, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern: fluorophore :: marker
    if "::" in channel_name:
        parts = channel_name.split("::")
        return parts[-1].strip()

    # Fallback: return as-is
    return channel_name


def extract_fluorophore(channel_name: str) -> str:
    """
    Extract fluorophore from channel name.

    Examples:
        "CD3-BUV395-A" -> "BUV395"
        "BV421-A :: CD45" -> "BV421"
    """
    fluorophore_patterns = [
        r'(BUV\d+)',
        r'(BV\d+)',
        r'(PE-Cy\d)',
        r'(PerCP-Cy\d\.?\d?)',
        r'(APC-Cy\d)',
        r'(APC-H\d)',
        r'(APC-R\d+)',
        r'(FITC)',
        r'(PE)',
        r'(APC)',
        r'(AF\d+)',
        r'(Zombie\s*\w+)',
    ]

    for pattern in fluorophore_patterns:
        match = re.search(pattern, channel_name, re.IGNORECASE)
        if match:
            return match.group(1)

    return "Unknown"


def batch_extract(
    wsp_dir: str | Path,
    output_dir: str | Path,
    dataset_info: dict[str, dict],
) -> list[dict]:
    """
    Extract hierarchies from multiple WSP files.

    Args:
        wsp_dir: Directory containing WSP files
        output_dir: Directory to save test case JSON files
        dataset_info: Dict mapping omip_id to {doi, flowrepository_id, context}

    Returns:
        List of extraction results
    """
    wsp_dir = Path(wsp_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for wsp_path in wsp_dir.glob("*.wsp"):
        print(f"\nProcessing: {wsp_path.name}")

        # Try to match with dataset info
        omip_id = None
        for oid, info in dataset_info.items():
            if info.get("flowrepository_id", "").lower() in wsp_path.stem.lower():
                omip_id = oid
                break
            if oid.lower().replace("-", "_") in wsp_path.stem.lower():
                omip_id = oid
                break

        if not omip_id:
            print(f"  Warning: No matching dataset info for {wsp_path.name}")
            continue

        info = dataset_info[omip_id]

        # Extract hierarchy
        result = extract_hierarchy_from_wsp(wsp_path)

        if result.success:
            print(f"  ✓ Extracted {result.n_gates} gates from {result.sample_used}")

            # Create test case
            try:
                test_case = create_test_case_json(
                    result,
                    omip_id=omip_id,
                    doi=info.get("doi", ""),
                    flowrepository_id=info.get("flowrepository_id", ""),
                    context=info.get("context", {}),
                )

                # Save to file
                output_path = output_dir / f"{omip_id.lower().replace('-', '_')}.json"
                with open(output_path, "w") as f:
                    json.dump(test_case, f, indent=2)
                print(f"  ✓ Saved: {output_path}")

            except Exception as e:
                print(f"  ✗ Failed to create test case: {e}")
        else:
            print(f"  ✗ Extraction failed: {result.error}")

        results.append({
            "wsp_path": str(wsp_path),
            "omip_id": omip_id,
            "success": result.success,
            "n_gates": result.n_gates,
            "error": result.error,
        })

    return results


def main():
    """CLI entry point for WSP extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract gating hierarchy from WSP files")
    parser.add_argument(
        "wsp_path",
        type=Path,
        help="Path to WSP file or directory containing WSP files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ground_truth/real"),
        help="Output directory for test case JSON files"
    )
    parser.add_argument(
        "--omip-id",
        type=str,
        help="OMIP ID for single file extraction"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed extraction info"
    )

    args = parser.parse_args()

    if args.wsp_path.is_file():
        # Single file extraction
        result = extract_hierarchy_from_wsp(args.wsp_path)

        print(f"\nExtraction Result for: {args.wsp_path}")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Samples: {result.n_samples}")
        print(f"Sample used: {result.sample_used}")
        print(f"Gates found: {result.n_gates}")

        if result.error:
            print(f"Error: {result.error}")

        if args.verbose and result.gate_names:
            print(f"\nGate names:")
            for name in result.gate_names:
                print(f"  - {name}")

        if args.verbose and result.hierarchy:
            print(f"\nHierarchy:")
            print(json.dumps(result.hierarchy, indent=2))

    else:
        print(f"Directory mode not fully implemented yet")
        print(f"Use single WSP file for now")


if __name__ == "__main__":
    main()
