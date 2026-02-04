#!/usr/bin/env python3
"""
Generate Alien Cell v2 test cases - VALID reasoning vs memorization test.

The key insight: we must TELL the model what names to use, then see if it
follows instructions or defaults to memorized immunology terms.

Test Design:
1. Provide a "lab naming convention" in the prompt
2. Ask model to use THOSE names in its hierarchy
3. If model outputs standard names anyway → memorization dominates
4. If model uses provided names → can follow instructions / reason

This is fundamentally different from v1, which just changed ground truth
labels without informing the model (invalid test).
"""

import json
import random
from copy import deepcopy
from pathlib import Path

# Same alien names as v1
ALIEN_NAMES = [
    "Glorp Cells", "Zorphan Population", "Quixar Subset", "Blemfark Cells",
    "Vexilon Population", "Wumbus Cells", "Flonkian Subset", "Grimblex Cells",
    "Snorfalax Population", "Plixnor Cells", "Zymbrix Subset", "Krondule Cells",
    "Frimblax Population", "Qwelnar Cells", "Blixtar Subset", "Morglian Cells",
    "Threxix Population", "Vondrak Cells", "Splonix Subset", "Flurbigan Cells",
    "Zanthrex Population", "Nimbular Cells", "Prexilox Subset", "Grindlax Cells",
]

# QC gates to preserve (don't need alien names)
PRESERVE_NAMES = {"All Events", "Singlets", "Lymphocytes", "Live Cells", "Live"}


def extract_gates_with_markers(node: dict, gates: list = None) -> list:
    """Extract all gates that have marker logic (biological gates)."""
    if gates is None:
        gates = []

    name = node.get("name", "")
    marker_logic = node.get("marker_logic", "")

    # Only include gates with marker logic and not QC gates
    if marker_logic and name not in PRESERVE_NAMES:
        gates.append({
            "name": name,
            "marker_logic": marker_logic,
            "markers": node.get("markers", [])
        })

    for child in node.get("children", []):
        extract_gates_with_markers(child, gates)

    return gates


def generate_naming_convention(gates: list) -> dict:
    """Generate alien name mapping for biological gates."""
    available_names = ALIEN_NAMES.copy()
    random.shuffle(available_names)

    mapping = {}
    for i, gate in enumerate(gates):
        if i < len(available_names):
            mapping[gate["name"]] = {
                "alien_name": available_names[i],
                "marker_logic": gate["marker_logic"]
            }

    return mapping


def format_naming_convention_text(mapping: dict) -> str:
    """Format the naming convention as text for the prompt."""
    lines = [
        "## Lab Naming Convention",
        "",
        "This laboratory uses non-standard population names. You MUST use these exact names in your hierarchy:",
        ""
    ]

    for _original, info in mapping.items():
        lines.append(f"- **{info['alien_name']}**: defined by `{info['marker_logic']}`")

    lines.append("")
    lines.append("Use standard names (Singlets, Lymphocytes, Live) for QC gates only.")
    lines.append("")

    return "\n".join(lines)


def transform_hierarchy(node: dict, mapping: dict) -> dict:
    """Transform hierarchy to use alien names.

    Args:
        node: Hierarchy node
        mapping: Simple mapping of original_name -> alien_name (strings)
    """
    new_node = deepcopy(node)

    name = node.get("name", "")
    if name in mapping:
        new_node["name"] = mapping[name]  # mapping values are strings

    if "children" in new_node and new_node["children"]:
        new_node["children"] = [
            transform_hierarchy(child, mapping)
            for child in new_node["children"]
        ]

    return new_node


def create_alien_v2_test_case(original_path: Path, output_dir: Path) -> dict:
    """Create an Alien Cell v2 test case with naming convention in prompt."""
    with open(original_path) as f:
        original = json.load(f)

    # Extract biological gates
    gates = extract_gates_with_markers(original["gating_hierarchy"]["root"])

    if not gates:
        return None  # Skip if no biological gates

    # Generate naming convention
    mapping = generate_naming_convention(gates)

    # Create simple mapping for hierarchy transform
    simple_mapping = {k: v["alien_name"] for k, v in mapping.items()}

    # Create test case
    alien = deepcopy(original)
    base_id = original["test_case_id"]
    alien["test_case_id"] = f"{base_id}-ALIEN-V2"
    alien["source_type"] = "synthetic"

    # Transform hierarchy
    alien["gating_hierarchy"]["root"] = transform_hierarchy(
        original["gating_hierarchy"]["root"],
        simple_mapping
    )

    # Store the naming convention text (for prompt injection)
    naming_convention_text = format_naming_convention_text(mapping)

    # Metadata
    alien["alien_cell_v2_metadata"] = {
        "version": "2.0",
        "name_mapping": mapping,
        "naming_convention_prompt": naming_convention_text,
        "original_test_case": base_id,
        "transformation_type": "instruction_following_test",
        "hypothesis": (
            "Model is TOLD the naming convention. If it still outputs standard "
            "immunology names, it's ignoring instructions (memorization dominates). "
            "If it uses the provided alien names, it can follow instructions."
        ),
        "validity_note": (
            "Unlike v1, this test is valid because the model has the information "
            "needed to produce the expected output."
        )
    }

    # Update context
    alien["context"]["additional_notes"] = (
        f"ALIEN CELL V2: Model is given naming convention in prompt. "
        f"Tests instruction following vs memorization. Original: {base_id}"
    )

    # Validation
    alien["validation"] = {
        "curator_notes": (
            f"Alien Cell V2 generated from {base_id}. "
            "Naming convention provided in prompt. Valid reasoning test."
        )
    }

    # Write output
    output_path = output_dir / f"{original_path.stem}_alien_v2.json"
    with open(output_path, "w") as f:
        json.dump(alien, f, indent=2)

    return {
        "original": base_id,
        "alien_v2": alien["test_case_id"],
        "transformations": len(mapping),
        "naming_convention_preview": naming_convention_text[:200] + "..."
    }


def main():
    project_root = Path(__file__).parent.parent
    verified_dir = project_root / "data" / "verified"
    output_dir = project_root / "data" / "alien_cell_v2"

    output_dir.mkdir(exist_ok=True)

    print("=== Generating Alien Cell V2 Test Cases ===\n")
    print("Key difference from V1:")
    print("  - V1: Changed ground truth labels (INVALID - model can't know)")
    print("  - V2: Provide naming convention IN PROMPT (VALID - tests instruction following)\n")

    results = []
    for test_file in sorted(verified_dir.glob("*.json")):
        print(f"Processing {test_file.name}...")
        result = create_alien_v2_test_case(test_file, output_dir)
        if result:
            results.append(result)
            print(f"  → {result['alien_v2']} ({result['transformations']} gates renamed)")

    # Write summary
    summary_path = output_dir / "_generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": "2026-02-04",
            "version": "2.0",
            "source": "verified",
            "test_cases": results,
            "total_cases": len(results),
            "purpose": "Valid test of instruction following vs memorization",
            "methodology": (
                "Naming convention is provided in the prompt. Model must use "
                "the given alien names. Failure to do so indicates memorization "
                "overriding explicit instructions."
            )
        }, f, indent=2)

    print(f"\nGenerated {len(results)} Alien Cell V2 test cases in {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
