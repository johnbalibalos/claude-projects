#!/usr/bin/env python3
"""
Generate Alien Cell test cases from verified OMIPs.

The Alien Cell test replaces real cell population names with nonsense words
while preserving marker logic. If a model relies on memorization ("CD4+ means
helper T cells"), it will fail. If it reasons from markers, it should succeed.

Example transformation:
- "CD4+ T cells" with logic "CD4+ CD8-" → "Zorphan Cells" with logic "CD4+ CD8-"
- "Live CD3+ T cells" → "Quixar Population"
"""

import json
import random
import string
from pathlib import Path
from copy import deepcopy

# Nonsense cell names that cannot exist in training data
ALIEN_NAMES = [
    "Glorp Cells",
    "Zorphan Population",
    "Quixar Subset",
    "Blemfark Cells",
    "Vexilon Population",
    "Wumbus Cells",
    "Flonkian Subset",
    "Grimblex Cells",
    "Snorfalax Population",
    "Plixnor Cells",
    "Zymbrix Subset",
    "Krondule Cells",
    "Frimblax Population",
    "Qwelnar Cells",
    "Blixtar Subset",
    "Morglian Cells",
    "Threxix Population",
    "Vondrak Cells",
    "Splonix Subset",
    "Flurbigan Cells",
    "Zanthrex Population",
    "Nimbular Cells",
    "Prexilox Subset",
    "Grindlax Cells",
    "Womblier Population",
    "Crexithan Cells",
    "Blundrix Subset",
    "Vexnorph Cells",
    "Slindergax Population",
    "Frimbuloid Cells",
    "Qwerxilan Subset",
    "Plonkthar Cells",
    "Grexinoid Population",
    "Zimblorian Cells",
    "Flonxitar Subset",
    "Bremulax Cells",
    "Snorgleplex Population",
    "Vrexalion Cells",
    "Plimbaxor Subset",
    "Grondulian Cells",
    "Wexnifar Population",
    "Slorpentine Cells",
    "Blexinova Subset",
    "Frondexian Cells",
    "Quexiblorp Population",
    "Zindralax Cells",
    "Glimphoran Subset",
    "Threxaloid Cells",
    "Vrondilex Population",
    "Splendifax Cells",
    "Kremblaxor Subset",
    "Flondravex Cells",
    "Plexindral Population",
    "Grombifex Cells",
    "Snorplexian Subset",
    "Vrembulax Cells",
    "Zlorpindal Population",
    "Blexitroph Cells",
    "Qwomblinar Subset",
    "Frindolex Cells",
]

# Names to preserve (QC gates that don't test biological knowledge)
PRESERVE_NAMES = {
    "All Events",
    "Singlets",
    "Lymphocytes",
    "Live Cells",
    "Live",
}


def transform_gate_name(original_name: str, name_mapping: dict) -> str:
    """Transform a gate name to an alien version, preserving QC gates."""
    if original_name in PRESERVE_NAMES:
        return original_name

    if original_name not in name_mapping:
        # Pick a random unused alien name
        used_names = set(name_mapping.values())
        available = [n for n in ALIEN_NAMES if n not in used_names]
        if available:
            name_mapping[original_name] = random.choice(available)
        else:
            # Fallback: generate random name
            suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
            name_mapping[original_name] = f"Xenocyte-{suffix} Cells"

    return name_mapping[original_name]


def transform_hierarchy(node: dict, name_mapping: dict) -> dict:
    """Recursively transform gate hierarchy to alien names."""
    new_node = deepcopy(node)

    # Transform the name but preserve marker_logic
    new_node["name"] = transform_gate_name(node["name"], name_mapping)

    # Recursively transform children
    if "children" in new_node and new_node["children"]:
        new_node["children"] = [
            transform_hierarchy(child, name_mapping)
            for child in new_node["children"]
        ]

    return new_node


def create_alien_test_case(original_path: Path, output_dir: Path) -> dict:
    """Create an alien cell version of a test case."""
    with open(original_path) as f:
        original = json.load(f)

    alien = deepcopy(original)
    name_mapping = {}

    # Update identifiers
    base_id = original["test_case_id"]
    alien["test_case_id"] = f"{base_id}-ALIEN"
    alien["source_type"] = "synthetic"  # Use valid enum value

    # Update context to indicate this is an alien test
    if "context" not in alien:
        alien["context"] = {}
    alien["context"]["additional_notes"] = (
        f"ALIEN CELL ABLATION: Population names replaced with nonsense words. "
        f"Marker logic preserved. Tests reasoning vs memorization. "
        f"Original: {base_id}"
    )

    # Transform the gating hierarchy
    if "gating_hierarchy" in alien and "root" in alien["gating_hierarchy"]:
        alien["gating_hierarchy"]["root"] = transform_hierarchy(
            original["gating_hierarchy"]["root"],
            name_mapping
        )

    # Store the mapping for analysis
    alien["alien_cell_metadata"] = {
        "name_mapping": name_mapping,
        "original_test_case": base_id,
        "transformation_type": "population_name_replacement",
        "hypothesis": "If model relies on memorization, performance should drop. "
                     "If model reasons from markers, performance should be maintained."
    }

    # Update validation (create if missing)
    if "validation" not in alien:
        alien["validation"] = {}
    alien["validation"]["curator_notes"] = (
        f"Alien cell ablation generated from {base_id}. "
        "Population names replaced with nonsense words. Marker logic preserved."
    )

    # Write output
    output_path = output_dir / f"{original_path.stem}_alien.json"
    with open(output_path, "w") as f:
        json.dump(alien, f, indent=2)

    return {
        "original": base_id,
        "alien": alien["test_case_id"],
        "transformations": len(name_mapping),
        "mapping": name_mapping,
    }


def main():
    project_root = Path(__file__).parent.parent
    verified_dir = project_root / "data" / "verified"
    alien_dir = project_root / "data" / "alien_cell"

    alien_dir.mkdir(exist_ok=True)

    results = []
    for test_file in sorted(verified_dir.glob("*.json")):
        print(f"Processing {test_file.name}...")
        result = create_alien_test_case(test_file, alien_dir)
        results.append(result)
        print(f"  → {result['alien']} ({result['transformations']} transformations)")

    # Write summary
    summary_path = alien_dir / "_generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": "2026-02-04",
            "source": "verified",
            "test_cases": results,
            "total_cases": len(results),
            "purpose": "Alien Cell ablation tests reasoning vs memorization",
        }, f, indent=2)

    print(f"\nGenerated {len(results)} alien cell test cases in {alien_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
