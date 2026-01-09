"""
Marker name normalization and alias handling for concordance calculation.

Flow cytometry markers often have multiple names:
- CD number aliases: B220 = CD45R, ICOS = CD278
- Combined formats: "B220 (CD45R)", "PD-1 (CD279)"

This module provides functions to normalize marker names for fair comparison.
"""

import re
from typing import Set

# Marker alias groups - all names in a group are equivalent
# First name in each group is the canonical form
MARKER_ALIAS_GROUPS = [
    # B cell markers
    ["B220", "CD45R"],

    # T cell activation/checkpoint
    ["ICOS", "CD278"],
    ["PD-1", "CD279"],
    ["PD-L1", "CD274"],
    ["PD-L2", "CD273"],
    ["CTLA-4", "CD152"],

    # Chemokine receptors
    ["CXCR5", "CD185"],
    ["CCR7", "CD197"],
    ["CXCR3", "CD183"],

    # Plasma cell markers
    ["CD138", "SYND-1", "SYNDECAN-1"],

    # MHC markers
    ["MHC CLASS II", "HLA-DR", "I-A/I-E", "MHC-II", "IA-IE"],
]

# Build lookup table: alias -> canonical form
MARKER_TO_CANONICAL = {}
for group in MARKER_ALIAS_GROUPS:
    canonical = group[0]
    for alias in group:
        MARKER_TO_CANONICAL[alias.upper()] = canonical.upper()


def normalize_marker_name(marker: str) -> str:
    """
    Normalize a marker name for comparison.

    - Uppercase
    - Remove special characters (hyphens become consistent)
    - Extract from combined format "Name (Alias)"
    """
    if not marker:
        return ""

    # Uppercase
    marker = marker.upper().strip()

    # Normalize different hyphen types to standard hyphen
    marker = marker.replace("‐", "-").replace("‑", "-").replace("−", "-")

    # Remove extra whitespace
    marker = " ".join(marker.split())

    return marker


def extract_marker_components(marker: str) -> Set[str]:
    """
    Extract all marker name components from a potentially combined name.

    "B220 (CD45R)" -> {"B220", "CD45R"}
    "PD-L2 (CD273)" -> {"PD-L2", "CD273"}
    "CD3" -> {"CD3"}
    """
    marker = normalize_marker_name(marker)
    components = set()

    # Check for parenthetical alias: "Name (Alias)"
    match = re.match(r'^(.+?)\s*\(([^)]+)\)$', marker)
    if match:
        primary = match.group(1).strip()
        alias = match.group(2).strip()
        components.add(primary)
        components.add(alias)
    else:
        components.add(marker)

    return components


def to_canonical_form(marker: str) -> str:
    """Convert a marker name to its canonical form using alias lookup."""
    normalized = normalize_marker_name(marker)
    return MARKER_TO_CANONICAL.get(normalized, normalized)


def get_canonical_markers(markers: Set[str]) -> Set[str]:
    """
    Convert a set of markers to their canonical forms.

    Expands combined names and maps aliases to canonical names.
    Returns a set of normalized, canonical marker names.
    """
    canonical = set()

    for marker in markers:
        # Extract components from combined names like "B220 (CD45R)"
        components = extract_marker_components(marker)

        for comp in components:
            # Map to canonical form
            canon = to_canonical_form(comp)
            canonical.add(canon)

    return canonical


def calculate_marker_concordance_with_aliases(
    markers_a: Set[str],
    markers_b: Set[str]
) -> dict:
    """
    Calculate concordance accounting for aliases and combined names.

    Returns dict with:
    - raw_jaccard: Jaccard without normalization
    - normalized_jaccard: Jaccard after expanding combined names
    - canonical_a: Expanded markers from set A
    - canonical_b: Expanded markers from set B
    """
    # Raw comparison (current behavior)
    if not markers_a and not markers_b:
        raw_jaccard = 1.0
    elif not markers_a or not markers_b:
        raw_jaccard = 0.0
    else:
        raw_intersection = markers_a & markers_b
        raw_union = markers_a | markers_b
        raw_jaccard = len(raw_intersection) / len(raw_union)

    # Normalized comparison
    canonical_a = get_canonical_markers(markers_a)
    canonical_b = get_canonical_markers(markers_b)

    if not canonical_a and not canonical_b:
        normalized_jaccard = 1.0
    elif not canonical_a or not canonical_b:
        normalized_jaccard = 0.0
    else:
        norm_intersection = canonical_a & canonical_b
        norm_union = canonical_a | canonical_b
        normalized_jaccard = len(norm_intersection) / len(norm_union)

    return {
        "raw_jaccard": raw_jaccard,
        "normalized_jaccard": normalized_jaccard,
        "canonical_a": sorted(canonical_a),
        "canonical_b": sorted(canonical_b),
        "canonical_intersection": sorted(canonical_a & canonical_b),
        "canonical_only_a": sorted(canonical_a - canonical_b),
        "canonical_only_b": sorted(canonical_b - canonical_a),
    }


# Test
if __name__ == "__main__":
    # Test with OMIP-076 data
    xml_markers = {
        "B220 (CD45R)", "CD19", "CD25", "CD4", "CD44", "CD45", "CD62L",
        "CD80", "CXCR5 (CD185)", "FOXP3", "ICOS (CD278)", "IGD", "IGM",
        "LIVE/DEAD", "MHC CLASS II (I‐A/I‐E)", "PD‐1 (CD279)",
        "PD‐L2 (CD273)", "SYND‐1 (CD138)", "TCRβ"
    }

    llm_markers = {
        "B220", "CD138", "CD185", "CD19", "CD25", "CD273", "CD278",
        "CD279", "CD4", "CD44", "CD45", "CD45R", "CD62L", "CD80",
        "CXCR5", "FOXP3", "I-A/I-E", "ICOS", "IGD", "IGM", "LIVE/DEAD",
        "MHC CLASS II", "PD-1", "PD-L2", "SYND-1", "TCRβ"
    }

    result = calculate_marker_concordance_with_aliases(xml_markers, llm_markers)

    print(f"Raw Jaccard: {result['raw_jaccard']:.2f}")
    print(f"Normalized Jaccard: {result['normalized_jaccard']:.2f}")
    print(f"\nCanonical intersection: {len(result['canonical_intersection'])} markers")
    print(f"Only in XML (canonical): {result['canonical_only_a']}")
    print(f"Only in LLM (canonical): {result['canonical_only_b']}")
