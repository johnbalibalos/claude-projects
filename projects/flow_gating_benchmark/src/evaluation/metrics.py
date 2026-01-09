"""
Evaluation metrics for gating hierarchy predictions.

Primary Metrics:
- Hierarchy F1: Precision/recall on gate names
- Structure Accuracy: % of parent-child relationships correct
- Critical Gate Recall: % of must-have gates present

Secondary Metrics:
- Hallucination Rate: Gates predicted that don't match markers
- Depth Accuracy: Whether hierarchy depth matches
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from curation.schemas import GatingHierarchy, GateNode, Panel

if TYPE_CHECKING:
    from evaluation.equivalences import AnnotationCapture, EquivalenceRegistry


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single prediction."""

    # Primary metrics
    hierarchy_f1: float = 0.0
    hierarchy_precision: float = 0.0
    hierarchy_recall: float = 0.0
    structure_accuracy: float = 0.0
    critical_gate_recall: float = 0.0

    # Secondary metrics
    hallucination_rate: float = 0.0
    depth_accuracy: float = 0.0

    # Details
    predicted_gates: list[str] = field(default_factory=list)
    ground_truth_gates: list[str] = field(default_factory=list)
    matching_gates: list[str] = field(default_factory=list)
    missing_gates: list[str] = field(default_factory=list)
    extra_gates: list[str] = field(default_factory=list)
    hallucinated_gates: list[str] = field(default_factory=list)
    missing_critical: list[str] = field(default_factory=list)

    # Structure details
    correct_relationships: int = 0
    total_relationships: int = 0
    structure_errors: list[str] = field(default_factory=list)

    # Parse info
    parse_success: bool = True
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hierarchy_f1": self.hierarchy_f1,
            "hierarchy_precision": self.hierarchy_precision,
            "hierarchy_recall": self.hierarchy_recall,
            "structure_accuracy": self.structure_accuracy,
            "critical_gate_recall": self.critical_gate_recall,
            "hallucination_rate": self.hallucination_rate,
            "depth_accuracy": self.depth_accuracy,
            "predicted_gates": self.predicted_gates,
            "ground_truth_gates": self.ground_truth_gates,
            "matching_gates": self.matching_gates,
            "missing_gates": self.missing_gates,
            "extra_gates": self.extra_gates,
            "hallucinated_gates": self.hallucinated_gates,
            "missing_critical": self.missing_critical,
            "correct_relationships": self.correct_relationships,
            "total_relationships": self.total_relationships,
            "structure_errors": self.structure_errors,
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
        }


# Cell type synonyms for semantic matching
# Maps variations to a canonical form
# Note: Keys should include both the original and normalized forms
# since normalize_gate_name may have already been applied
CELL_TYPE_SYNONYMS: dict[str, str] = {
    # T cell variations (include forms after " cells" removal)
    "t cells": "t_cells",
    "t": "t_cells",  # After " cells" removal
    "t-cells": "t_cells",
    "t lymphocytes": "t_cells",
    "t lymphs": "t_cells",
    "cd3+ t cells": "t_cells",
    "cd3+ t": "t_cells",
    "cd3+": "t_cells",  # In context often means T cells
    "cd3+ t cells": "t_cells",
    # B cell variations (include forms after " cells" removal)
    "b cells": "b_cells",
    "b": "b_cells",  # After " cells" removal
    "b-cells": "b_cells",
    "b lymphocytes": "b_cells",
    "b lymphs": "b_cells",
    "cd19+ b cells": "b_cells",
    "cd19+ b": "b_cells",
    "cd19+": "b_cells",  # In context often means B cells
    "cd20+ b cells": "b_cells",
    "cd20+ b": "b_cells",
    "cd20+": "b_cells",
    # NK cell variations
    "nk cells": "nk_cells",
    "nk": "nk_cells",
    "natural killer cells": "nk_cells",
    "natural killer": "nk_cells",
    "cd56+ nk cells": "nk_cells",
    "cd56+ nk": "nk_cells",
    "cd56+cd3-": "nk_cells",
    "cd3- cd56+": "nk_cells",
    "cd3-cd56+": "nk_cells",
    # Monocyte variations
    "monocytes": "monocytes",
    "monos": "monocytes",
    "cd14+ monocytes": "monocytes",
    "cd14+ monos": "monocytes",
    "cd14+": "monocytes",
    # Lymphocyte variations
    "lymphocytes": "lymphocytes",
    "lymphs": "lymphocytes",
    # Singlet variations - CRITICAL FIX
    "singlets": "singlets",
    "singlets (fsc)": "singlets",
    "singlets (ssc)": "singlets",
    "singlets (fsc-a vs fsc-h)": "singlets",
    "singlets (ssc-a vs ssc-h)": "singlets",
    "fsc singlets": "singlets",
    "ssc singlets": "singlets",
    "singlet": "singlets",
    # Live/Dead variations
    "live cells": "live",
    "live": "live",
    "live/dead": "live",
    "viable cells": "live",
    "viable": "live",
    # Leukocyte variations
    "leukocytes": "leukocytes",
    "cd45+ leukocytes": "leukocytes",
    "cd45+": "leukocytes",
    "cd45+ cells": "leukocytes",
    "white blood cells": "leukocytes",
    "wbc": "leukocytes",
    # CD4+ T cell variations
    "cd4+ t cells": "cd4_t_cells",
    "cd4+ t": "cd4_t_cells",
    "cd4 t cells": "cd4_t_cells",
    "helper t cells": "cd4_t_cells",
    "t helper": "cd4_t_cells",
    "th cells": "cd4_t_cells",
    # CD8+ T cell variations
    "cd8+ t cells": "cd8_t_cells",
    "cd8+ t": "cd8_t_cells",
    "cd8 t cells": "cd8_t_cells",
    "cytotoxic t cells": "cd8_t_cells",
    "cytotoxic t": "cd8_t_cells",
    "ctl": "cd8_t_cells",
    # Dendritic cell variations
    "dendritic cells": "dendritic_cells",
    "dc": "dendritic_cells",
    "dcs": "dendritic_cells",
    # Gamma-delta T cell variations
    "gd t cells": "gamma_delta_t",
    "gamma delta t cells": "gamma_delta_t",
    "γδ t cells": "gamma_delta_t",
    "gd t": "gamma_delta_t",
    "gammadelta t": "gamma_delta_t",
    # Regulatory T cell variations
    "tregs": "regulatory_t",
    "treg": "regulatory_t",
    "regulatory t cells": "regulatory_t",
    "regulatory t": "regulatory_t",
    "cd4+cd25+foxp3+": "regulatory_t",
    # Time gate variations
    "time gate": "time",
    "time": "time",
    # All events variations
    "all events": "all_events",
    "all": "all_events",
    "root": "all_events",
    "ungated": "all_events",
}


def normalize_gate_name(name: str) -> str:
    """
    Normalize gate name for comparison.

    Handles common variations in gate naming conventions including:
    - Parenthetical qualifiers: "Singlets (FSC)" -> "singlets"
    - Positive/negative notation: "CD4 positive" -> "cd4+"
    - Common abbreviations: "lymphocytes" -> "lymphs"

    Note: This normalization is intentionally conservative to avoid false
    matches between distinct populations. For example, "CD14+ monocytes"
    and "classical monos" are different populations and should not match
    through this simple text normalization.
    """
    import re

    normalized = name.lower().strip()

    # Remove parenthetical qualifiers like "(FSC)", "(SSC-A vs SSC-H)", etc.
    # But preserve marker expressions like "CD3+" or "CD19-"
    normalized = re.sub(r'\s*\([^)]*\)\s*', ' ', normalized)

    # Remove common suffixes/prefixes
    replacements = [
        (" positive", "+"),
        ("positive", "+"),
        (" negative", "-"),
        ("negative", "-"),
        (" cells", ""),
        (" cell", ""),
        ("lymphocytes", "lymphs"),
        ("monocytes", "monos"),
        ("neutrophils", "neuts"),
    ]

    for old, new in replacements:
        normalized = normalized.replace(old, new)

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    return normalized


def normalize_gate_semantic(name: str) -> str:
    """
    Normalize gate name with semantic synonym matching.

    This is a more aggressive normalization that maps common cell type
    variations to canonical forms. Use this for parent matching where
    "CD3+ T cells" should match "T cells".

    Returns the canonical form if a synonym match is found,
    otherwise returns the basic normalized form.
    """
    import re

    normalized = normalize_gate_name(name)

    # Check for exact match in synonyms first
    if normalized in CELL_TYPE_SYNONYMS:
        return CELL_TYPE_SYNONYMS[normalized]

    # Check if the normalized name contains a known synonym pattern
    # Use word boundary matching for short synonyms to avoid false positives
    for synonym, canonical in CELL_TYPE_SYNONYMS.items():
        # For very short synonyms (1-2 chars), require word boundaries
        if len(synonym) <= 2:
            # Match as whole word or at start/end with non-alpha boundaries
            pattern = r'(^|[^a-z])' + re.escape(synonym) + r'($|[^a-z])'
            if re.search(pattern, normalized):
                return canonical
        elif synonym in normalized:
            return canonical

    return normalized


def extract_gate_names(hierarchy: GatingHierarchy | dict) -> set[str]:
    """
    Extract all gate names from a hierarchy.

    Args:
        hierarchy: GatingHierarchy object or dict representation

    Returns:
        Set of gate names
    """
    gates = set()

    if isinstance(hierarchy, GatingHierarchy):

        def traverse(node: GateNode):
            gates.add(node.name)
            for child in node.children:
                traverse(child)

        traverse(hierarchy.root)
    else:
        # Handle dict representation
        def traverse_dict(node: dict):
            if "name" in node:
                gates.add(node["name"])
            for child in node.get("children", []):
                traverse_dict(child)

        if "root" in hierarchy:
            traverse_dict(hierarchy["root"])
        elif "name" in hierarchy:
            traverse_dict(hierarchy)

    return gates


def extract_parent_map(hierarchy: GatingHierarchy | dict) -> dict[str, str | None]:
    """
    Extract parent-child relationships from hierarchy.

    Args:
        hierarchy: GatingHierarchy object or dict

    Returns:
        Dict mapping gate name to parent name
    """
    parent_map: dict[str, str | None] = {}

    if isinstance(hierarchy, GatingHierarchy):

        def traverse(node: GateNode, parent: str | None = None):
            parent_map[node.name] = parent
            for child in node.children:
                traverse(child, node.name)

        traverse(hierarchy.root)
    else:

        def traverse_dict(node: dict, parent: str | None = None):
            if "name" in node:
                parent_map[node["name"]] = parent
                for child in node.get("children", []):
                    traverse_dict(child, node["name"])

        if "root" in hierarchy:
            traverse_dict(hierarchy["root"])
        elif "name" in hierarchy:
            traverse_dict(hierarchy)

    return parent_map


def get_hierarchy_depth(hierarchy: GatingHierarchy | dict) -> int:
    """Get maximum depth of the hierarchy."""

    def get_depth(node: GateNode | dict, current: int = 0) -> int:
        if isinstance(node, GateNode):
            children = node.children
        else:
            children = node.get("children", [])

        if not children:
            return current

        return max(get_depth(child, current + 1) for child in children)

    if isinstance(hierarchy, GatingHierarchy):
        return get_depth(hierarchy.root)
    elif "root" in hierarchy:
        return get_depth(hierarchy["root"])
    elif "name" in hierarchy:
        return get_depth(hierarchy)
    return 0


def compute_hierarchy_f1(
    predicted: GatingHierarchy | dict,
    ground_truth: GatingHierarchy | dict,
    fuzzy_match: bool = True,
    equivalence_registry: EquivalenceRegistry | None = None,
    annotation_capture: AnnotationCapture | None = None,
    test_case_id: str | None = None,
) -> tuple[float, float, float, list[str], list[str], list[str]]:
    """
    Compute precision, recall, and F1 for gate names.

    Args:
        predicted: Predicted hierarchy
        ground_truth: Ground truth hierarchy
        fuzzy_match: Whether to use fuzzy name matching
        equivalence_registry: Optional registry for enhanced matching
        annotation_capture: Optional capture for near-miss pairs
        test_case_id: Test case ID for annotation capture context

    Returns:
        Tuple of (f1, precision, recall, matching, missing, extra)
    """
    pred_gates = extract_gate_names(predicted)
    gt_gates = extract_gate_names(ground_truth)

    if equivalence_registry is not None:
        # Use enhanced matching with equivalence registry
        matching, missing, extra = _match_with_registry(
            pred_gates=pred_gates,
            gt_gates=gt_gates,
            registry=equivalence_registry,
            annotation_capture=annotation_capture,
            test_case_id=test_case_id,
        )
    elif fuzzy_match:
        # Normalize for comparison (original behavior)
        pred_normalized = {normalize_gate_name(g): g for g in pred_gates}
        gt_normalized = {normalize_gate_name(g): g for g in gt_gates}

        matching_keys = set(pred_normalized.keys()) & set(gt_normalized.keys())
        matching = [pred_normalized[k] for k in matching_keys]
        missing = [gt_normalized[k] for k in set(gt_normalized.keys()) - matching_keys]
        extra = [pred_normalized[k] for k in set(pred_normalized.keys()) - matching_keys]
    else:
        matching = list(pred_gates & gt_gates)
        missing = list(gt_gates - pred_gates)
        extra = list(pred_gates - gt_gates)

    precision = len(matching) / len(pred_gates) if pred_gates else 0.0
    recall = len(matching) / len(gt_gates) if gt_gates else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall, matching, missing, extra


def _match_with_registry(
    pred_gates: set[str],
    gt_gates: set[str],
    registry: EquivalenceRegistry,
    annotation_capture: AnnotationCapture | None = None,
    test_case_id: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Match gates using equivalence registry with optional near-miss capture.

    Args:
        pred_gates: Predicted gate names
        gt_gates: Ground truth gate names
        registry: Equivalence registry for matching
        annotation_capture: Optional capture for near-misses
        test_case_id: Test case ID for context

    Returns:
        Tuple of (matching, missing, extra)
    """
    matching = []
    matched_gt = set()
    matched_pred = set()

    # Build canonical lookup for ground truth
    gt_canonical_map: dict[str, list[str]] = {}
    for gt in gt_gates:
        canonical = registry.get_canonical(gt)
        if canonical not in gt_canonical_map:
            gt_canonical_map[canonical] = []
        gt_canonical_map[canonical].append(gt)

    # Match predictions against ground truth
    for pred in pred_gates:
        pred_canonical = registry.get_canonical(pred)

        if pred_canonical in gt_canonical_map:
            # Found a match
            gt_matches = gt_canonical_map[pred_canonical]
            if gt_matches:
                gt_match = gt_matches[0]  # Take first match
                matching.append(pred)
                matched_gt.add(gt_match)
                matched_pred.add(pred)

    # Identify missing and extra
    missing = [g for g in gt_gates if g not in matched_gt]
    extra = [g for g in pred_gates if g not in matched_pred]

    # Capture near-misses for unmatched pairs
    if annotation_capture and test_case_id:
        for pred in extra:
            for gt in missing:
                annotation_capture.check_and_capture(
                    predicted=pred,
                    ground_truth=gt,
                    test_case_id=test_case_id,
                    parent_context=None,
                )

    return matching, missing, extra


def compute_structure_accuracy(
    predicted: GatingHierarchy | dict,
    ground_truth: GatingHierarchy | dict,
    use_semantic_matching: bool = True,
) -> tuple[float, int, int, list[str]]:
    """
    Compute accuracy of parent-child relationships.

    Args:
        predicted: Predicted hierarchy
        ground_truth: Ground truth hierarchy
        use_semantic_matching: If True, use semantic normalization that maps
            cell type synonyms to canonical forms (e.g., "CD3+ T cells" matches
            "T cells"). If False, use basic text normalization only.

    Returns:
        Tuple of (accuracy, correct_count, total_count, errors)
    """
    pred_parents = extract_parent_map(predicted)
    gt_parents = extract_parent_map(ground_truth)

    # Only compare gates that exist in both
    common_gates = set(pred_parents.keys()) & set(gt_parents.keys())

    if not common_gates:
        return 0.0, 0, 0, ["No common gates to compare"]

    correct = 0
    errors = []

    # Choose normalization function based on matching mode
    normalize_fn = normalize_gate_semantic if use_semantic_matching else normalize_gate_name

    for gate in common_gates:
        pred_parent = pred_parents.get(gate)
        gt_parent = gt_parents.get(gate)

        # Normalize parent names for comparison
        pred_norm = normalize_fn(pred_parent) if pred_parent else None
        gt_norm = normalize_fn(gt_parent) if gt_parent else None

        if pred_norm == gt_norm:
            correct += 1
        else:
            errors.append(
                f"Gate '{gate}': predicted parent='{pred_parent}', "
                f"expected parent='{gt_parent}'"
            )

    accuracy = correct / len(common_gates)
    return accuracy, correct, len(common_gates), errors


# Default critical gates for PBMC samples - these are always important
# Maps canonical form to list of acceptable variations
DEFAULT_CRITICAL_GATES_CANONICAL = {
    "singlets": ["singlets", "singlet", "singlets (fsc)", "singlets (ssc)"],
    "live": ["live", "live cells", "live/dead", "viable", "viable cells"],
    "lymphocytes": ["lymphocytes", "lymphs"],
}

# Legacy list for backwards compatibility
DEFAULT_CRITICAL_GATES = [
    "singlets",
    "live",
    "live/dead",
    "lymphocytes",
    "lymphs",
    "cd45+",
    "cd45 positive",
]

# Panel-specific critical gates based on markers present
MARKER_CRITICAL_GATES: dict[str, list[str]] = {
    "cd45": ["leukocytes", "cd45+", "cd45+ cells"],
    "cd3": ["t cells", "cd3+", "t lymphocytes"],
    "cd19": ["b cells", "cd19+", "b lymphocytes"],
    "cd20": ["b cells", "cd20+", "b lymphocytes"],
    "cd56": ["nk cells", "cd56+", "natural killer"],
    "cd14": ["monocytes", "cd14+"],
    "cd11c": ["dendritic cells", "myeloid dc"],
}


def derive_panel_critical_gates(panel: Panel | list[dict]) -> list[str]:
    """
    Derive critical gates based on panel markers.

    Args:
        panel: Panel definition with markers

    Returns:
        List of critical gate names relevant to this panel
    """
    # Get panel markers
    if isinstance(panel, Panel):
        panel_markers = {m.lower() for m in panel.markers}
    else:
        panel_markers = {entry["marker"].lower() for entry in panel}

    # Always include core QC gates
    critical = ["singlets", "live"]

    # Add marker-specific critical gates
    for marker, gates in MARKER_CRITICAL_GATES.items():
        if marker in panel_markers:
            # Add the first (most common) gate name
            critical.append(gates[0])

    return critical


def compute_critical_gate_recall(
    predicted: GatingHierarchy | dict,
    ground_truth: GatingHierarchy | dict,
    critical_gates: list[str] | None = None,
    panel: Panel | list[dict] | None = None,
) -> tuple[float, list[str]]:
    """
    Compute recall of critical/must-have gates.

    Uses semantic matching to handle naming variations.

    Args:
        predicted: Predicted hierarchy
        ground_truth: Ground truth hierarchy
        critical_gates: List of critical gate names (optional)
        panel: Panel definition for deriving panel-specific critical gates

    Returns:
        Tuple of (recall, list of missing critical gates)
    """
    pred_gates = extract_gate_names(predicted)
    # Use semantic normalization for better matching
    pred_semantic = {normalize_gate_semantic(g) for g in pred_gates}
    pred_normalized = {normalize_gate_name(g) for g in pred_gates}

    # Get critical gates from ground truth if not specified
    if critical_gates is None:
        if isinstance(ground_truth, GatingHierarchy):
            critical_gates = ground_truth.get_critical_gates()
        else:
            critical_gates = []

    # Fall back to panel-derived or default critical gates
    if not critical_gates:
        if panel is not None:
            critical_gates = derive_panel_critical_gates(panel)
        else:
            # Use ground truth gates that match known critical patterns
            gt_gates = extract_gate_names(ground_truth)
            gt_normalized = {normalize_gate_name(g): g for g in gt_gates}

            critical_gates = [
                gt_normalized[norm]
                for norm in DEFAULT_CRITICAL_GATES
                if norm in gt_normalized
            ]

    if not critical_gates:
        return 1.0, []  # No critical gates defined

    missing = []
    for gate in critical_gates:
        gate_norm = normalize_gate_name(gate)
        gate_semantic = normalize_gate_semantic(gate)

        # Check both normalized and semantic forms
        if gate_norm not in pred_normalized and gate_semantic not in pred_semantic:
            # Also check if any prediction maps to same canonical form
            found = False
            for pred_gate in pred_gates:
                if normalize_gate_semantic(pred_gate) == gate_semantic:
                    found = True
                    break
            if not found:
                missing.append(gate)

    recall = (len(critical_gates) - len(missing)) / len(critical_gates)
    return recall, missing


# Cell populations that imply specific markers
# If a population name matches but required markers aren't in panel, it's likely hallucinated
POPULATION_REQUIRED_MARKERS: dict[str, list[str]] = {
    "regulatory t": ["foxp3", "cd25"],
    "tregs": ["foxp3", "cd25"],
    "th1": ["ifng", "tbet", "ifn-g"],
    "th2": ["il4", "gata3", "il-4"],
    "th17": ["il17", "rorgt", "il-17"],
    "tfh": ["cxcr5", "pd1", "pd-1"],
    "nk cells": ["cd56", "cd16"],
    "nkt": ["cd56", "cd3"],
    "dendritic": ["cd11c", "hla-dr"],
    "plasmacytoid": ["cd123", "cd303"],
    "classical mono": ["cd14", "cd16"],
    "non-classical mono": ["cd14", "cd16"],
    "intermediate mono": ["cd14", "cd16"],
    "memory b": ["cd27", "igd"],
    "naive b": ["cd27", "igd"],
    "plasma": ["cd38", "cd138"],
    "stem": ["cd34"],
    "hematopoietic stem": ["cd34", "cd38"],
}


def compute_hallucination_rate(
    predicted: GatingHierarchy | dict,
    panel: Panel | list[dict],
) -> tuple[float, list[str]]:
    """
    Compute rate of hallucinated gates (markers not in panel).

    A gate is considered hallucinated if it references a marker
    that doesn't exist in the panel.

    Args:
        predicted: Predicted hierarchy
        panel: Panel definition

    Returns:
        Tuple of (hallucination_rate, list of hallucinated gates)
    """
    # Get panel markers
    if isinstance(panel, Panel):
        panel_markers = {m.lower() for m in panel.markers}
    else:
        panel_markers = {entry["marker"].lower() for entry in panel}

    # Add common non-marker dimensions
    panel_markers.update(["fsc-a", "fsc-h", "ssc-a", "ssc-h", "time", "fsc", "ssc"])

    # Extract gates with their markers
    pred_gates = extract_gate_names(predicted)
    hallucinated = []

    # Check if gate names contain markers not in panel
    for gate in pred_gates:
        gate_lower = gate.lower()

        # Skip generic gates
        if any(
            skip in gate_lower
            for skip in ["singlets", "doublets", "live", "dead", "all events", "time", "root"]
        ):
            continue

        # Look for marker references in gate name
        # This is a heuristic - gate names often include marker names
        found_marker = False
        for marker in panel_markers:
            if marker in gate_lower:
                found_marker = True
                break

        # Check for +/- marker references without matching panel markers
        if not found_marker and ("+" in gate or "-" in gate):
            hallucinated.append(gate)
            continue

        # Check for population names that imply specific markers
        for population, required_markers in POPULATION_REQUIRED_MARKERS.items():
            if population in gate_lower:
                # At least one of the required markers should be in the panel
                has_required = any(
                    any(req in pm for pm in panel_markers)
                    for req in required_markers
                )
                if not has_required:
                    hallucinated.append(gate)
                    break

    rate = len(hallucinated) / len(pred_gates) if pred_gates else 0.0
    return rate, hallucinated


def compute_depth_accuracy(
    predicted: GatingHierarchy | dict,
    ground_truth: GatingHierarchy | dict,
) -> float:
    """
    Compute how close predicted depth is to ground truth.

    Returns 1.0 if depths match, decreasing for larger differences.
    """
    pred_depth = get_hierarchy_depth(predicted)
    gt_depth = get_hierarchy_depth(ground_truth)

    if gt_depth == 0:
        return 1.0 if pred_depth == 0 else 0.0

    # Calculate accuracy as 1 - (normalized difference)
    diff = abs(pred_depth - gt_depth)
    return max(0.0, 1.0 - diff / gt_depth)


def evaluate_prediction(
    predicted: GatingHierarchy | dict,
    ground_truth: GatingHierarchy,
    panel: Panel,
    critical_gates: list[str] | None = None,
    equivalence_registry: EquivalenceRegistry | None = None,
    annotation_capture: AnnotationCapture | None = None,
    test_case_id: str | None = None,
) -> EvaluationResult:
    """
    Compute all evaluation metrics for a prediction.

    Args:
        predicted: Predicted hierarchy
        ground_truth: Ground truth GatingHierarchy
        panel: Panel definition
        critical_gates: Optional list of critical gates
        equivalence_registry: Optional registry for enhanced matching
        annotation_capture: Optional capture for near-miss pairs
        test_case_id: Test case ID for annotation capture context

    Returns:
        Complete EvaluationResult
    """
    result = EvaluationResult()

    # Hierarchy F1
    f1, precision, recall, matching, missing, extra = compute_hierarchy_f1(
        predicted,
        ground_truth,
        equivalence_registry=equivalence_registry,
        annotation_capture=annotation_capture,
        test_case_id=test_case_id,
    )
    result.hierarchy_f1 = f1
    result.hierarchy_precision = precision
    result.hierarchy_recall = recall
    result.matching_gates = matching
    result.missing_gates = missing
    result.extra_gates = extra

    # Structure accuracy
    accuracy, correct, total, errors = compute_structure_accuracy(predicted, ground_truth)
    result.structure_accuracy = accuracy
    result.correct_relationships = correct
    result.total_relationships = total
    result.structure_errors = errors

    # Critical gate recall
    critical_recall, missing_critical = compute_critical_gate_recall(
        predicted, ground_truth, critical_gates
    )
    result.critical_gate_recall = critical_recall
    result.missing_critical = missing_critical

    # Hallucination rate
    halluc_rate, hallucinated = compute_hallucination_rate(predicted, panel)
    result.hallucination_rate = halluc_rate
    result.hallucinated_gates = hallucinated

    # Depth accuracy
    result.depth_accuracy = compute_depth_accuracy(predicted, ground_truth)

    # Store gate lists
    result.predicted_gates = list(extract_gate_names(predicted))
    result.ground_truth_gates = list(extract_gate_names(ground_truth))

    return result
