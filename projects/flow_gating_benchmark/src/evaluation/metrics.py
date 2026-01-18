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
from typing import TYPE_CHECKING, Any

from .hierarchy import (
    extract_all_parent_relationships,
    extract_gate_names,
    get_hierarchy_depth,
)
from .normalization import (
    is_valid_parent,
    normalize_gate_name,
    normalize_gate_semantic,
)

if TYPE_CHECKING:
    from curation.schemas import GatingHierarchy, Panel
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

    # Schema version for compatibility checking
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_version": self.schema_version,
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


# Default critical gates for PBMC samples (flow cytometry)
DEFAULT_CRITICAL_GATES = ["singlets", "live", "live/dead", "lymphocytes", "lymphs", "cd45+"]

# Default critical gates for mass cytometry (no scatter gates)
DEFAULT_CRITICAL_GATES_CYTOF = ["live", "live/dead", "lymphocytes", "lymphs", "cd45+"]

# Mass cytometry isotope patterns (metal labels instead of fluorophores)
CYTOF_ISOTOPE_PATTERN = r"^\d{2,3}[A-Za-z]{1,2}$"  # e.g., "89Y", "145Nd", "176Yb"

# Panel-specific critical gates based on markers present
# Maps marker (lowercase) to list of acceptable gate names for that population
# First name in list is the canonical form used for reporting
MARKER_CRITICAL_GATES: dict[str, list[str]] = {
    # Major lineage markers
    "cd45": ["leukocytes", "cd45+", "cd45+ cells"],
    "cd3": ["t cells", "cd3+", "t lymphocytes"],
    "cd19": ["b cells", "cd19+", "b lymphocytes"],
    "cd20": ["b cells", "cd20+", "b lymphocytes"],
    "cd56": ["nk cells", "cd56+", "natural killer"],
    "cd14": ["monocytes", "cd14+"],
    "cd11c": ["dendritic cells", "myeloid dc", "dc"],
    # T cell subsets - if marker is in panel, that subset should be gated
    "cd4": ["cd4+ t cells", "cd4+", "helper t", "cd4 t cells"],
    "cd8": ["cd8+ t cells", "cd8+", "cytotoxic t", "cd8 t cells"],
    # Memory/naive markers (when combined with lineage markers)
    "cd45ra": ["naive", "cd45ra+"],
    "cd45ro": ["memory", "cd45ro+"],
    # Regulatory T cell markers
    "foxp3": ["tregs", "regulatory t", "foxp3+"],
    # Myeloid subsets
    "cd123": ["plasmacytoid dc", "pdc", "cd123+"],
    "cd11b": ["myeloid cells", "cd11b+"],
    # Other lymphocyte subsets
    "cd27": ["memory b", "cd27+"],  # Often used for B cell memory
}

# Populations that imply specific markers
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
        test_case_id: Test case ID for annotation context

    Returns:
        Tuple of (f1, precision, recall, matching, missing, extra)
    """
    pred_gates = extract_gate_names(predicted)
    gt_gates = extract_gate_names(ground_truth)

    if equivalence_registry is not None:
        matching, missing, extra = _match_with_registry(
            pred_gates, gt_gates, equivalence_registry, annotation_capture, test_case_id
        )
    elif fuzzy_match:
        # Build mappings that preserve ALL gates, even when normalized forms collide
        # Previously: dict comprehension silently dropped duplicates
        from collections import defaultdict

        pred_by_norm: dict[str, list[str]] = defaultdict(list)
        gt_by_norm: dict[str, list[str]] = defaultdict(list)

        for g in pred_gates:
            pred_by_norm[normalize_gate_name(g)].append(g)
        for g in gt_gates:
            gt_by_norm[normalize_gate_name(g)].append(g)

        matching = []
        missing = []
        extra = []

        # For each normalized form, match as many gates as possible
        all_norm_keys = set(pred_by_norm.keys()) | set(gt_by_norm.keys())
        for norm_key in all_norm_keys:
            pred_list = pred_by_norm.get(norm_key, [])
            gt_list = gt_by_norm.get(norm_key, [])

            # Match min(len(pred), len(gt)) gates
            n_matched = min(len(pred_list), len(gt_list))
            matching.extend(pred_list[:n_matched])

            # Remaining predicted are extra
            extra.extend(pred_list[n_matched:])

            # Remaining ground truth are missing
            missing.extend(gt_list[n_matched:])
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
    """Match gates using equivalence registry."""
    matching = []
    matched_gt = set()
    matched_pred = set()

    gt_canonical_map: dict[str, list[str]] = {}
    for gt in gt_gates:
        canonical = registry.get_canonical(gt)
        if canonical not in gt_canonical_map:
            gt_canonical_map[canonical] = []
        gt_canonical_map[canonical].append(gt)

    for pred in pred_gates:
        pred_canonical = registry.get_canonical(pred)
        if pred_canonical in gt_canonical_map and gt_canonical_map[pred_canonical]:
            gt_match = gt_canonical_map[pred_canonical][0]
            matching.append(pred)
            matched_gt.add(gt_match)
            matched_pred.add(pred)

    missing = [g for g in gt_gates if g not in matched_gt]
    extra = [g for g in pred_gates if g not in matched_pred]

    if annotation_capture and test_case_id:
        for pred in extra:
            for gt in missing:
                annotation_capture.check_and_capture(
                    predicted=pred, ground_truth=gt, test_case_id=test_case_id, parent_context=None
                )

    return matching, missing, extra


def compute_structure_accuracy(
    predicted: GatingHierarchy | dict,
    ground_truth: GatingHierarchy | dict,
    use_semantic_matching: bool = True,
    use_hierarchy: bool = True,
) -> tuple[float, int, int, list[str]]:
    """
    Compute accuracy of parent-child relationships.

    Uses extract_all_parent_relationships() to handle hierarchies with
    duplicate gate names correctly (e.g., "Singlets (FSC)" and "Singlets (SSC)").

    Args:
        predicted: Predicted hierarchy
        ground_truth: Ground truth hierarchy
        use_semantic_matching: If True, use semantic normalization
        use_hierarchy: If True, accept alternative valid parents from CELL_TYPE_HIERARCHY
                      (e.g., "CD4 T cells" under "Lymphocytes" is valid even if
                       ground truth has it under "T cells")

    Returns:
        Tuple of (accuracy, correct_count, total_count, errors)
    """
    # Get all relationships, preserving duplicates
    pred_rels = extract_all_parent_relationships(predicted)
    gt_rels = extract_all_parent_relationships(ground_truth)

    if not gt_rels:
        return 0.0, 0, 0, ["No ground truth relationships to compare"]

    normalize_fn = normalize_gate_semantic if use_semantic_matching else normalize_gate_name

    # Build normalized relationship sets for comparison
    # Use (normalized_gate, normalized_parent, depth) for matching
    def normalize_rel(rel: tuple[str, str | None, int]) -> tuple[str, str | None, int]:
        gate, parent, depth = rel
        return (normalize_fn(gate), normalize_fn(parent) if parent else None, depth)

    pred_normalized = {normalize_rel(r) for r in pred_rels}
    gt_normalized = [normalize_rel(r) for r in gt_rels]

    # Build lookup for predicted parents by (gate, depth)
    pred_parent_lookup: dict[tuple[str, int], str | None] = {}
    for g, p, d in pred_normalized:
        pred_parent_lookup[(g, d)] = p

    correct = 0
    errors = []

    for gt_rel in gt_normalized:
        gt_gate, gt_parent, gt_depth = gt_rel

        # Check for exact match first
        if gt_rel in pred_normalized:
            correct += 1
            continue

        # Check if prediction has this gate at this depth
        pred_parent = pred_parent_lookup.get((gt_gate, gt_depth))

        if pred_parent is not None:
            # Gate exists - check if predicted parent is valid via hierarchy
            if use_hierarchy and gt_gate and pred_parent:
                # Accept if predicted parent is a valid parent according to hierarchy
                if is_valid_parent(gt_gate, pred_parent, use_hierarchy=True):
                    correct += 1
                    continue

            errors.append(
                f"Gate '{gt_gate}' (depth {gt_depth}): "
                f"predicted parent='{pred_parent}', expected='{gt_parent}'"
            )
        else:
            errors.append(
                f"Gate '{gt_gate}' (depth {gt_depth}): "
                f"not found in prediction"
            )

    total = len(gt_normalized)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, errors


def is_mass_cytometry_panel(panel: Panel | list[dict[str, Any]]) -> bool:
    """
    Detect if a panel is mass cytometry (CyTOF) based on fluorophore patterns.

    Mass cytometry uses metal isotopes (e.g., "89Y", "145Nd", "176Yb") instead
    of fluorophores (e.g., "FITC", "PE", "APC").
    """
    import re

    if hasattr(panel, 'entries'):
        entries = panel.entries  # type: ignore[union-attr]
    else:
        entries = panel  # type: ignore[assignment]

    isotope_count = 0
    total_with_fluor = 0

    for entry in entries:
        fluor = entry.get("fluorophore") if isinstance(entry, dict) else getattr(entry, "fluorophore", None)
        if fluor:
            total_with_fluor += 1
            # Check if it matches isotope pattern (e.g., "89Y", "145Nd")
            if re.match(CYTOF_ISOTOPE_PATTERN, str(fluor)):
                isotope_count += 1

    # If majority of fluorophores are isotopes, it's mass cytometry
    return total_with_fluor > 0 and (isotope_count / total_with_fluor) > 0.5


def derive_panel_critical_gates(
    panel: Panel | list[dict[str, Any]],
    technology: str | None = None,
) -> list[str]:
    """
    Derive critical gates based on panel markers.

    Args:
        panel: Panel definition
        technology: Optional override ("flow_cytometry" or "mass_cytometry")
                   If not provided, auto-detected from panel fluorophores.

    Returns:
        List of critical gate names expected in the hierarchy.

    Notes:
        - Flow cytometry: includes singlets (scatter-based doublet exclusion)
        - Mass cytometry (CyTOF): no scatter gates (singlets not required)
    """
    if hasattr(panel, 'markers'):
        panel_markers = {m.lower() for m in panel.markers}  # type: ignore[union-attr]
    else:
        panel_markers = {entry["marker"].lower() for entry in panel}  # type: ignore[index]

    # Auto-detect technology if not provided
    if technology is None:
        is_cytof = is_mass_cytometry_panel(panel)
    else:
        is_cytof = technology.lower() in ("mass_cytometry", "cytof", "mass cytometry")

    # Start with technology-appropriate base gates
    if is_cytof:
        # Mass cytometry: no scatter gates
        critical = ["live"]
    else:
        # Flow cytometry: includes singlets
        critical = ["singlets", "live"]

    for marker, gates in MARKER_CRITICAL_GATES.items():
        if marker in panel_markers:
            critical.append(gates[0])

    return critical


def compute_critical_gate_recall(
    predicted: GatingHierarchy | dict,
    ground_truth: GatingHierarchy | dict,
    critical_gates: list[str] | None = None,
    panel: Panel | list[dict] | None = None,
    technology: str | None = None,
) -> tuple[float, list[str]]:
    """
    Compute recall of critical/must-have gates.

    Args:
        predicted: Predicted hierarchy
        ground_truth: Ground truth hierarchy
        critical_gates: List of critical gate names
        panel: Panel definition for deriving panel-specific critical gates
        technology: Optional technology type ("flow_cytometry" or "mass_cytometry")
                   Affects which gates are considered critical (e.g., singlets for flow only)

    Returns:
        Tuple of (recall, list of missing critical gates)
    """
    pred_gates = extract_gate_names(predicted)
    pred_semantic = {normalize_gate_semantic(g) for g in pred_gates}
    pred_normalized = {normalize_gate_name(g) for g in pred_gates}

    if critical_gates is None:
        if hasattr(ground_truth, 'get_critical_gates'):
            critical_gates = ground_truth.get_critical_gates()  # type: ignore[union-attr]
        else:
            critical_gates = []

    if not critical_gates:
        if panel is not None:
            critical_gates = derive_panel_critical_gates(panel, technology=technology)
        else:
            gt_gates = extract_gate_names(ground_truth)
            gt_normalized = {normalize_gate_name(g): g for g in gt_gates}
            # Use technology-appropriate defaults
            is_cytof = technology and technology.lower() in ("mass_cytometry", "cytof", "mass cytometry")
            default_gates = DEFAULT_CRITICAL_GATES_CYTOF if is_cytof else DEFAULT_CRITICAL_GATES
            critical_gates = [gt_normalized[norm] for norm in default_gates if norm in gt_normalized]

    if not critical_gates:
        return 1.0, []

    missing = []
    for gate in critical_gates:
        gate_norm = normalize_gate_name(gate)
        gate_semantic = normalize_gate_semantic(gate)

        if gate_norm not in pred_normalized and gate_semantic not in pred_semantic:
            found = any(normalize_gate_semantic(pg) == gate_semantic for pg in pred_gates)
            if not found:
                missing.append(gate)

    recall = (len(critical_gates) - len(missing)) / len(critical_gates)
    return recall, missing


def compute_hallucination_rate(
    predicted: GatingHierarchy | dict[str, Any],
    panel: Panel | list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """
    Compute rate of hallucinated gates (markers not in panel).

    Args:
        predicted: Predicted hierarchy
        panel: Panel definition

    Returns:
        Tuple of (hallucination_rate, list of hallucinated gates)
    """
    if hasattr(panel, 'markers'):
        panel_markers = {m.lower() for m in panel.markers}  # type: ignore[union-attr]
    else:
        panel_markers = {entry["marker"].lower() for entry in panel}  # type: ignore[index]

    # Add common non-marker dimensions
    panel_markers.update(["fsc-a", "fsc-h", "ssc-a", "ssc-h", "time", "fsc", "ssc"])

    pred_gates = extract_gate_names(predicted)
    hallucinated = []

    skip_terms = ["singlets", "doublets", "live", "dead", "all events", "time", "root"]

    for gate in pred_gates:
        gate_lower = gate.lower()

        if any(skip in gate_lower for skip in skip_terms):
            continue

        found_marker = any(marker in gate_lower for marker in panel_markers)

        if not found_marker and ("+" in gate or "-" in gate):
            hallucinated.append(gate)
            continue

        for population, required_markers in POPULATION_REQUIRED_MARKERS.items():
            if population in gate_lower:
                has_required = any(any(req in pm for pm in panel_markers) for req in required_markers)
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
        test_case_id: Test case ID for annotation context

    Returns:
        Complete EvaluationResult
    """
    result = EvaluationResult()

    # Hierarchy F1
    f1, precision, recall, matching, missing, extra = compute_hierarchy_f1(
        predicted, ground_truth,
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

    # Critical gate recall - pass panel for marker-based critical gate derivation
    critical_recall, missing_critical = compute_critical_gate_recall(
        predicted, ground_truth, critical_gates, panel=panel
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
