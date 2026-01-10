"""
Enhanced evaluation metrics with improved fuzzy matching and semantic scoring.

Improvements over metrics.py:
1. Better normalization (handles hyphenation, spacing, parentheticals)
2. Semantic equivalences (QC gates, population hierarchies)
3. Marker-based validation (checks if predictions use valid panel markers)

Usage:
    from evaluation.enhanced_metrics import rescore_results, EnhancedEvaluationResult

    enhanced = rescore_results(original_results, ground_truth_dir)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============================================================
# SEMANTIC EQUIVALENCE DEFINITIONS
# ============================================================

# QC gates that should be treated as equivalent
QC_GATE_EQUIVALENCES = {
    'cleaned events', 'singlets', 'singlet', 'time gate', 'time',
    'singlets (fsc)', 'singlets (ssc)', 'singlets (fsc-a vs fsc-h)',
    'qc gate', 'quality control', 'viable cells', 'live cells',
}

# Population parent-child relationships (parent -> children)
POPULATION_HIERARCHY = {
    'monocytes': {
        'classical monocytes', 'intermediate monocytes',
        'non-classical monocytes', 'nonclassical monocytes',
    },
    'non-classical monocytes': {
        'slan+ nonclassical monocytes', 'slan- nonclassical monocytes',
        'slan+ non-classical monocytes', 'slan- non-classical monocytes',
    },
    'nonclassical monocytes': {
        'slan+ nonclassical monocytes', 'slan- nonclassical monocytes',
    },
    't cells': {
        'cd4+ t cells', 'cd8+ t cells', 'gamma delta t cells',
        'regulatory t cells', 'memory t cells', 'naive t cells',
    },
    'b cells': {
        'naive b cells', 'memory b cells', 'plasma cells',
        'class-switched b cells', 'marginal zone b cells',
    },
    'dendritic cells': {
        'myeloid dc', 'plasmacytoid dc', 'cd1c+ mdc', 'cd141+ mdc', 'pdcs',
    },
    'leukocytes': {
        'lymphocytes', 'monocytes', 'granulocytes', 'dendritic cells',
    },
    'lymphocytes': {
        't cells', 'b cells', 'nk cells',
    },
}

# Gates that are semantically equivalent (bidirectional)
SEMANTIC_EQUIVALENCES = [
    # QC variations
    {'cleaned events', 'singlets', 'time gate'},
    {'viable cells', 'live cells', 'live/dead-'},

    # Population naming variations
    {'non-classical monocytes', 'nonclassical monocytes'},
    {'other monocyte subsets', 'non-classical monocytes', 'intermediate monocytes'},
    {'non-granulocytes', 'mononuclear cells', 'pbmc'},

    # Dendritic cells
    {'cd1c+ mdcs', 'cd1c+ myeloid dc', 'mdc1'},
    {'cd141+ mdcs', 'cd141+ myeloid dc', 'mdc2'},
    {'pdcs', 'plasmacytoid dc', 'plasmacytoid dendritic cells'},

    # T cell subsets
    {'gamma delta t cells', 'gd t cells', 'γδ t cells'},
    {'regulatory t cells', 'tregs', 'treg'},
]


def normalize_gate_name_v2(name: str) -> str:
    """
    Enhanced normalization for gate names.

    Handles:
    - Hyphenation variations (non-classical vs nonclassical)
    - Parenthetical content removal
    - Common suffixes (cells, monocytes -> mono)
    - Unicode normalization
    """
    n = name.lower().strip()

    # Unicode normalization (handle special characters)
    n = n.replace('‐', '-').replace('−', '-')  # Different hyphen types
    n = n.replace('γδ', 'gd').replace('γ', 'gamma').replace('δ', 'delta')

    # Remove parenthetical qualifiers
    n = re.sub(r'\s*\([^)]*\)\s*', ' ', n)

    # Normalize spacing and hyphens
    n = n.replace('-', ' ').replace('_', ' ')
    n = re.sub(r'\s+', ' ', n).strip()

    # Common replacements
    replacements = [
        ('non classical', 'nonclassical'),
        ('monocytes', 'mono'),
        ('monocyte', 'mono'),
        ('lymphocytes', 'lymph'),
        ('lymphocyte', 'lymph'),
        (' cells', ''),
        (' cell', ''),
        ('positive', '+'),
        ('negative', '-'),
        (' hi', '+'),
        (' lo', '-'),
        (' high', '+'),
        (' low', '-'),
    ]

    for old, new in replacements:
        n = n.replace(old, new)

    return n.strip()


def find_semantic_equivalence(gate: str) -> set[str]:
    """Find all gates that are semantically equivalent to the given gate."""
    gate_lower = gate.lower().strip()

    equivalents = {gate_lower}

    # Check explicit equivalence sets
    for equiv_set in SEMANTIC_EQUIVALENCES:
        if gate_lower in equiv_set or normalize_gate_name_v2(gate) in {normalize_gate_name_v2(e) for e in equiv_set}:
            equivalents.update(equiv_set)

    # Check QC gates
    if gate_lower in QC_GATE_EQUIVALENCES or normalize_gate_name_v2(gate) in {normalize_gate_name_v2(q) for q in QC_GATE_EQUIVALENCES}:
        equivalents.update(QC_GATE_EQUIVALENCES)

    return equivalents


def is_parent_of(parent: str, child: str) -> bool:
    """Check if parent is a hierarchical parent of child."""
    parent_lower = parent.lower().strip()
    child_lower = child.lower().strip()

    if parent_lower in POPULATION_HIERARCHY:
        children = POPULATION_HIERARCHY[parent_lower]
        # Check direct match
        if child_lower in children:
            return True
        # Check normalized match
        child_norm = normalize_gate_name_v2(child)
        if any(normalize_gate_name_v2(c) == child_norm for c in children):
            return True

    return False


@dataclass
class EnhancedMatch:
    """Represents a match between predicted and ground truth gates."""
    predicted: str
    ground_truth: str
    match_type: str  # 'exact', 'normalized', 'semantic', 'parent', 'child'
    confidence: float = 1.0


@dataclass
class EnhancedEvaluationResult:
    """Enhanced evaluation result with detailed matching information."""

    test_case_id: str
    condition: str

    # Original metrics
    original_f1: float
    original_precision: float
    original_recall: float

    # Enhanced metrics
    enhanced_f1: float
    enhanced_precision: float
    enhanced_recall: float

    # Match details
    matches: list[EnhancedMatch] = field(default_factory=list)
    match_counts: dict[str, int] = field(default_factory=dict)

    # Still unmatched
    missing_gates: list[str] = field(default_factory=list)
    extra_gates: list[str] = field(default_factory=list)

    # Marker-based metrics
    marker_valid_gates: int = 0
    marker_invalid_gates: int = 0
    marker_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'test_case_id': self.test_case_id,
            'condition': self.condition,
            'original_f1': self.original_f1,
            'enhanced_f1': self.enhanced_f1,
            'improvement': self.enhanced_f1 - self.original_f1,
            'match_counts': self.match_counts,
            'missing_gates': self.missing_gates,
            'extra_gates': self.extra_gates,
            'marker_coverage': self.marker_coverage,
        }


def compute_enhanced_matches(
    pred_gates: set[str],
    gt_gates: set[str],
) -> tuple[list[EnhancedMatch], set[str], set[str]]:
    """
    Compute matches using enhanced matching logic.

    Returns:
        Tuple of (matches, unmatched_gt, unmatched_pred)
    """
    matches = []
    matched_gt = set()
    matched_pred = set()

    # Build normalized lookup
    pred_norm = {normalize_gate_name_v2(g): g for g in pred_gates}
    gt_norm = {normalize_gate_name_v2(g): g for g in gt_gates}

    # Pass 1: Exact normalized matches
    for norm, pred in pred_norm.items():
        if norm in gt_norm and gt_norm[norm] not in matched_gt:
            gt = gt_norm[norm]
            match_type = 'exact' if pred.lower() == gt.lower() else 'normalized'
            matches.append(EnhancedMatch(pred, gt, match_type))
            matched_gt.add(gt)
            matched_pred.add(pred)

    # Pass 2: Semantic equivalences
    remaining_pred = pred_gates - matched_pred
    remaining_gt = gt_gates - matched_gt

    for pred in remaining_pred.copy():
        pred_equivs = find_semantic_equivalence(pred)

        for gt in remaining_gt.copy():
            gt_equivs = find_semantic_equivalence(gt)

            # Check if any equivalents match
            if pred_equivs & gt_equivs:
                matches.append(EnhancedMatch(pred, gt, 'semantic'))
                matched_gt.add(gt)
                matched_pred.add(pred)
                remaining_pred.discard(pred)
                remaining_gt.discard(gt)
                break

            # Check normalized equivalents
            pred_norm_equivs = {normalize_gate_name_v2(e) for e in pred_equivs}
            gt_norm_equivs = {normalize_gate_name_v2(e) for e in gt_equivs}

            if pred_norm_equivs & gt_norm_equivs:
                matches.append(EnhancedMatch(pred, gt, 'semantic'))
                matched_gt.add(gt)
                matched_pred.add(pred)
                remaining_pred.discard(pred)
                remaining_gt.discard(gt)
                break

    # Pass 3: Parent-child relationships
    remaining_pred = pred_gates - matched_pred
    remaining_gt = gt_gates - matched_gt

    for pred in remaining_pred.copy():
        for gt in remaining_gt.copy():
            if is_parent_of(pred, gt):
                matches.append(EnhancedMatch(pred, gt, 'parent', confidence=0.7))
                matched_gt.add(gt)
                matched_pred.add(pred)
                remaining_pred.discard(pred)
                remaining_gt.discard(gt)
                break
            elif is_parent_of(gt, pred):
                matches.append(EnhancedMatch(pred, gt, 'child', confidence=0.7))
                matched_gt.add(gt)
                matched_pred.add(pred)
                remaining_pred.discard(pred)
                remaining_gt.discard(gt)
                break

    unmatched_gt = gt_gates - matched_gt
    unmatched_pred = pred_gates - matched_pred

    return matches, unmatched_gt, unmatched_pred


def compute_marker_validation(
    pred_gates: set[str],
    panel_markers: set[str],
) -> tuple[int, int, float]:
    """
    Validate predicted gates against panel markers.

    Returns:
        Tuple of (valid_count, invalid_count, marker_coverage)
    """
    valid = 0
    invalid = 0
    used_markers = set()

    # Add common non-marker dimensions
    extended_markers = panel_markers | {'fsc', 'ssc', 'time', 'fsc-a', 'fsc-h', 'ssc-a', 'ssc-h'}

    for gate in pred_gates:
        gate_lower = gate.lower()

        # Skip generic gates
        if any(skip in gate_lower for skip in ['all events', 'singlets', 'time', 'root']):
            valid += 1
            continue

        # Check if gate references panel markers
        gate_markers = set()
        for marker in extended_markers:
            if re.search(r'\b' + re.escape(marker) + r'\b', gate_lower):
                gate_markers.add(marker)

        if gate_markers:
            valid += 1
            used_markers.update(gate_markers & panel_markers)
        else:
            # Check if it's a known population name
            known = ['monocytes', 'mono', 'classical', 'intermediate',
                    'nonclassical', 'leukocytes', 'lymphocytes', 'granulocytes',
                    't cells', 'b cells', 'nk cells', 'dendritic']
            if any(pop in gate_lower for pop in known):
                valid += 1
            else:
                invalid += 1

    coverage = len(used_markers) / len(panel_markers) if panel_markers else 0
    return valid, invalid, coverage


def rescore_single(
    original_result: dict,
    ground_truth: dict,
) -> EnhancedEvaluationResult:
    """Re-score a single result with enhanced matching."""

    eval_data = original_result['evaluation']
    pred_gates = set(eval_data['predicted_gates'])
    gt_gates = set(eval_data['ground_truth_gates'])

    # Get panel markers
    panel_markers = {e['marker'].lower() for e in ground_truth['panel']['entries']}

    # Compute enhanced matches
    matches, missing, extra = compute_enhanced_matches(pred_gates, gt_gates)

    # Compute metrics
    n_pred = len(pred_gates)
    n_gt = len(gt_gates)
    n_matched = len(matches)

    precision = n_matched / n_pred if n_pred > 0 else 0
    recall = n_matched / n_gt if n_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Count match types
    match_counts = {}
    for m in matches:
        match_counts[m.match_type] = match_counts.get(m.match_type, 0) + 1

    # Marker validation
    valid, invalid, coverage = compute_marker_validation(pred_gates, panel_markers)

    return EnhancedEvaluationResult(
        test_case_id=original_result['test_case_id'],
        condition=original_result['condition'],
        original_f1=eval_data['hierarchy_f1'],
        original_precision=eval_data['hierarchy_precision'],
        original_recall=eval_data['hierarchy_recall'],
        enhanced_f1=f1,
        enhanced_precision=precision,
        enhanced_recall=recall,
        matches=matches,
        match_counts=match_counts,
        missing_gates=list(missing),
        extra_gates=list(extra),
        marker_valid_gates=valid,
        marker_invalid_gates=invalid,
        marker_coverage=coverage,
    )


def rescore_results(
    results_path: Path | str,
    ground_truth_dir: Path | str,
) -> list[EnhancedEvaluationResult]:
    """
    Re-score all results with enhanced matching.

    Args:
        results_path: Path to experiment results JSON
        ground_truth_dir: Directory containing ground truth files

    Returns:
        List of EnhancedEvaluationResult
    """
    results_path = Path(results_path)
    ground_truth_dir = Path(ground_truth_dir)

    with open(results_path) as f:
        data = json.load(f)

    # Load ground truth files
    ground_truths = {}
    for gt_file in ground_truth_dir.glob('omip_*.json'):
        with open(gt_file) as f:
            gt = json.load(f)
            # Normalize ID format
            omip_id = gt.get('omip_id', '').upper().replace('_', '-')
            ground_truths[omip_id] = gt

    enhanced_results = []
    for result in data['results']:
        tc_id = result['test_case_id']
        if tc_id in ground_truths:
            enhanced = rescore_single(result, ground_truths[tc_id])
            enhanced_results.append(enhanced)

    return enhanced_results


def print_comparison_report(enhanced_results: list[EnhancedEvaluationResult]) -> None:
    """Print a comparison report of original vs enhanced scoring."""

    print("=" * 70)
    print("ENHANCED SCORING COMPARISON REPORT")
    print("=" * 70)

    # Group by test case
    by_test_case: dict[str, list[EnhancedEvaluationResult]] = {}
    for r in enhanced_results:
        if r.test_case_id not in by_test_case:
            by_test_case[r.test_case_id] = []
        by_test_case[r.test_case_id].append(r)

    total_orig_f1 = 0
    total_enhanced_f1 = 0
    n = 0

    for tc_id in sorted(by_test_case.keys()):
        results = by_test_case[tc_id]

        avg_orig = sum(r.original_f1 for r in results) / len(results)
        avg_enhanced = sum(r.enhanced_f1 for r in results) / len(results)
        improvement = avg_enhanced - avg_orig

        print(f"\n{tc_id}: Original F1={avg_orig:.3f} -> Enhanced F1={avg_enhanced:.3f} ({improvement:+.3f})")

        # Show match type breakdown for best condition
        best = max(results, key=lambda r: r.enhanced_f1)
        if best.match_counts:
            counts = ", ".join(f"{k}:{v}" for k, v in sorted(best.match_counts.items()))
            print(f"  Best ({best.condition.split('_', 1)[1]}): {counts}")
        if best.missing_gates:
            print(f"  Still missing: {best.missing_gates[:3]}...")

        total_orig_f1 += avg_orig
        total_enhanced_f1 += avg_enhanced
        n += 1

    print("\n" + "=" * 70)
    print(f"OVERALL: Original={total_orig_f1/n:.3f} -> Enhanced={total_enhanced_f1/n:.3f} ({(total_enhanced_f1-total_orig_f1)/n:+.3f})")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python enhanced_metrics.py <results.json> <ground_truth_dir>")
        sys.exit(1)

    results = rescore_results(sys.argv[1], sys.argv[2])
    print_comparison_report(results)
