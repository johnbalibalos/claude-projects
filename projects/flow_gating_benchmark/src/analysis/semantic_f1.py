"""
Semantic F1 Metric for Flow Cytometry Gate Matching

Addresses limitations of string-based F1:
- Case sensitivity ("B Cells" vs "B cells")
- Abbreviations ("Tregs" vs "Regulatory T cells")
- Minor word variations ("CD56bright NK" vs "CD56bright NK cells")

Uses tiered approach:
1. Normalization (free)
2. Fuzzy matching with Jaccard (free)
3. Optional LLM verification for edge cases (not implemented here)
"""

import re
from typing import Set, Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# Tier 1: Normalization
# ============================================================================

ABBREVIATIONS = {
    # T cell subsets
    'tregs': 'regulatory t cells',
    'treg': 'regulatory t cells',
    'tfh': 't follicular helper cells',
    'tfr': 't follicular regulatory cells',
    'tcm': 'central memory t cells',
    'tem': 'effector memory t cells',
    'temra': 'terminally differentiated effector memory t cells',
    'th1': 't helper 1 cells',
    'th2': 't helper 2 cells',
    'th17': 't helper 17 cells',

    # Other abbreviations
    'nk': 'natural killer',
    'dc': 'dendritic cells',
    'dcs': 'dendritic cells',
    'mdc': 'myeloid dendritic cells',
    'mdcs': 'myeloid dendritic cells',
    'pdc': 'plasmacytoid dendritic cells',
    'pdcs': 'plasmacytoid dendritic cells',
    'asc': 'antibody secreting cells',
    'ascs': 'antibody secreting cells',
    'pbmc': 'peripheral blood mononuclear cells',
}

# Marker patterns to strip (these cause false mismatches)
MARKER_PATTERN = re.compile(r'\([^)]*[+\-][^)]*\)')  # (CD25+CD127-)
# Match marker prefixes like "CD4+CD44-CD62L+" or "CD3-CD56+"
MARKER_PREFIX_PATTERN = re.compile(
    r'^([A-Za-z]+\d*[+\-]+\s*)+',  # More general: letters+optional digits+plus/minus
    re.IGNORECASE
)


def normalize_gate_name(name: str) -> str:
    """
    Normalize a gate name for semantic comparison.

    Transformations:
    - Lowercase
    - Expand abbreviations
    - Remove marker annotations in parentheses
    - Standardize cell suffix
    - Normalize whitespace
    """
    if not name:
        return ""

    # Lowercase
    normalized = name.lower().strip()

    # Remove marker annotations like (CD25+CD127-)
    normalized = MARKER_PATTERN.sub('', normalized)

    # Remove marker prefixes like "CD4+CD44-CD62L+"
    normalized = MARKER_PREFIX_PATTERN.sub('', normalized)

    # Expand abbreviations (whole word matching)
    words = normalized.split()
    expanded_words = []
    for word in words:
        # Strip punctuation for matching
        clean_word = word.strip('(),')
        if clean_word in ABBREVIATIONS:
            expanded_words.append(ABBREVIATIONS[clean_word])
        else:
            expanded_words.append(word)
    normalized = ' '.join(expanded_words)

    # Standardize "cells" suffix
    # "B cell" -> "B cells", "NK" -> "NK cells" (if it's a cell type)
    cell_types = ['t cells', 'b cells', 'nk cells', 'monocytes', 'dendritic cells']
    if any(ct in normalized for ct in ['t cell', 'b cell', 'nk cell']):
        normalized = re.sub(r'\bcell\b', 'cells', normalized)

    # Normalize whitespace
    normalized = ' '.join(normalized.split())

    return normalized


def normalize_gate_set(gates: List[str]) -> Set[str]:
    """Normalize a list of gate names to a set."""
    return {normalize_gate_name(g) for g in gates if g}


# ============================================================================
# Tier 2: Fuzzy Matching
# ============================================================================

def tokenize(name: str) -> Set[str]:
    """Convert gate name to word tokens for Jaccard comparison."""
    # Split on whitespace and common delimiters
    tokens = re.split(r'[\s+\-/()]+', name.lower())
    # Remove empty tokens and very short ones
    return {t for t in tokens if len(t) > 1}


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two gate names."""
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union if union > 0 else 0.0


def find_best_match(pred: str, ground_truth_set: Set[str],
                    threshold: float = 0.5) -> Tuple[str, float]:
    """
    Find the best matching ground truth gate for a prediction.

    Returns (best_match, similarity_score).
    Returns (None, 0) if no match above threshold.
    """
    best_match = None
    best_score = 0.0

    pred_normalized = normalize_gate_name(pred)

    for gt in ground_truth_set:
        gt_normalized = normalize_gate_name(gt)

        # Exact match after normalization
        if pred_normalized == gt_normalized:
            return (gt, 1.0)

        # Fuzzy match
        score = jaccard_similarity(pred_normalized, gt_normalized)
        if score > best_score:
            best_score = score
            best_match = gt

    if best_score >= threshold:
        return (best_match, best_score)

    return (None, 0.0)


# ============================================================================
# Semantic F1 Calculation
# ============================================================================

@dataclass
class SemanticF1Result:
    """Result of semantic F1 calculation."""
    precision: float
    recall: float
    f1: float

    # Breakdown
    exact_matches: int
    fuzzy_matches: int
    unmatched_pred: int
    unmatched_gt: int

    # For debugging
    match_details: List[Dict]


def compute_semantic_f1(
    predicted_gates: List[str],
    ground_truth_gates: List[str],
    fuzzy_threshold: float = 0.5,
    fuzzy_weight: float = 0.8  # Fuzzy matches count as 0.8 of exact match
) -> SemanticF1Result:
    """
    Compute semantic F1 score with normalization and fuzzy matching.

    Args:
        predicted_gates: List of predicted gate names
        ground_truth_gates: List of ground truth gate names
        fuzzy_threshold: Minimum Jaccard similarity for fuzzy match
        fuzzy_weight: Weight for fuzzy matches (1.0 = same as exact)

    Returns:
        SemanticF1Result with scores and breakdown
    """
    pred_set = set(predicted_gates)
    gt_set = set(ground_truth_gates)

    if not pred_set or not gt_set:
        return SemanticF1Result(
            precision=0.0, recall=0.0, f1=0.0,
            exact_matches=0, fuzzy_matches=0,
            unmatched_pred=len(pred_set), unmatched_gt=len(gt_set),
            match_details=[]
        )

    # Track matches
    matched_gt = set()
    match_details = []
    exact_matches = 0
    fuzzy_matches = 0
    weighted_matches = 0.0

    for pred in pred_set:
        pred_norm = normalize_gate_name(pred)

        # Try exact match first (after normalization)
        exact_found = False
        for gt in gt_set:
            if gt in matched_gt:
                continue
            gt_norm = normalize_gate_name(gt)
            if pred_norm == gt_norm:
                matched_gt.add(gt)
                exact_matches += 1
                weighted_matches += 1.0
                match_details.append({
                    'pred': pred,
                    'gt': gt,
                    'type': 'exact',
                    'score': 1.0
                })
                exact_found = True
                break

        if exact_found:
            continue

        # Try fuzzy match
        best_match, best_score = find_best_match(
            pred,
            gt_set - matched_gt,
            threshold=fuzzy_threshold
        )

        if best_match:
            matched_gt.add(best_match)
            fuzzy_matches += 1
            weighted_matches += fuzzy_weight
            match_details.append({
                'pred': pred,
                'gt': best_match,
                'type': 'fuzzy',
                'score': best_score
            })
        else:
            match_details.append({
                'pred': pred,
                'gt': None,
                'type': 'unmatched',
                'score': 0.0
            })

    # Calculate precision and recall
    precision = weighted_matches / len(pred_set) if pred_set else 0
    recall = weighted_matches / len(gt_set) if gt_set else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return SemanticF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_matches=exact_matches,
        fuzzy_matches=fuzzy_matches,
        unmatched_pred=len(pred_set) - exact_matches - fuzzy_matches,
        unmatched_gt=len(gt_set) - len(matched_gt),
        match_details=match_details
    )


# ============================================================================
# Comparison with Original F1
# ============================================================================

def compute_original_f1(predicted_gates: List[str],
                        ground_truth_gates: List[str]) -> float:
    """Compute original string-matching F1 for comparison."""
    pred_set = set(predicted_gates)
    gt_set = set(ground_truth_gates)

    if not pred_set or not gt_set:
        return 0.0

    matches = len(pred_set & gt_set)
    precision = matches / len(pred_set)
    recall = matches / len(gt_set)

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test cases
    print("=== Normalization Tests ===\n")

    test_names = [
        "B Cells",
        "b cells",
        "CD4+CD44-CD62L+ Naive T Cells",
        "Naive T cells",
        "Tregs",
        "Regulatory T Cells (CD25+CD127-)",
        "CD56bright NK",
        "CD56bright NK cells",
    ]

    for name in test_names:
        print(f"'{name}' → '{normalize_gate_name(name)}'")

    print("\n=== Semantic F1 Test ===\n")

    pred = [
        "All Events",
        "Singlets",
        "Live Cells",
        "CD4+ T Cells",  # Case difference
        "Tregs",  # Abbreviation
        "CD56bright NK",  # Missing "cells"
        "Fake Population",  # Hallucination
    ]

    gt = [
        "All Events",
        "Singlets",
        "Live cells",
        "CD4+ T cells",
        "Regulatory T cells",
        "CD56bright NK cells",
        "B cells",  # Missing from pred
    ]

    result = compute_semantic_f1(pred, gt)
    original = compute_original_f1(pred, gt)

    print(f"Original F1: {original:.3f}")
    print(f"Semantic F1: {result.f1:.3f}")
    print(f"Improvement: {result.f1 - original:+.3f}")
    print(f"\nBreakdown:")
    print(f"  Exact matches: {result.exact_matches}")
    print(f"  Fuzzy matches: {result.fuzzy_matches}")
    print(f"  Unmatched pred: {result.unmatched_pred}")
    print(f"  Unmatched gt: {result.unmatched_gt}")

    print("\nMatch details:")
    for m in result.match_details:
        if m['type'] == 'exact':
            print(f"  ✓ '{m['pred']}' = '{m['gt']}'")
        elif m['type'] == 'fuzzy':
            print(f"  ~ '{m['pred']}' ≈ '{m['gt']}' ({m['score']:.2f})")
        else:
            print(f"  ✗ '{m['pred']}' (no match)")
