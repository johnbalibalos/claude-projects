#!/usr/bin/env python3
"""
Improved structure error analysis with biological context awareness.

Key improvements over analyze_structure_errors.py:
1. Marker alias integration (CCR7 = CD197, etc.)
2. Sample type context (CD45 optional for PBMCs)
3. Hard vs soft constraint separation
4. Semantic population matching
5. Valid gating order alternatives
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =============================================================================
# MARKER ALIASES (from marker_aliases.py)
# =============================================================================

MARKER_ALIAS_GROUPS = [
    # B cell markers
    ["B220", "CD45R"],
    # T cell activation/checkpoint
    ["ICOS", "CD278"],
    ["PD-1", "PD1", "CD279", "PDCD1"],
    ["PD-L1", "PDL1", "CD274"],
    ["PD-L2", "PDL2", "CD273"],
    ["CTLA-4", "CTLA4", "CD152"],
    ["LAG-3", "LAG3", "CD223"],
    ["TIM-3", "TIM3", "CD366", "HAVCR2"],
    # Chemokine receptors
    ["CXCR5", "CD185"],
    ["CCR7", "CD197"],
    ["CXCR3", "CD183"],
    ["CCR5", "CD195"],
    ["CXCR4", "CD184"],
    ["CCR4", "CD194"],
    ["CCR6", "CD196"],
    ["CX3CR1", "CD369"],
    # Integrins/adhesion
    ["CD103", "ITGAE", "INTEGRIN AE"],
    ["CD49D", "ITGA4", "VLA-4"],
    ["CD29", "ITGB1"],
    ["L-SELECTIN", "CD62L"],
    # Plasma cell markers
    ["CD138", "SYND-1", "SYNDECAN-1", "SDC1"],
    # MHC markers
    ["MHC CLASS II", "HLA-DR", "I-A/I-E", "MHC-II", "IA-IE", "MHCII"],
    # IL receptors
    ["IL-7R", "IL7R", "CD127"],
    ["IL-2R", "IL2R", "CD25"],
    ["IL-15R", "IL15R", "CD122"],
    # Other
    ["KLRG1", "KLRG-1"],
    ["NKG2A", "CD159A"],
    ["NKG2D", "CD314"],
    ["NKP46", "NCR1", "CD335"],
    ["NKP44", "NCR2", "CD336"],
    ["NKP30", "NCR3", "CD337"],
]

# Build bidirectional alias lookup
MARKER_TO_CANONICAL: dict[str, str] = {}
for group in MARKER_ALIAS_GROUPS:
    canonical = group[0].upper()
    for alias in group:
        MARKER_TO_CANONICAL[alias.upper()] = canonical
        # Also handle with/without hyphens
        MARKER_TO_CANONICAL[alias.upper().replace("-", "")] = canonical
        MARKER_TO_CANONICAL[alias.upper().replace("-", " ")] = canonical


# =============================================================================
# CELL TYPE SYNONYMS
# =============================================================================

CELL_TYPE_SYNONYMS: dict[str, str] = {
    # T cells
    "t cells": "t_cells",
    "t-cells": "t_cells",
    "t lymphocytes": "t_cells",
    "cd3+ t cells": "t_cells",
    "cd3+": "cd3_t_cells",
    # CD4 T cells
    "cd4+ t cells": "cd4_t_cells",
    "cd4 t cells": "cd4_t_cells",
    "helper t cells": "cd4_t_cells",
    "th cells": "cd4_t_cells",
    # CD8 T cells
    "cd8+ t cells": "cd8_t_cells",
    "cd8 t cells": "cd8_t_cells",
    "cytotoxic t cells": "cd8_t_cells",
    "ctl": "cd8_t_cells",
    # Memory subsets
    "naive": "naive",
    "central memory": "cm",
    "cm": "cm",
    "tcm": "cm",
    "effector memory": "em",
    "em": "em",
    "tem": "em",
    "temra": "temra",
    "emra": "temra",
    # B cells
    "b cells": "b_cells",
    "b-cells": "b_cells",
    "b lymphocytes": "b_cells",
    "cd19+ b cells": "b_cells",
    "cd19+": "cd19_b_cells",
    "cd20+": "cd20_b_cells",
    # NK cells
    "nk cells": "nk_cells",
    "nk": "nk_cells",
    "natural killer cells": "nk_cells",
    "cd56+ nk cells": "nk_cells",
    "cd56+cd3-": "nk_cells",
    "cd3-cd56+": "nk_cells",
    # Monocytes
    "monocytes": "monocytes",
    "monos": "monocytes",
    "cd14+ monocytes": "monocytes",
    "cd14+": "cd14_monocytes",
    "classical monocytes": "classical_monocytes",
    "non-classical monocytes": "nonclassical_monocytes",
    "nonclassical monocytes": "nonclassical_monocytes",
    "intermediate monocytes": "intermediate_monocytes",
    # QC gates
    "singlets": "singlets",
    "single cells": "singlets",
    "live cells": "live",
    "live": "live",
    "viable cells": "live",
    "viable": "live",
    "lymphocytes": "lymphocytes",
    "lymphs": "lymphocytes",
    "leukocytes": "leukocytes",
    "cd45+ leukocytes": "leukocytes",
    "cd45+": "leukocytes",
    "all events": "all_events",
    "root": "all_events",
    "ungated": "all_events",
    # Tregs
    "tregs": "tregs",
    "treg": "tregs",
    "regulatory t cells": "tregs",
    # DCs
    "dendritic cells": "dcs",
    "dcs": "dcs",
    "myeloid dcs": "mdcs",
    "mdcs": "mdcs",
    "plasmacytoid dcs": "pdcs",
    "pdcs": "pdcs",
}


# =============================================================================
# HARD BIOLOGICAL CONSTRAINTS
# =============================================================================

# Mutually exclusive lineage markers (should NEVER co-occur in single cell)
LINEAGE_EXCLUSIVITY_RULES = [
    ({"cd3", "cd3+"}, {"cd19", "cd19+", "cd20", "cd20+"}),  # T vs B
    ({"cd3", "cd3+"}, {"cd14", "cd14+"}),  # T vs Monocyte
    ({"cd19", "cd19+", "cd20", "cd20+"}, {"cd14", "cd14+"}),  # B vs Monocyte
]


# =============================================================================
# SOFT CONSTRAINTS (context-dependent)
# =============================================================================

# Sample types where CD45 gating is optional
CD45_OPTIONAL_SAMPLES = [
    "pbmc", "pbmcs", "human pbmc",
    "peripheral blood mononuclear",
    "whole blood", "blood",
    "cryopreserved pbmc",
]

# Valid gating order alternatives (these are NOT errors)
VALID_GATING_ORDERS = [
    # Both singlets→live and live→singlets are acceptable
    ("singlets", "live"),
    ("live", "singlets"),
    # Both orders for lymphocyte gating
    ("live", "lymphocytes"),
    ("lymphocytes", "live"),  # Less common but valid
]


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_marker_in_name(name: str) -> str:
    """Replace marker aliases with canonical forms in a gate name."""
    result = name.upper()

    # Sort by length descending to match longer patterns first
    for alias, canonical in sorted(MARKER_TO_CANONICAL.items(), key=lambda x: -len(x[0])):
        # Use word boundary matching
        pattern = r'\b' + re.escape(alias) + r'\b'
        result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)

    return result.lower()


def normalize_gate_name(name: str) -> str:
    """Comprehensive gate name normalization."""
    if not name:
        return ""

    n = name.lower().strip()

    # Normalize markers first
    n = normalize_marker_in_name(n)

    # Normalize +/- notation
    n = re.sub(r'\s*\+\s*', '+', n)
    n = re.sub(r'\s*-\s*', '-', n)
    n = re.sub(r'\s+positive\b', '+', n)
    n = re.sub(r'\s+negative\b', '-', n)
    n = re.sub(r'\bpositive\b', '+', n)
    n = re.sub(r'\bnegative\b', '-', n)

    # Remove parenthetical qualifiers like (FSC), (SSC-A vs SSC-H)
    n = re.sub(r'\s*\([^)]*\)\s*', ' ', n)

    # Normalize whitespace
    n = ' '.join(n.split())

    # Check cell type synonyms
    if n in CELL_TYPE_SYNONYMS:
        return CELL_TYPE_SYNONYMS[n]

    # Check partial matches
    for synonym, canonical in CELL_TYPE_SYNONYMS.items():
        if synonym in n:
            return canonical

    return n


def extract_markers_from_gate(gate_name: str) -> dict[str, str]:
    """
    Extract marker names from a gate name with their polarity.

    Returns dict mapping marker -> polarity ('+', '-', or '?')
    """
    markers = {}
    name = gate_name.upper()

    # CD markers: CD followed by digits, optionally with letter suffix, then +/-
    cd_pattern = r'\b(CD\d+[A-Z]?)([+-])?\b'
    for match in re.finditer(cd_pattern, name):
        marker = match.group(1).lower()
        polarity = match.group(2) or "?"
        markers[marker] = polarity

    # Chemokine receptors
    chemokine_pattern = r'\b(CCR\d+|CXCR\d+|CX3CR\d+)([+-])?\b'
    for match in re.finditer(chemokine_pattern, name, re.IGNORECASE):
        marker = match.group(1).lower()
        polarity = match.group(2) or "?"
        markers[marker] = polarity

    return markers


def get_positive_markers(gate: dict) -> set[str]:
    """Get markers that are explicitly positive in a gate."""
    markers = gate.get("markers", {})
    if isinstance(markers, set):
        return markers  # Old format compatibility
    return {m for m, p in markers.items() if p == "+"}


# =============================================================================
# HIERARCHY UTILITIES
# =============================================================================

def extract_all_gates(hierarchy: dict) -> list[dict]:
    """Extract all gates with their context."""
    gates = []

    def traverse(node: dict, parent: str | None = None, depth: int = 0, path: list[str] = None):
        if path is None:
            path = []

        if "name" in node:
            current_path = path + [node["name"]]
            gates.append({
                "name": node["name"],
                "normalized": normalize_gate_name(node["name"]),
                "parent": parent,
                "depth": depth,
                "path": current_path,
                "markers": extract_markers_from_gate(node["name"]),
            })
            for child in node.get("children", []):
                traverse(child, node["name"], depth + 1, current_path)

    if "root" in hierarchy:
        traverse(hierarchy["root"])
    elif "name" in hierarchy:
        traverse(hierarchy)

    return gates


def get_terminal_populations(hierarchy: dict) -> list[dict]:
    """Get leaf nodes (populations with no children)."""
    all_gates = extract_all_gates(hierarchy)

    # Build parent set
    parents = set()
    for gate in all_gates:
        if gate["parent"]:
            parents.add(normalize_gate_name(gate["parent"]))

    # Return gates that are not parents
    return [g for g in all_gates if g["normalized"] not in parents]


# =============================================================================
# BIOLOGICAL VALIDATION
# =============================================================================

def check_lineage_exclusivity(gate: dict) -> list[str]:
    """Check if a gate violates lineage exclusivity rules (positive markers only)."""
    violations = []
    positive_markers = get_positive_markers(gate)
    name = gate["name"]

    for group_a, group_b in LINEAGE_EXCLUSIVITY_RULES:
        # Only check POSITIVE markers - CD3- CD19+ is fine
        has_a = bool(positive_markers & group_a)
        has_b = bool(positive_markers & group_b)

        if has_a and has_b:
            violations.append(
                f"HARD: '{name}' has mutually exclusive POSITIVE markers "
                f"({positive_markers & group_a} + {positive_markers & group_b})"
            )

    return violations


def check_cd4_cd8_double_positive(gate: dict, context: dict) -> list[str]:
    """Check for CD4+CD8+ which is rare in periphery."""
    warnings = []
    markers = gate.get("markers", {})

    # Check if BOTH are explicitly positive
    has_cd4_positive = markers.get("cd4") == "+"
    has_cd8_positive = markers.get("cd8") == "+"

    if has_cd4_positive and has_cd8_positive:
        sample = context.get("sample_type", "").lower()
        if "thymus" not in sample and "thymocyte" not in sample:
            warnings.append(
                f"WARNING: '{gate['name']}' is CD4+CD8+ (rare in periphery, "
                f"<3% normal, check if intentional)"
            )

    return warnings


# =============================================================================
# IMPROVED MATCHING
# =============================================================================

@dataclass
class MatchResult:
    """Result of comparing predicted vs ground truth."""
    # Exact/semantic matches
    matched_gates: list[tuple[str, str]] = field(default_factory=list)  # (pred, gt)

    # True missing (not in prediction, no equivalent found)
    truly_missing: list[str] = field(default_factory=list)

    # Missing but acceptable (context-dependent)
    acceptable_missing: list[tuple[str, str]] = field(default_factory=list)  # (gate, reason)

    # Extra predictions (not necessarily errors)
    extra_predictions: list[str] = field(default_factory=list)

    # Hard biological violations
    hard_violations: list[str] = field(default_factory=list)

    # Soft structural deviations
    soft_deviations: list[str] = field(default_factory=list)

    # Statistics
    total_gt_gates: int = 0
    total_pred_gates: int = 0


def compare_hierarchies(
    predicted: dict,
    ground_truth: dict,
    context: dict,
) -> MatchResult:
    """
    Compare predicted hierarchy against ground truth with biological awareness.
    """
    result = MatchResult()

    pred_gates = extract_all_gates(predicted)
    gt_gates = extract_all_gates(ground_truth)

    result.total_pred_gates = len(pred_gates)
    result.total_gt_gates = len(gt_gates)

    # Build normalized lookup
    pred_normalized = {g["normalized"]: g for g in pred_gates}
    pred_names_lower = {g["name"].lower(): g for g in pred_gates}

    gt_normalized = {g["normalized"]: g for g in gt_gates}

    sample_type = context.get("sample_type", "").lower()
    is_pbmc = any(s in sample_type for s in CD45_OPTIONAL_SAMPLES)

    matched_gt = set()
    matched_pred = set()

    # Pass 1: Exact normalized matches
    for gt_norm, gt_gate in gt_normalized.items():
        if gt_norm in pred_normalized:
            result.matched_gates.append((pred_normalized[gt_norm]["name"], gt_gate["name"]))
            matched_gt.add(gt_gate["name"])
            matched_pred.add(pred_normalized[gt_norm]["name"])

    # Pass 2: Fuzzy matching for unmatched
    for gt_gate in gt_gates:
        if gt_gate["name"] in matched_gt:
            continue

        gt_norm = gt_gate["normalized"]
        gt_name_lower = gt_gate["name"].lower()

        # Try direct lowercase match
        if gt_name_lower in pred_names_lower:
            pred_gate = pred_names_lower[gt_name_lower]
            if pred_gate["name"] not in matched_pred:
                result.matched_gates.append((pred_gate["name"], gt_gate["name"]))
                matched_gt.add(gt_gate["name"])
                matched_pred.add(pred_gate["name"])
                continue

        # Try partial match (for gates like "CD4+ T cells" vs "CD4 T cells")
        for pred_gate in pred_gates:
            if pred_gate["name"] in matched_pred:
                continue

            pred_norm = pred_gate["normalized"]

            # Check if one contains the other
            if gt_norm in pred_norm or pred_norm in gt_norm:
                result.matched_gates.append((pred_gate["name"], gt_gate["name"]))
                matched_gt.add(gt_gate["name"])
                matched_pred.add(pred_gate["name"])
                break

    # Categorize unmatched GT gates
    for gt_gate in gt_gates:
        if gt_gate["name"] in matched_gt:
            continue

        gt_name = gt_gate["name"]
        gt_norm = gt_gate["normalized"]

        # Check if this is an acceptable missing gate

        # CD45 gates in PBMC samples
        if is_pbmc and ("cd45" in gt_norm or "leukocyte" in gt_norm):
            result.acceptable_missing.append((gt_name, "CD45 optional for PBMC samples"))
            continue

        # Time gate (optional)
        if gt_norm in ["time", "time gate"]:
            result.acceptable_missing.append((gt_name, "Time gate optional if acquisition stable"))
            continue

        # Otherwise truly missing
        result.truly_missing.append(gt_name)

    # Extra predictions
    for pred_gate in pred_gates:
        if pred_gate["name"] not in matched_pred:
            result.extra_predictions.append(pred_gate["name"])

    # Check for hard biological violations in predictions
    for pred_gate in pred_gates:
        violations = check_lineage_exclusivity(pred_gate)
        result.hard_violations.extend(violations)

        warnings = check_cd4_cd8_double_positive(pred_gate, context)
        result.soft_deviations.extend(warnings)

    return result


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def parse_hierarchy_from_response(raw_response: str) -> dict | None:
    """Extract JSON hierarchy from LLM response."""
    if not raw_response:
        return None

    # Try to find JSON in the response
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    first_brace = raw_response.find('{')
    if first_brace != -1:
        depth = 0
        for i, char in enumerate(raw_response[first_brace:], first_brace):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw_response[first_brace:i+1])
                    except json.JSONDecodeError:
                        pass
                    break

    return None


def load_test_cases(data_dir: Path) -> dict[str, dict]:
    """Load ground truth test cases."""
    test_cases = {}
    for json_file in data_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            test_case_id = data.get("test_case_id", json_file.stem)
            test_cases[test_case_id] = data
    return test_cases


def main():
    """Run improved analysis."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "full_benchmark_20260114"
    data_dir = project_root / "data" / "verified"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    predictions_file = results_dir / "predictions.json"
    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        return

    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file) as f:
        predictions = json.load(f)

    print(f"Loading test cases from {data_dir}...")
    test_cases = load_test_cases(data_dir)
    print(f"Loaded {len(test_cases)} test cases")

    # Aggregate results
    all_results: list[MatchResult] = []
    results_by_model: dict[str, list[MatchResult]] = defaultdict(list)
    results_by_test_case: dict[str, list[MatchResult]] = defaultdict(list)

    hard_violations_all: list[str] = []
    soft_deviations_all: list[str] = []

    print(f"\nAnalyzing {len(predictions)} predictions with biological context...")

    parsed_count = 0
    for i, pred in enumerate(predictions):
        test_case_id = pred.get("test_case_id")
        model = pred.get("model", "unknown")

        if test_case_id not in test_cases:
            continue

        tc = test_cases[test_case_id]
        gt_hierarchy = tc.get("gating_hierarchy", tc.get("hierarchy", {}))

        # Build context
        context = tc.get("context", {})
        if "sample_type" not in context:
            context["sample_type"] = context.get("sample", "unknown")

        # Parse predicted hierarchy
        raw_response = pred.get("raw_response", "")
        parsed = parse_hierarchy_from_response(raw_response)
        if not parsed:
            continue

        parsed_count += 1

        # Compare
        match_result = compare_hierarchies(parsed, gt_hierarchy, context)
        all_results.append(match_result)
        results_by_model[model].append(match_result)
        results_by_test_case[test_case_id].append(match_result)

        hard_violations_all.extend(match_result.hard_violations)
        soft_deviations_all.extend(match_result.soft_deviations)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(predictions)}")

    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================

    print("\n" + "=" * 80)
    print("IMPROVED ANALYSIS RESULTS (with biological context)")
    print("=" * 80)

    print(f"\nPredictions analyzed: {parsed_count}")

    # Aggregate statistics
    total_matched = sum(len(r.matched_gates) for r in all_results)
    total_truly_missing = sum(len(r.truly_missing) for r in all_results)
    total_acceptable_missing = sum(len(r.acceptable_missing) for r in all_results)
    total_extra = sum(len(r.extra_predictions) for r in all_results)
    total_gt = sum(r.total_gt_gates for r in all_results)
    total_pred = sum(r.total_pred_gates for r in all_results)

    print(f"\n--- Gate Matching Summary ---")
    print(f"Total GT gates across all predictions: {total_gt}")
    print(f"Total predicted gates: {total_pred}")
    print(f"Matched gates: {total_matched} ({100*total_matched/total_gt:.1f}% of GT)")
    print(f"Truly missing gates: {total_truly_missing} ({100*total_truly_missing/total_gt:.1f}% of GT)")
    print(f"Acceptable missing (context-dependent): {total_acceptable_missing}")
    print(f"Extra predictions: {total_extra}")

    # Compare to old analysis
    old_missing = 47928  # From previous analysis
    improvement = old_missing - total_truly_missing
    print(f"\n--- Comparison to Previous Analysis ---")
    print(f"Previous 'MISSING_GATE' count: {old_missing}")
    print(f"New 'truly missing' count: {total_truly_missing}")
    print(f"Reduction: {improvement} ({100*improvement/old_missing:.1f}% were false positives)")

    # Hard violations
    print(f"\n--- Hard Biological Violations ---")
    print(f"Total: {len(hard_violations_all)}")
    if hard_violations_all:
        violation_counts = Counter(hard_violations_all)
        print("Most common:")
        for v, count in violation_counts.most_common(10):
            print(f"  {count:4d}x {v}")
    else:
        print("  None found! Models respect lineage exclusivity rules.")

    # Soft deviations
    print(f"\n--- Soft Deviations (warnings, not errors) ---")
    print(f"Total: {len(soft_deviations_all)}")
    if soft_deviations_all:
        for d in soft_deviations_all[:5]:
            print(f"  • {d}")

    # By model
    print(f"\n--- Results by Model ---")
    print(f"{'Model':<20} {'Matched':>10} {'Truly Missing':>15} {'Acceptable Missing':>20} {'Match Rate':>12}")
    print("-" * 80)
    for model in sorted(results_by_model.keys()):
        results = results_by_model[model]
        matched = sum(len(r.matched_gates) for r in results)
        missing = sum(len(r.truly_missing) for r in results)
        acceptable = sum(len(r.acceptable_missing) for r in results)
        gt_total = sum(r.total_gt_gates for r in results)
        rate = 100 * matched / gt_total if gt_total > 0 else 0
        print(f"{model:<20} {matched:>10} {missing:>15} {acceptable:>20} {rate:>11.1f}%")

    # By test case
    print(f"\n--- Results by Test Case ---")
    print(f"{'Test Case':<20} {'Sample Type':<30} {'Matched':>10} {'Truly Missing':>15}")
    print("-" * 80)
    for tc_id in sorted(results_by_test_case.keys()):
        results = results_by_test_case[tc_id]
        tc = test_cases.get(tc_id, {})
        sample = tc.get("context", {}).get("sample_type", tc.get("context", {}).get("sample", "unknown"))[:28]
        matched = sum(len(r.matched_gates) for r in results)
        missing = sum(len(r.truly_missing) for r in results)
        print(f"{tc_id:<20} {sample:<30} {matched:>10} {missing:>15}")

    # Sample truly missing gates
    print(f"\n--- Sample of Truly Missing Gates ---")
    truly_missing_sample = []
    for r in all_results[:100]:
        truly_missing_sample.extend(r.truly_missing[:3])

    missing_counts = Counter(truly_missing_sample)
    print("Most commonly missing (sample):")
    for gate, count in missing_counts.most_common(15):
        print(f"  {count:3d}x {gate}")

    # Sample acceptable missing
    print(f"\n--- Sample of Acceptable Missing (not errors) ---")
    acceptable_sample = []
    for r in all_results[:100]:
        acceptable_sample.extend(r.acceptable_missing[:3])

    for gate, reason in acceptable_sample[:10]:
        print(f"  • {gate}: {reason}")

    # Save results
    output_file = results_dir / "improved_analysis_results.json"
    output_data = {
        "summary": {
            "predictions_analyzed": parsed_count,
            "total_gt_gates": total_gt,
            "total_pred_gates": total_pred,
            "matched_gates": total_matched,
            "truly_missing": total_truly_missing,
            "acceptable_missing": total_acceptable_missing,
            "extra_predictions": total_extra,
            "hard_violations": len(hard_violations_all),
            "soft_deviations": len(soft_deviations_all),
            "match_rate": total_matched / total_gt if total_gt > 0 else 0,
        },
        "comparison_to_previous": {
            "old_missing_count": old_missing,
            "new_truly_missing": total_truly_missing,
            "false_positive_rate": improvement / old_missing if old_missing > 0 else 0,
        },
        "hard_violations": list(set(hard_violations_all)),
        "by_model": {
            model: {
                "matched": sum(len(r.matched_gates) for r in results),
                "truly_missing": sum(len(r.truly_missing) for r in results),
                "acceptable_missing": sum(len(r.acceptable_missing) for r in results),
            }
            for model, results in results_by_model.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
