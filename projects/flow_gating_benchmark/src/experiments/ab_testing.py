#!/usr/bin/env python3
"""
A/B Testing Framework for Gating Strategy Evaluation.

Compares LLM predictions against two ground truth standards:
- HIPC (2016): Expert-validated standardized phenotype definitions
- OMIP: Paper-specific gating strategies from recent publications

This allows us to evaluate:
1. Which standard LLMs align with more naturally
2. Whether HIPC standardization helps or hurts prediction accuracy
3. Where LLM "hallucinations" actually match expert standards
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GatingStandard(str, Enum):
    """Ground truth standards for comparison."""

    HIPC_2016 = "hipc_2016"  # Expert-validated HIPC standardization
    OMIP_PAPER = "omip_paper"  # Paper-specific gating from OMIP publications


class CellPopulationDefinition(BaseModel):
    """Expert-validated cell population definition from HIPC."""

    name: str = Field(..., description="Population name (e.g., 'T cells')")
    canonical_names: list[str] = Field(
        default_factory=list,
        description="Acceptable alternative names",
    )
    positive_markers: list[str] = Field(
        default_factory=list,
        description="Required positive markers",
    )
    negative_markers: list[str] = Field(
        default_factory=list,
        description="Required negative markers",
    )
    optional_markers: list[str] = Field(
        default_factory=list,
        description="Optional markers (either acceptable)",
    )
    parent_population: str | None = Field(
        None,
        description="Parent population in hierarchy",
    )
    notes: str | None = Field(None, description="Additional notes")
    source: str = Field("HIPC 2016", description="Source reference")


# HIPC 2016 Expert-Validated Cell Population Definitions
# Reference: https://www.nature.com/articles/srep20686
HIPC_CELL_DEFINITIONS: dict[str, CellPopulationDefinition] = {
    # T cell lineage
    "t_cells": CellPopulationDefinition(
        name="T cells",
        canonical_names=["T lymphocytes", "CD3+ cells", "CD3+ T cells"],
        positive_markers=["CD3"],
        negative_markers=["CD19", "CD20"],  # Exclude B cells
        parent_population="Lymphocytes",
        notes="Primary T cell gate; CD19- recommended when B cell markers in panel",
        source="HIPC 2016 Table 1",
    ),
    "cd4_t_cells": CellPopulationDefinition(
        name="CD4+ T cells",
        canonical_names=["Helper T cells", "CD4 T cells", "Th cells"],
        positive_markers=["CD3", "CD4"],
        negative_markers=["CD8"],
        parent_population="T cells",
        notes="CD3+ CD4+ CD8- helper T cells",
        source="HIPC 2016 Table 2",
    ),
    "cd8_t_cells": CellPopulationDefinition(
        name="CD8+ T cells",
        canonical_names=["Cytotoxic T cells", "CD8 T cells", "CTL"],
        positive_markers=["CD3", "CD8"],
        negative_markers=["CD4"],
        parent_population="T cells",
        notes="CD3+ CD4- CD8+ cytotoxic T cells",
        source="HIPC 2016 Table 2",
    ),
    "cd4_naive": CellPopulationDefinition(
        name="CD4+ Naive",
        canonical_names=["Naive CD4", "CD4 Naive T cells"],
        positive_markers=["CD3", "CD4", "CD45RA", "CCR7"],
        negative_markers=["CD8"],
        parent_population="CD4+ T cells",
        notes="CD45RA+ CCR7+ naive helper T cells",
        source="HIPC 2016 Table 2",
    ),
    "cd4_cm": CellPopulationDefinition(
        name="CD4+ Central Memory",
        canonical_names=["CD4 CM", "CD4+ CM", "Central Memory CD4"],
        positive_markers=["CD3", "CD4", "CCR7"],
        negative_markers=["CD8", "CD45RA"],
        parent_population="CD4+ T cells",
        notes="CD45RA- CCR7+ central memory helper T cells",
        source="HIPC 2016 Table 2",
    ),
    "cd4_em": CellPopulationDefinition(
        name="CD4+ Effector Memory",
        canonical_names=["CD4 EM", "CD4+ EM", "Effector Memory CD4"],
        positive_markers=["CD3", "CD4"],
        negative_markers=["CD8", "CD45RA", "CCR7"],
        parent_population="CD4+ T cells",
        notes="CD45RA- CCR7- effector memory helper T cells",
        source="HIPC 2016 Table 2",
    ),
    "cd4_temra": CellPopulationDefinition(
        name="CD4+ TEMRA",
        canonical_names=["CD4 TEMRA", "CD4+ EMRA", "Terminally differentiated CD4"],
        positive_markers=["CD3", "CD4", "CD45RA"],
        negative_markers=["CD8", "CCR7"],
        parent_population="CD4+ T cells",
        notes="CD45RA+ CCR7- terminally differentiated effector memory",
        source="HIPC 2016 Table 2",
    ),
    # Similar definitions for CD8 subsets
    "cd8_naive": CellPopulationDefinition(
        name="CD8+ Naive",
        canonical_names=["Naive CD8", "CD8 Naive T cells"],
        positive_markers=["CD3", "CD8", "CD45RA", "CCR7"],
        negative_markers=["CD4"],
        parent_population="CD8+ T cells",
        source="HIPC 2016 Table 2",
    ),
    "cd8_cm": CellPopulationDefinition(
        name="CD8+ Central Memory",
        canonical_names=["CD8 CM", "CD8+ CM"],
        positive_markers=["CD3", "CD8", "CCR7"],
        negative_markers=["CD4", "CD45RA"],
        parent_population="CD8+ T cells",
        source="HIPC 2016 Table 2",
    ),
    "cd8_em": CellPopulationDefinition(
        name="CD8+ Effector Memory",
        canonical_names=["CD8 EM", "CD8+ EM"],
        positive_markers=["CD3", "CD8"],
        negative_markers=["CD4", "CD45RA", "CCR7"],
        parent_population="CD8+ T cells",
        source="HIPC 2016 Table 2",
    ),
    "cd8_temra": CellPopulationDefinition(
        name="CD8+ TEMRA",
        canonical_names=["CD8 TEMRA", "CD8+ EMRA"],
        positive_markers=["CD3", "CD8", "CD45RA"],
        negative_markers=["CD4", "CCR7"],
        parent_population="CD8+ T cells",
        source="HIPC 2016 Table 2",
    ),
    # B cell lineage
    "b_cells": CellPopulationDefinition(
        name="B cells",
        canonical_names=["B lymphocytes", "CD19+ cells", "CD20+ cells"],
        positive_markers=["CD19"],  # CD20 also acceptable
        negative_markers=["CD3"],
        optional_markers=["CD20"],  # Either CD19+ or CD20+ acceptable
        parent_population="Lymphocytes",
        notes="CD3- CD19+ (or CD20+); either marker acceptable, both not required",
        source="HIPC 2016 Table 1",
    ),
    "naive_b": CellPopulationDefinition(
        name="Naive B cells",
        canonical_names=["Naive B", "IgD+ B cells"],
        positive_markers=["CD19", "IgD"],
        negative_markers=["CD3", "CD27"],
        parent_population="B cells",
        source="HIPC 2016 Table 2",
    ),
    "memory_b": CellPopulationDefinition(
        name="Memory B cells",
        canonical_names=["Memory B", "CD27+ B cells"],
        positive_markers=["CD19", "CD27"],
        negative_markers=["CD3"],
        parent_population="B cells",
        source="HIPC 2016 Table 2",
    ),
    "transitional_b": CellPopulationDefinition(
        name="Transitional B cells",
        canonical_names=["Transitional B"],
        positive_markers=["CD19", "CD24", "CD38"],
        negative_markers=["CD3"],
        parent_population="B cells",
        notes="CD24hi CD38hi transitional B cells",
        source="HIPC 2016 Table 2",
    ),
    "plasmablasts": CellPopulationDefinition(
        name="Plasmablasts",
        canonical_names=["Plasma cells", "ASC"],
        positive_markers=["CD19", "CD27", "CD38"],
        negative_markers=["CD3"],
        parent_population="B cells",
        notes="CD27++ CD38++ plasmablasts/plasma cells",
        source="HIPC 2016 Table 2",
    ),
    # NK cell lineage
    "nk_cells": CellPopulationDefinition(
        name="NK cells",
        canonical_names=["Natural Killer cells", "CD56+ cells"],
        positive_markers=["CD56"],  # CD16 optional
        negative_markers=["CD3"],
        optional_markers=["CD16"],
        parent_population="Lymphocytes",
        notes="CD3- CD56+ and/or CD16+",
        source="HIPC 2016 Table 1",
    ),
    "cd56bright_nk": CellPopulationDefinition(
        name="CD56bright NK",
        canonical_names=["CD56bright", "Regulatory NK"],
        positive_markers=["CD56"],
        negative_markers=["CD3", "CD16"],
        parent_population="NK cells",
        notes="CD3- CD56bright CD16dim/- regulatory NK cells",
        source="HIPC 2016 Table 2",
    ),
    "cd56dim_nk": CellPopulationDefinition(
        name="CD56dim NK",
        canonical_names=["CD56dim", "Cytotoxic NK"],
        positive_markers=["CD56", "CD16"],
        negative_markers=["CD3"],
        parent_population="NK cells",
        notes="CD3- CD56dim CD16+ cytotoxic NK cells",
        source="HIPC 2016 Table 2",
    ),
    # Monocyte lineage
    "monocytes": CellPopulationDefinition(
        name="Monocytes",
        canonical_names=["CD14+ cells", "Mono"],
        positive_markers=["CD14"],
        negative_markers=["CD3"],
        parent_population="Leukocytes",
        notes="CD14+ monocytes",
        source="HIPC 2016 Table 1",
    ),
    "classical_monocytes": CellPopulationDefinition(
        name="Classical Monocytes",
        canonical_names=["Classical Mono", "CD14++ CD16-"],
        positive_markers=["CD14"],
        negative_markers=["CD16"],
        parent_population="Monocytes",
        notes="CD14++ CD16- classical monocytes",
        source="HIPC 2016 Table 2",
    ),
    "intermediate_monocytes": CellPopulationDefinition(
        name="Intermediate Monocytes",
        canonical_names=["Intermediate Mono", "CD14++ CD16+"],
        positive_markers=["CD14", "CD16"],
        negative_markers=[],
        parent_population="Monocytes",
        notes="CD14++ CD16+ intermediate monocytes",
        source="HIPC 2016 Table 2",
    ),
    "nonclassical_monocytes": CellPopulationDefinition(
        name="Non-classical Monocytes",
        canonical_names=["Non-classical Mono", "CD14dim CD16++"],
        positive_markers=["CD16"],
        negative_markers=[],
        parent_population="Monocytes",
        notes="CD14dim CD16++ non-classical monocytes",
        source="HIPC 2016 Table 2",
    ),
}


@dataclass
class ABTestResult:
    """Result of A/B comparison for a single prediction."""

    test_case_id: str
    model: str

    # Scores against each standard
    hipc_score: float = 0.0
    omip_score: float = 0.0

    # Detailed matches
    hipc_matches: list[str] = field(default_factory=list)
    hipc_misses: list[str] = field(default_factory=list)
    omip_matches: list[str] = field(default_factory=list)
    omip_misses: list[str] = field(default_factory=list)

    # Cases where prediction matches HIPC but not OMIP (or vice versa)
    hipc_only_matches: list[str] = field(default_factory=list)
    omip_only_matches: list[str] = field(default_factory=list)

    # Marker logic comparison
    correct_negative_markers: list[str] = field(default_factory=list)
    missing_negative_markers: list[str] = field(default_factory=list)

    @property
    def hipc_advantage(self) -> float:
        """How much better HIPC scoring is vs OMIP."""
        return self.hipc_score - self.omip_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "model": self.model,
            "hipc_score": self.hipc_score,
            "omip_score": self.omip_score,
            "hipc_advantage": self.hipc_advantage,
            "hipc_matches": self.hipc_matches,
            "hipc_misses": self.hipc_misses,
            "omip_matches": self.omip_matches,
            "omip_misses": self.omip_misses,
            "hipc_only_matches": self.hipc_only_matches,
            "omip_only_matches": self.omip_only_matches,
            "correct_negative_markers": self.correct_negative_markers,
            "missing_negative_markers": self.missing_negative_markers,
        }


@dataclass
class ABTestSummary:
    """Aggregate results across all test cases."""

    model: str
    n_test_cases: int = 0

    # Aggregate scores
    mean_hipc_score: float = 0.0
    mean_omip_score: float = 0.0
    mean_hipc_advantage: float = 0.0

    # Win rates
    hipc_wins: int = 0  # Cases where HIPC score > OMIP score
    omip_wins: int = 0  # Cases where OMIP score > HIPC score
    ties: int = 0

    # Negative marker analysis
    negative_marker_recall: float = 0.0  # How often LLM includes required negatives

    # Individual results
    results: list[ABTestResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_test_cases": self.n_test_cases,
            "mean_hipc_score": self.mean_hipc_score,
            "mean_omip_score": self.mean_omip_score,
            "mean_hipc_advantage": self.mean_hipc_advantage,
            "hipc_wins": self.hipc_wins,
            "omip_wins": self.omip_wins,
            "ties": self.ties,
            "hipc_win_rate": self.hipc_wins / self.n_test_cases if self.n_test_cases else 0,
            "negative_marker_recall": self.negative_marker_recall,
        }


class ABTester:
    """
    A/B Testing framework for comparing gating predictions.

    Evaluates LLM predictions against both:
    1. HIPC 2016 expert-validated definitions
    2. OMIP paper-specific ground truth
    """

    def __init__(self):
        self.hipc_definitions = HIPC_CELL_DEFINITIONS

    def normalize_population_name(self, name: str) -> str:
        """Normalize population name for matching."""
        normalized = name.lower().strip()
        # Remove common variations
        normalized = normalized.replace("_", " ").replace("-", " ")
        normalized = normalized.replace(" cells", "").replace(" cell", "")
        normalized = normalized.replace("lymphocytes", "").replace("lymphocyte", "")
        normalized = " ".join(normalized.split())  # Collapse whitespace
        return normalized

    def find_hipc_match(self, gate_name: str) -> CellPopulationDefinition | None:
        """Find matching HIPC definition for a gate name."""
        normalized = self.normalize_population_name(gate_name)

        # Keyword patterns to match against HIPC definitions
        # Maps keywords to HIPC definition keys
        keyword_matches = {
            "t cell": "t_cells",
            "t ": "t_cells",
            "cd3+": "t_cells",
            "cd4+ t": "cd4_t_cells",
            "cd4 t": "cd4_t_cells",
            "helper t": "cd4_t_cells",
            "cd8+ t": "cd8_t_cells",
            "cd8 t": "cd8_t_cells",
            "cytotoxic t": "cd8_t_cells",
            "naive cd4": "cd4_naive",
            "cd4 naive": "cd4_naive",
            "cd4+ naive": "cd4_naive",
            "naive cd8": "cd8_naive",
            "cd8 naive": "cd8_naive",
            "cd8+ naive": "cd8_naive",
            "central memory cd4": "cd4_cm",
            "cd4 cm": "cd4_cm",
            "cd4+ cm": "cd4_cm",
            "central memory cd8": "cd8_cm",
            "cd8 cm": "cd8_cm",
            "cd8+ cm": "cd8_cm",
            "effector memory cd4": "cd4_em",
            "cd4 em": "cd4_em",
            "cd4+ em": "cd4_em",
            "effector memory cd8": "cd8_em",
            "cd8 em": "cd8_em",
            "cd8+ em": "cd8_em",
            "temra cd4": "cd4_temra",
            "cd4 temra": "cd4_temra",
            "cd4+ temra": "cd4_temra",
            "temra cd8": "cd8_temra",
            "cd8 temra": "cd8_temra",
            "cd8+ temra": "cd8_temra",
            "b cell": "b_cells",
            "b ": "b_cells",
            "cd19+": "b_cells",
            "cd20+": "b_cells",
            "naive b": "naive_b",
            "memory b": "memory_b",
            "transitional b": "transitional_b",
            "plasmablast": "plasmablasts",
            "plasma": "plasmablasts",
            "nk cell": "nk_cells",
            "nk ": "nk_cells",
            "natural killer": "nk_cells",
            "cd56+": "nk_cells",
            "cd56bright": "cd56bright_nk",
            "cd56dim": "cd56dim_nk",
            "monocyte": "monocytes",
            "cd14+": "monocytes",
            "classical mono": "classical_monocytes",
            "non classical": "nonclassical_monocytes",
            "nonclassical": "nonclassical_monocytes",
            "intermediate mono": "intermediate_monocytes",
        }

        # First try exact match with canonical names
        for key, defn in self.hipc_definitions.items():
            if self.normalize_population_name(defn.name) == normalized:
                return defn
            for alt in defn.canonical_names:
                if self.normalize_population_name(alt) == normalized:
                    return defn

        # Then try keyword matching
        for keyword, defn_key in keyword_matches.items():
            if keyword in normalized:
                if defn_key in self.hipc_definitions:
                    return self.hipc_definitions[defn_key]

        return None

    def check_marker_logic(
        self,
        predicted_markers: list[dict],
        hipc_defn: CellPopulationDefinition,
    ) -> tuple[list[str], list[str]]:
        """
        Check if predicted marker logic matches HIPC definition.

        Returns:
            Tuple of (correct_negatives, missing_negatives)
        """
        correct = []
        missing = []

        # Extract predicted negative markers
        pred_negatives = {
            m["marker"].lower()
            for m in predicted_markers
            if not m.get("positive", True)
        }

        # Check required negatives from HIPC
        for neg_marker in hipc_defn.negative_markers:
            if neg_marker.lower() in pred_negatives:
                correct.append(neg_marker)
            else:
                missing.append(neg_marker)

        return correct, missing

    def evaluate_prediction(
        self,
        predicted_hierarchy: dict,
        omip_ground_truth: dict,
        test_case_id: str,
        model: str,
    ) -> ABTestResult:
        """
        Evaluate a single prediction against both standards.

        Args:
            predicted_hierarchy: LLM's predicted gating hierarchy
            omip_ground_truth: OMIP paper ground truth
            test_case_id: Test case identifier
            model: Model name

        Returns:
            ABTestResult with comparison metrics
        """
        result = ABTestResult(test_case_id=test_case_id, model=model)

        # Extract all gates from prediction
        predicted_gates = self._extract_gates(predicted_hierarchy)
        omip_gates = self._extract_gates(omip_ground_truth)

        # Score against OMIP
        omip_matches = []
        for pred_gate in predicted_gates:
            pred_name = self.normalize_population_name(pred_gate["name"])
            for omip_gate in omip_gates:
                omip_name = self.normalize_population_name(omip_gate["name"])
                if pred_name == omip_name:
                    omip_matches.append(pred_gate["name"])
                    break

        result.omip_matches = omip_matches
        result.omip_misses = [
            g["name"] for g in omip_gates
            if self.normalize_population_name(g["name"]) not in
               [self.normalize_population_name(m) for m in omip_matches]
        ]
        result.omip_score = len(omip_matches) / len(omip_gates) if omip_gates else 0

        # Score against HIPC
        hipc_matches = []
        all_correct_negatives = []
        all_missing_negatives = []

        for pred_gate in predicted_gates:
            hipc_defn = self.find_hipc_match(pred_gate["name"])
            if hipc_defn:
                hipc_matches.append(pred_gate["name"])

                # Check marker logic
                if pred_gate.get("marker_logic"):
                    correct, missing = self.check_marker_logic(
                        pred_gate["marker_logic"],
                        hipc_defn,
                    )
                    all_correct_negatives.extend(correct)
                    all_missing_negatives.extend(missing)

        result.hipc_matches = hipc_matches
        result.correct_negative_markers = all_correct_negatives
        result.missing_negative_markers = all_missing_negatives

        # Calculate HIPC score based on name matching and marker logic
        # Name score: what fraction of predicted gates match HIPC definitions
        name_score = len(hipc_matches) / len(predicted_gates) if predicted_gates else 0

        # Marker score: how well do matched gates have correct negative markers
        # Only calculate if we have matches to evaluate
        if all_correct_negatives or all_missing_negatives:
            marker_score = len(all_correct_negatives) / (len(all_correct_negatives) + len(all_missing_negatives))
        elif hipc_matches:
            # We have name matches but no marker logic to evaluate
            marker_score = name_score  # Use name score as proxy
        else:
            # No matches at all - score is 0
            marker_score = 0.0

        # Combined score - weight name matching more heavily
        result.hipc_score = (name_score * 0.7 + marker_score * 0.3) if hipc_matches else name_score

        # Find exclusive matches
        hipc_match_set = set(self.normalize_population_name(m) for m in hipc_matches)
        omip_match_set = set(self.normalize_population_name(m) for m in omip_matches)

        result.hipc_only_matches = [
            m for m in hipc_matches
            if self.normalize_population_name(m) not in omip_match_set
        ]
        result.omip_only_matches = [
            m for m in omip_matches
            if self.normalize_population_name(m) not in hipc_match_set
        ]

        return result

    def _extract_gates(self, hierarchy: dict) -> list[dict]:
        """Recursively extract all gates from hierarchy."""
        gates = []

        def traverse(node: dict):
            gates.append(node)
            for child in node.get("children", []):
                traverse(child)

        if "root" in hierarchy:
            traverse(hierarchy["root"])
        elif "name" in hierarchy:
            traverse(hierarchy)

        return gates

    def run_ab_test(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        model: str,
    ) -> ABTestSummary:
        """
        Run A/B test across multiple predictions.

        Args:
            predictions: List of {test_case_id, predicted_hierarchy}
            ground_truths: List of {test_case_id, gating_hierarchy}
            model: Model name

        Returns:
            ABTestSummary with aggregate results
        """
        # Create lookup for ground truths
        gt_lookup = {gt["test_case_id"]: gt["gating_hierarchy"] for gt in ground_truths}

        results = []
        for pred in predictions:
            test_id = pred["test_case_id"]
            if test_id not in gt_lookup:
                continue

            result = self.evaluate_prediction(
                pred["predicted_hierarchy"],
                gt_lookup[test_id],
                test_id,
                model,
            )
            results.append(result)

        # Aggregate
        summary = ABTestSummary(model=model, n_test_cases=len(results), results=results)

        if results:
            summary.mean_hipc_score = sum(r.hipc_score for r in results) / len(results)
            summary.mean_omip_score = sum(r.omip_score for r in results) / len(results)
            summary.mean_hipc_advantage = sum(r.hipc_advantage for r in results) / len(results)

            for r in results:
                if r.hipc_score > r.omip_score:
                    summary.hipc_wins += 1
                elif r.omip_score > r.hipc_score:
                    summary.omip_wins += 1
                else:
                    summary.ties += 1

            # Negative marker recall
            total_correct = sum(len(r.correct_negative_markers) for r in results)
            total_expected = total_correct + sum(len(r.missing_negative_markers) for r in results)
            summary.negative_marker_recall = total_correct / total_expected if total_expected else 1.0

        return summary


def save_hipc_definitions(output_path: Path) -> None:
    """Export HIPC definitions to JSON for reference."""
    definitions = {
        key: defn.model_dump()
        for key, defn in HIPC_CELL_DEFINITIONS.items()
    }

    with open(output_path, "w") as f:
        json.dump(definitions, f, indent=2)

    print(f"Saved HIPC definitions to {output_path}")


if __name__ == "__main__":
    # Export HIPC definitions
    output_dir = Path(__file__).parent.parent.parent / "data" / "reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_hipc_definitions(output_dir / "hipc_2016_definitions.json")
