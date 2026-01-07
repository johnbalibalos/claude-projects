"""
Generate panel design test cases for ablation study.

Test cases should include:
1. In-distribution: Fluorophore pairs that appear in OMIP panels (retrieval should work)
2. Out-of-distribution: Novel pairs NOT in any OMIP (tests generalization)
3. Adversarial: Pairs where OMIP precedent conflicts with spectral physics
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import random
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flow_panel_optimizer.data.fluorophore_database import (
    FLUOROPHORE_DATABASE,
    get_fluorophore,
    calculate_spectral_overlap,
    get_known_overlap,
    list_all_fluorophores,
)


class TestCaseType(Enum):
    IN_DISTRIBUTION = "in_distribution"      # Pairs found in OMIP corpus
    NEAR_DISTRIBUTION = "near_distribution"  # Same markers, ~50% different fluorophores
    OUT_OF_DISTRIBUTION = "out_of_distribution"  # Novel pairs
    ADVERSARIAL = "adversarial"              # OMIP says OK but physics says bad (or vice versa)


@dataclass
class PanelDesignTestCase:
    """A single panel design evaluation task.

    Quality metrics follow EasyPanel methodology:
    - total_similarity_score: Sum of ALL pairwise similarities (lower is better)
    - avg_similarity: total_similarity_score / num_pairs
    - n_critical_pairs: Pairs with similarity > 0.90 (should be 0)
    - quality_rating: Based on avg_similarity thresholds
    """
    id: str
    case_type: TestCaseType
    biological_question: str  # e.g., "Design panel for T cell exhaustion"
    required_markers: list[str]  # e.g., ["CD3", "CD4", "CD8", "PD-1", "TIM-3"]
    marker_expression: dict[str, str]  # marker -> "high"/"medium"/"low"
    candidate_fluorophores: list[str]  # Available fluorophores to choose from
    ground_truth_assignments: dict[str, str]  # Optimal marker -> fluorophore mapping
    ground_truth_complexity_index: float  # Total similarity score for optimal assignment
    ground_truth_avg_similarity: float = 0.0  # Average pairwise similarity
    ground_truth_n_critical_pairs: int = 0  # Pairs with similarity > 0.90
    ground_truth_quality_rating: str = "unknown"  # EXCELLENT/GOOD/ACCEPTABLE/POOR
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "case_type": self.case_type.value,
            "biological_question": self.biological_question,
            "required_markers": self.required_markers,
            "marker_expression": self.marker_expression,
            "candidate_fluorophores": self.candidate_fluorophores,
            "ground_truth_assignments": self.ground_truth_assignments,
            "ground_truth_complexity_index": self.ground_truth_complexity_index,
            "ground_truth_avg_similarity": self.ground_truth_avg_similarity,
            "ground_truth_n_critical_pairs": self.ground_truth_n_critical_pairs,
            "ground_truth_quality_rating": self.ground_truth_quality_rating,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PanelDesignTestCase":
        return cls(
            id=data["id"],
            case_type=TestCaseType(data["case_type"]),
            biological_question=data["biological_question"],
            required_markers=data["required_markers"],
            marker_expression=data["marker_expression"],
            candidate_fluorophores=data["candidate_fluorophores"],
            ground_truth_assignments=data["ground_truth_assignments"],
            ground_truth_complexity_index=data["ground_truth_complexity_index"],
            ground_truth_avg_similarity=data.get("ground_truth_avg_similarity", 0.0),
            ground_truth_n_critical_pairs=data.get("ground_truth_n_critical_pairs", 0),
            ground_truth_quality_rating=data.get("ground_truth_quality_rating", "unknown"),
            notes=data.get("notes"),
        )


@dataclass
class TestSuite:
    """Collection of test cases for ablation study."""
    name: str
    test_cases: list[PanelDesignTestCase]

    def filter_by_type(self, case_type: TestCaseType) -> list[PanelDesignTestCase]:
        return [tc for tc in self.test_cases if tc.case_type == case_type]

    def to_json(self, path: Path) -> None:
        """Save test suite to JSON file."""
        data = {
            "name": self.name,
            "test_cases": [tc.to_dict() for tc in self.test_cases]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "TestSuite":
        """Load test suite from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            name=data["name"],
            test_cases=[PanelDesignTestCase.from_dict(tc) for tc in data["test_cases"]]
        )


# OMIP panel data - extract fluorophore assignments
OMIP_PANEL_DATA = {
    "OMIP-030": {
        "markers": ["CD3", "CD4", "CD8", "CD45RA", "CD127", "CD25", "CCR7", "CD161", "CXCR3", "Viability"],
        "assignments": {
            "CD3": "Pacific Blue",
            "CD4": "PerCP-Cy5.5",
            "CD8": "APC-Fire750",
            "CD45RA": "FITC",
            "CD127": "PE",
            "CD25": "PE-Cy7",
            "CCR7": "PE-Cy5",
            "CD161": "BV711",
            "CXCR3": "APC",
            "Viability": "LIVE/DEAD Blue",
        },
        "expression": {
            "CD3": "high", "CD4": "high", "CD8": "high", "CD45RA": "high",
            "CD127": "medium", "CD25": "medium", "CCR7": "medium",
            "CD161": "medium", "CXCR3": "low", "Viability": "high",
        },
        "question": "Design a 10-color panel for human T cell subset analysis"
    },
    "OMIP-047": {
        "markers": ["Viability", "CD3", "CD14", "CD19", "CD20", "CD27", "IgD", "CD38", "CD21", "CD10", "IgG", "IgA", "CXCR3", "CCR7", "IL-21R", "Ki67"],
        "assignments": {
            "Viability": "LIVE/DEAD Aqua",
            "CD3": "BV421",
            "CD14": "BV421",
            "CD19": "BV785",
            "CD20": "APC-Cy7",
            "CD27": "BV605",
            "IgD": "PE-Cy7",
            "CD38": "BB515",
            "CD21": "BV711",
            "CD10": "PE",
            "IgG": "PerCP-Cy5.5",
            "IgA": "APC",
            "CXCR3": "FITC",
            "CCR7": "PE-Dazzle 594",
            "IL-21R": "BV510",
            "Ki67": "Alexa Fluor 700",
        },
        "expression": {
            "Viability": "high", "CD3": "high", "CD14": "high", "CD19": "high",
            "CD20": "high", "CD27": "medium", "IgD": "medium", "CD38": "medium",
            "CD21": "medium", "CD10": "low", "IgG": "medium", "IgA": "medium",
            "CXCR3": "low", "CCR7": "low", "IL-21R": "low", "Ki67": "low",
        },
        "question": "Design a 16-color panel for B cell phenotyping"
    },
    "OMIP-063": {
        "markers": ["Viability", "CD45", "CD3", "CD4", "CD8", "CD19", "CD56", "CD14", "CD16", "HLA-DR", "CD45RA", "CCR7", "CD27", "CD28", "CD57", "CD127", "CD25", "CD38", "CD11c", "CD123"],
        "assignments": {
            "Viability": "LIVE/DEAD Blue",
            "CD45": "BUV395",
            "CD3": "BUV496",
            "CD4": "BV750",
            "CD8": "BUV805",
            "CD19": "BV480",
            "CD56": "BV605",
            "CD14": "BV650",
            "CD16": "APC-Cy7",
            "HLA-DR": "BV785",
            "CD45RA": "BV510",
            "CCR7": "PE",
            "CD27": "BV421",
            "CD28": "BB515",
            "CD57": "FITC",
            "CD127": "PE-Cy7",
            "CD25": "PE-Dazzle 594",
            "CD38": "PerCP-Cy5.5",
            "CD11c": "APC",
            "CD123": "BV711",
        },
        "expression": {
            "Viability": "high", "CD45": "high", "CD3": "high", "CD4": "high",
            "CD8": "high", "CD19": "high", "CD56": "high", "CD14": "high",
            "CD16": "medium", "HLA-DR": "high", "CD45RA": "high", "CCR7": "medium",
            "CD27": "medium", "CD28": "medium", "CD57": "medium", "CD127": "medium",
            "CD25": "medium", "CD38": "medium", "CD11c": "high", "CD123": "medium",
        },
        "question": "Design a 20-color panel for broad human immunophenotyping"
    },
    "OMIP-069": {
        "markers": ["CD45", "Viability", "CD3", "CD4", "CD8", "TCR-gd", "CD45RA", "CD45RO", "CCR7", "CD27", "CD28", "CD57", "CD95", "CD25", "CD127", "CD39", "CXCR3", "CCR4", "CCR6", "CXCR5", "CD19", "CD20", "IgD", "CD38", "CD56", "CD16", "NKG2D", "CD14", "HLA-DR", "CD11c", "CD123"],
        "assignments": {
            "CD45": "BUV395",
            "Viability": "LIVE/DEAD Blue",
            "CD3": "BUV496",
            "CD4": "BUV563",
            "CD8": "BUV805",
            "TCR-gd": "BV421",
            "CD45RA": "BV570",
            "CD45RO": "BV650",
            "CCR7": "BV785",
            "CD27": "BV510",
            "CD28": "BB515",
            "CD57": "FITC",
            "CD95": "PE-Cy5",
            "CD25": "PE-Cy7",
            "CD127": "PE-Dazzle 594",
            "CD39": "PE-Cy5.5",
            "CXCR3": "BV711",
            "CCR4": "PE",
            "CCR6": "BV605",
            "CXCR5": "BV750",
            "CD19": "BUV661",
            "CD20": "APC-R700",
            "IgD": "PerCP-Cy5.5",
            "CD38": "APC-Fire750",
            "CD56": "BUV737",
            "CD16": "APC-Cy7",
            "NKG2D": "APC",
            "CD14": "BUV615",
            "HLA-DR": "Alexa Fluor 700",
            "CD11c": "PE-CF594",
            "CD123": "Super Bright 436",
        },
        "expression": {
            "CD45": "high", "Viability": "high", "CD3": "high", "CD4": "high",
            "CD8": "high", "TCR-gd": "medium", "CD45RA": "high", "CD45RO": "high",
            "CCR7": "medium", "CD27": "medium", "CD28": "medium", "CD57": "medium",
            "CD95": "medium", "CD25": "medium", "CD127": "medium", "CD39": "low",
            "CXCR3": "low", "CCR4": "low", "CCR6": "low", "CXCR5": "low",
            "CD19": "high", "CD20": "high", "IgD": "medium", "CD38": "medium",
            "CD56": "high", "CD16": "medium", "NKG2D": "low", "CD14": "high",
            "HLA-DR": "high", "CD11c": "high", "CD123": "medium",
        },
        "question": "Design a 31-color panel for deep immunophenotyping"
    },
}


def _calculate_panel_complexity(assignments: dict[str, str]) -> float:
    """Calculate total similarity score for a panel assignment.

    Uses the EasyPanel methodology: sum of ALL pairwise similarities.
    This is the proper metric for panel quality - lower is better.

    References:
        - EasyPanel: https://flow-cytometry.net/rules-for-spectral-panel-design/
        - Sum of ALL pairs, not just high-overlap pairs
    """
    fluorophores = list(assignments.values())

    if len(fluorophores) < 2:
        return 0.0

    total_similarity = 0.0
    for i, f1 in enumerate(fluorophores):
        for j, f2 in enumerate(fluorophores):
            if i >= j:
                continue
            sim = get_known_overlap(f1, f2)
            if sim is None:
                fluor1 = get_fluorophore(f1)
                fluor2 = get_fluorophore(f2)
                if fluor1 and fluor2:
                    sim = calculate_spectral_overlap(fluor1, fluor2)
                else:
                    sim = 0.0
            # Sum ALL pairwise similarities (EasyPanel method)
            total_similarity += sim

    return round(total_similarity, 4)


def _calculate_panel_metrics(assignments: dict[str, str]) -> dict:
    """Calculate comprehensive quality metrics for a panel assignment.

    Uses EasyPanel methodology:
    - total_similarity_score: Sum of ALL pairwise similarities
    - avg_similarity: total_similarity_score / num_pairs
    - max_similarity: Highest pairwise similarity
    - n_critical_pairs: Count of pairs with similarity > 0.90
    - quality_rating: Based on avg_similarity thresholds

    Quality thresholds (EasyPanel):
    - EXCELLENT: avg_similarity < 0.25, no critical pairs
    - GOOD: avg_similarity < 0.35, no critical pairs
    - ACCEPTABLE: avg_similarity < 0.50, ≤1 critical pair
    - POOR: otherwise

    Returns:
        dict with all quality metrics
    """
    fluorophores = list(assignments.values())
    n = len(fluorophores)

    if n < 2:
        return {
            "total_similarity_score": 0.0,
            "avg_similarity": 0.0,
            "max_similarity": 0.0,
            "n_critical_pairs": 0,
            "n_high_risk_pairs": 0,
            "quality_rating": "EXCELLENT",
            "n_pairs": 0,
        }

    total_similarity = 0.0
    max_similarity = 0.0
    n_critical = 0
    n_high_risk = 0
    n_pairs = n * (n - 1) // 2

    for i, f1 in enumerate(fluorophores):
        for j, f2 in enumerate(fluorophores):
            if i >= j:
                continue
            sim = get_known_overlap(f1, f2)
            if sim is None:
                fluor1 = get_fluorophore(f1)
                fluor2 = get_fluorophore(f2)
                if fluor1 and fluor2:
                    sim = calculate_spectral_overlap(fluor1, fluor2)
                else:
                    sim = 0.0

            total_similarity += sim
            max_similarity = max(max_similarity, sim)

            if sim > 0.90:
                n_critical += 1
            elif sim > 0.70:
                n_high_risk += 1

    avg_similarity = total_similarity / n_pairs if n_pairs > 0 else 0.0

    # Determine quality rating (EasyPanel thresholds)
    if avg_similarity < 0.25 and n_critical == 0:
        quality_rating = "EXCELLENT"
    elif avg_similarity < 0.35 and n_critical == 0:
        quality_rating = "GOOD"
    elif avg_similarity < 0.50 and n_critical <= 1:
        quality_rating = "ACCEPTABLE"
    else:
        quality_rating = "POOR"

    return {
        "total_similarity_score": round(total_similarity, 4),
        "avg_similarity": round(avg_similarity, 4),
        "max_similarity": round(max_similarity, 4),
        "n_critical_pairs": n_critical,
        "n_high_risk_pairs": n_high_risk,
        "quality_rating": quality_rating,
        "n_pairs": n_pairs,
    }


def _get_all_omip_fluorophore_pairs() -> set[tuple[str, str]]:
    """Extract all fluorophore pairs used in OMIP panels."""
    pairs = set()
    for panel_data in OMIP_PANEL_DATA.values():
        fluors = list(panel_data["assignments"].values())
        for i, f1 in enumerate(fluors):
            for f2 in fluors[i+1:]:
                pairs.add(tuple(sorted([f1, f2])))
    return pairs


def _get_available_fluorophores() -> list[str]:
    """Get all fluorophores in our database."""
    return list(FLUOROPHORE_DATABASE.keys())


def generate_in_distribution_cases(n: int = 20) -> list[PanelDesignTestCase]:
    """
    Generate test cases using fluorophore pairs from OMIP panels.

    These cases SHOULD be solvable via retrieval - the pairs exist in training data.
    """
    test_cases = []

    # Use subsets of OMIP panels
    panel_subsets = [
        # Small T cell panels from OMIP-030
        {
            "id": "in_dist_tcell_naive",
            "question": "Design a 5-color panel for naive vs memory T cell distinction",
            "markers": ["CD3", "CD4", "CD45RA", "CCR7", "Viability"],
            "source": "OMIP-030"
        },
        {
            "id": "in_dist_tcell_treg",
            "question": "Design a 6-color panel for regulatory T cell identification",
            "markers": ["CD3", "CD4", "CD25", "CD127", "CXCR3", "Viability"],
            "source": "OMIP-030"
        },
        # B cell panels from OMIP-047
        {
            "id": "in_dist_bcell_basic",
            "question": "Design a 6-color panel for basic B cell phenotyping",
            "markers": ["CD19", "CD20", "CD27", "IgD", "CD38", "Viability"],
            "source": "OMIP-047"
        },
        {
            "id": "in_dist_bcell_memory",
            "question": "Design an 8-color panel for B cell memory subsets",
            "markers": ["CD19", "CD20", "CD27", "IgD", "IgG", "IgA", "CD38", "Viability"],
            "source": "OMIP-047"
        },
        # Myeloid panels from OMIP-063
        {
            "id": "in_dist_myeloid_basic",
            "question": "Design a 6-color panel for monocyte/DC identification",
            "markers": ["CD14", "CD16", "CD11c", "CD123", "HLA-DR", "Viability"],
            "source": "OMIP-063"
        },
        # Full lineage panels
        {
            "id": "in_dist_lineage_basic",
            "question": "Design an 8-color lineage panel",
            "markers": ["CD3", "CD4", "CD8", "CD19", "CD56", "CD14", "CD45", "Viability"],
            "source": "OMIP-063"
        },
        # Larger panels from OMIP-069
        {
            "id": "in_dist_tcell_full",
            "question": "Design a 10-color panel for comprehensive T cell phenotyping",
            "markers": ["CD3", "CD4", "CD8", "CD45RA", "CD45RO", "CCR7", "CD27", "CD28", "CD57", "Viability"],
            "source": "OMIP-069"
        },
        {
            "id": "in_dist_th_subsets",
            "question": "Design an 8-color panel for T helper subset identification",
            "markers": ["CD3", "CD4", "CXCR3", "CCR4", "CCR6", "CXCR5", "CD45RA", "Viability"],
            "source": "OMIP-069"
        },
        {
            "id": "in_dist_nk_basic",
            "question": "Design a 6-color panel for NK cell analysis",
            "markers": ["CD3", "CD56", "CD16", "NKG2D", "CD45", "Viability"],
            "source": "OMIP-069"
        },
        {
            "id": "in_dist_activation",
            "question": "Design a 7-color panel for T cell activation status",
            "markers": ["CD3", "CD4", "CD8", "CD25", "CD38", "HLA-DR", "Viability"],
            "source": "OMIP-069"
        },
    ]

    # Generate more variations by combining markers from same source
    additional_combos = [
        ("in_dist_tcell_cd8", "Design a CD8+ T cell memory panel",
         ["CD3", "CD8", "CD45RA", "CCR7", "CD27", "CD57", "Viability"], "OMIP-069"),
        ("in_dist_bcell_dev", "Design a B cell development panel",
         ["CD19", "CD20", "CD10", "CD21", "IgD", "Viability"], "OMIP-047"),
        ("in_dist_immune_broad", "Design a broad immune profiling panel",
         ["CD3", "CD19", "CD56", "CD14", "HLA-DR", "CD45", "Viability"], "OMIP-063"),
        ("in_dist_treg_func", "Design a Treg functional panel",
         ["CD3", "CD4", "CD25", "CD127", "CD39", "Viability"], "OMIP-069"),
    ]

    for combo in additional_combos:
        panel_subsets.append({
            "id": combo[0],
            "question": combo[1],
            "markers": combo[2],
            "source": combo[3]
        })

    for subset in panel_subsets[:n]:
        source = subset["source"]
        source_panel = OMIP_PANEL_DATA[source]

        # Build ground truth from source panel
        ground_truth = {
            m: source_panel["assignments"][m]
            for m in subset["markers"]
            if m in source_panel["assignments"]
        }

        expression = {
            m: source_panel["expression"].get(m, "medium")
            for m in subset["markers"]
        }

        # Candidate fluorophores: all from database
        candidates = _get_available_fluorophores()

        # Calculate comprehensive metrics (EasyPanel methodology)
        metrics = _calculate_panel_metrics(ground_truth)

        test_cases.append(PanelDesignTestCase(
            id=subset["id"],
            case_type=TestCaseType.IN_DISTRIBUTION,
            biological_question=subset["question"],
            required_markers=subset["markers"],
            marker_expression=expression,
            candidate_fluorophores=candidates,
            ground_truth_assignments=ground_truth,
            ground_truth_complexity_index=metrics["total_similarity_score"],
            ground_truth_avg_similarity=metrics["avg_similarity"],
            ground_truth_n_critical_pairs=metrics["n_critical_pairs"],
            ground_truth_quality_rating=metrics["quality_rating"],
            notes=f"Derived from {source}"
        ))

    return test_cases


def generate_near_distribution_cases(n: int = 15) -> list[PanelDesignTestCase]:
    """
    Generate test cases with same markers as OMIP but ~50% different fluorophores.

    These cases test generalization to "nearby" panels - same markers, similar
    but not identical fluorophore choices. This is a realistic scenario where
    a user has similar markers to published panels but different instrument
    or fluorophore availability.

    Key characteristics:
    - Same markers as an OMIP panel subset
    - At least 50% of fluorophores are different from OMIP
    - Alternative fluorophores are spectrally reasonable substitutes
    """
    test_cases = []
    all_fluors = _get_available_fluorophores()

    # Define fluorophore substitutions (spectrally similar alternatives)
    FLUOROPHORE_SUBSTITUTES = {
        # Violet laser alternatives
        "BV421": ["Pacific Blue", "eFluor 450", "V450"],
        "Pacific Blue": ["BV421", "eFluor 450"],
        "BV510": ["V500", "BV480"],
        "BV605": ["BV570", "BV650"],
        "BV711": ["BV750", "BV650"],
        "BV785": ["BV750", "BV711"],

        # Blue laser alternatives
        "FITC": ["BB515", "Alexa Fluor 488"],
        "BB515": ["FITC", "Alexa Fluor 488"],
        "PerCP-Cy5.5": ["PE-Cy5.5", "PerCP"],
        "PE": ["PE-CF594", "PE-Dazzle 594"],
        "PE-Cy7": ["PE-Cy5.5", "PE-CF594"],
        "PE-Cy5": ["PE-Cy5.5", "PerCP-Cy5.5"],

        # Red laser alternatives
        "APC": ["Alexa Fluor 647", "APC-R700"],
        "Alexa Fluor 647": ["APC", "APC-R700"],
        "APC-Cy7": ["APC-Fire750", "Alexa Fluor 700"],
        "APC-Fire750": ["APC-Cy7", "Alexa Fluor 700"],
        "Alexa Fluor 700": ["APC-R700", "APC-Cy7"],

        # UV laser alternatives
        "BUV395": ["BUV496", "BUV563"],
        "BUV496": ["BUV395", "BUV563"],
        "BUV805": ["BUV737", "BUV661"],
    }

    # Source panels to modify
    near_dist_panels = [
        {
            "id": "near_dist_tcell_basic",
            "source": "OMIP-030",
            "markers": ["CD3", "CD4", "CD8", "CD45RA", "CCR7", "CD127"],
            "question": "Design a 6-color T cell panel (similar to OMIP-030)",
            "swap_percentage": 0.5,  # Swap ~50% of fluorophores
        },
        {
            "id": "near_dist_bcell_phenotype",
            "source": "OMIP-047",
            "markers": ["CD19", "CD20", "CD27", "IgD", "CD38", "CD21"],
            "question": "Design a 6-color B cell panel (similar to OMIP-047)",
            "swap_percentage": 0.5,
        },
        {
            "id": "near_dist_lineage_panel",
            "source": "OMIP-063",
            "markers": ["CD3", "CD4", "CD8", "CD19", "CD56", "CD14", "HLA-DR"],
            "question": "Design a 7-color lineage panel (similar to OMIP-063)",
            "swap_percentage": 0.5,
        },
        {
            "id": "near_dist_tcell_memory",
            "source": "OMIP-069",
            "markers": ["CD3", "CD4", "CD8", "CD45RA", "CD45RO", "CCR7", "CD27", "CD28"],
            "question": "Design an 8-color T cell memory panel (similar to OMIP-069)",
            "swap_percentage": 0.5,
        },
        {
            "id": "near_dist_myeloid_dc",
            "source": "OMIP-063",
            "markers": ["CD14", "CD16", "CD11c", "CD123", "HLA-DR", "CD45"],
            "question": "Design a 6-color myeloid/DC panel (similar to OMIP-063)",
            "swap_percentage": 0.6,  # Swap 60%
        },
        {
            "id": "near_dist_activation",
            "source": "OMIP-069",
            "markers": ["CD3", "CD4", "CD8", "CD25", "CD38", "CD127"],
            "question": "Design a 6-color activation panel (similar to OMIP-069)",
            "swap_percentage": 0.5,
        },
    ]

    for panel_spec in near_dist_panels[:n]:
        source_data = OMIP_PANEL_DATA[panel_spec["source"]]

        # Get original OMIP assignments for these markers
        original_assignments = {
            m: source_data["assignments"].get(m)
            for m in panel_spec["markers"]
            if m in source_data["assignments"]
        }

        expression = {
            m: source_data["expression"].get(m, "medium")
            for m in panel_spec["markers"]
        }

        # Create modified assignments by swapping some fluorophores
        modified_assignments = {}
        markers_to_swap = list(original_assignments.keys())
        n_to_swap = max(1, int(len(markers_to_swap) * panel_spec["swap_percentage"]))

        # Swap first n_to_swap markers with alternatives
        for i, marker in enumerate(markers_to_swap):
            orig_fluor = original_assignments[marker]
            if orig_fluor is None:
                continue

            if i < n_to_swap and orig_fluor in FLUOROPHORE_SUBSTITUTES:
                # Find a substitute that isn't already used
                substitutes = FLUOROPHORE_SUBSTITUTES[orig_fluor]
                used_fluors = list(modified_assignments.values())

                for sub in substitutes:
                    if sub in all_fluors and sub not in used_fluors:
                        modified_assignments[marker] = sub
                        break
                else:
                    # No valid substitute found, keep original
                    modified_assignments[marker] = orig_fluor
            else:
                # Keep original
                modified_assignments[marker] = orig_fluor

        # Calculate how many were actually swapped
        n_same = sum(1 for m in modified_assignments
                     if modified_assignments[m] == original_assignments.get(m))
        fluor_overlap = n_same / len(modified_assignments) if modified_assignments else 0

        # Calculate comprehensive metrics
        metrics = _calculate_panel_metrics(modified_assignments)

        test_cases.append(PanelDesignTestCase(
            id=panel_spec["id"],
            case_type=TestCaseType.NEAR_DISTRIBUTION,
            biological_question=panel_spec["question"],
            required_markers=panel_spec["markers"],
            marker_expression=expression,
            candidate_fluorophores=all_fluors,
            ground_truth_assignments=modified_assignments,
            ground_truth_complexity_index=metrics["total_similarity_score"],
            ground_truth_avg_similarity=metrics["avg_similarity"],
            ground_truth_n_critical_pairs=metrics["n_critical_pairs"],
            ground_truth_quality_rating=metrics["quality_rating"],
            notes=f"Modified from {panel_spec['source']} - {fluor_overlap*100:.0f}% fluorophore overlap"
        ))

    return test_cases


def generate_out_of_distribution_cases(n: int = 20) -> list[PanelDesignTestCase]:
    """
    Generate test cases with fluorophore combinations NOT in any OMIP.

    These cases test whether the system can generalize beyond memorized panels.
    Critical for testing MCP value - retrieval should fail here.
    """
    test_cases = []

    omip_pairs = _get_all_omip_fluorophore_pairs()
    all_fluors = _get_available_fluorophores()

    # Novel biological questions not directly covered by existing OMIPs
    novel_panels = [
        {
            "id": "ood_exhaustion",
            "question": "Design a panel for T cell exhaustion markers",
            "markers": ["CD3", "CD8", "PD-1", "TIM-3", "LAG-3", "Viability"],
            "expression": {"CD3": "high", "CD8": "high", "PD-1": "low", "TIM-3": "low", "LAG-3": "low", "Viability": "high"}
        },
        {
            "id": "ood_ilc",
            "question": "Design a panel for innate lymphoid cell subsets",
            "markers": ["CD127", "CD161", "CRTH2", "CD117", "CD56", "Viability"],
            "expression": {"CD127": "medium", "CD161": "medium", "CRTH2": "low", "CD117": "low", "CD56": "high", "Viability": "high"}
        },
        {
            "id": "ood_tfh",
            "question": "Design a panel for T follicular helper cells",
            "markers": ["CD3", "CD4", "CXCR5", "PD-1", "ICOS", "Viability"],
            "expression": {"CD3": "high", "CD4": "high", "CXCR5": "low", "PD-1": "low", "ICOS": "low", "Viability": "high"}
        },
        {
            "id": "ood_mait",
            "question": "Design a panel for MAIT cell identification",
            "markers": ["CD3", "CD161", "TCR-Va7.2", "CD8", "CD4", "Viability"],
            "expression": {"CD3": "high", "CD161": "medium", "TCR-Va7.2": "medium", "CD8": "high", "CD4": "high", "Viability": "high"}
        },
        {
            "id": "ood_granulocyte",
            "question": "Design a panel for granulocyte subsets",
            "markers": ["CD66b", "CD16", "CD11b", "CD15", "CD45", "Viability"],
            "expression": {"CD66b": "high", "CD16": "medium", "CD11b": "high", "CD15": "high", "CD45": "high", "Viability": "high"}
        },
    ]

    # Generate optimal assignments using spectral analysis
    for panel in novel_panels[:n]:
        # Select fluorophores that minimize overlap
        assignments = {}
        used_fluors = []

        markers_by_expression = sorted(
            panel["markers"],
            key=lambda m: {"high": 0, "medium": 1, "low": 2}.get(panel["expression"].get(m, "medium"), 1)
        )

        for marker in markers_by_expression:
            exp_level = panel["expression"].get(marker, "medium")

            # Find best fluorophore not yet used
            best_fluor = None
            best_score = float('inf')

            for fluor_name in all_fluors:
                if fluor_name in used_fluors:
                    continue

                fluor = get_fluorophore(fluor_name)
                if not fluor:
                    continue

                # Check brightness requirements
                if exp_level == "low" and fluor.relative_brightness < 60:
                    continue

                # Calculate max similarity to already selected
                max_sim = 0
                for used in used_fluors:
                    sim = get_known_overlap(fluor_name, used)
                    if sim is None:
                        used_fluor = get_fluorophore(used)
                        if used_fluor:
                            sim = calculate_spectral_overlap(fluor, used_fluor)
                        else:
                            sim = 0
                    max_sim = max(max_sim, sim)

                if max_sim < best_score:
                    best_score = max_sim
                    best_fluor = fluor_name

            if best_fluor:
                assignments[marker] = best_fluor
                used_fluors.append(best_fluor)

        # Calculate comprehensive metrics (EasyPanel methodology)
        metrics = _calculate_panel_metrics(assignments)

        test_cases.append(PanelDesignTestCase(
            id=panel["id"],
            case_type=TestCaseType.OUT_OF_DISTRIBUTION,
            biological_question=panel["question"],
            required_markers=panel["markers"],
            marker_expression=panel["expression"],
            candidate_fluorophores=all_fluors,
            ground_truth_assignments=assignments,
            ground_truth_complexity_index=metrics["total_similarity_score"],
            ground_truth_avg_similarity=metrics["avg_similarity"],
            ground_truth_n_critical_pairs=metrics["n_critical_pairs"],
            ground_truth_quality_rating=metrics["quality_rating"],
            notes="Novel marker combination not in OMIP corpus"
        ))

    # Add more OOD cases with unusual fluorophore selections
    unusual_fluor_panels = [
        {
            "id": "ood_uv_heavy",
            "question": "Design a panel emphasizing UV laser fluorophores",
            "markers": ["CD3", "CD4", "CD8", "CD19", "CD56", "Viability"],
            "expression": {"CD3": "high", "CD4": "high", "CD8": "high", "CD19": "high", "CD56": "high", "Viability": "high"},
            "preferred_fluors": ["BUV395", "BUV496", "BUV563", "BUV615", "BUV661", "BUV737", "BUV805"]
        },
        {
            "id": "ood_polymer_only",
            "question": "Design a panel using only Brilliant Violet dyes",
            "markers": ["CD3", "CD4", "CD8", "CD45RA", "CCR7", "Viability"],
            "expression": {"CD3": "high", "CD4": "high", "CD8": "high", "CD45RA": "high", "CCR7": "medium", "Viability": "high"},
            "preferred_fluors": ["BV421", "BV480", "BV510", "BV570", "BV605", "BV650", "BV711", "BV750", "BV785"]
        },
    ]

    for panel in unusual_fluor_panels:
        if len(test_cases) >= n:
            break

        # Generate assignments from preferred fluorophores
        assignments = {}
        used = []
        for marker in panel["markers"]:
            for fluor in panel["preferred_fluors"]:
                if fluor not in used and get_fluorophore(fluor):
                    assignments[marker] = fluor
                    used.append(fluor)
                    break

        # Calculate comprehensive metrics (EasyPanel methodology)
        metrics = _calculate_panel_metrics(assignments)

        test_cases.append(PanelDesignTestCase(
            id=panel["id"],
            case_type=TestCaseType.OUT_OF_DISTRIBUTION,
            biological_question=panel["question"],
            required_markers=panel["markers"],
            marker_expression=panel["expression"],
            candidate_fluorophores=all_fluors,
            ground_truth_assignments=assignments,
            ground_truth_complexity_index=metrics["total_similarity_score"],
            ground_truth_avg_similarity=metrics["avg_similarity"],
            ground_truth_n_critical_pairs=metrics["n_critical_pairs"],
            ground_truth_quality_rating=metrics["quality_rating"],
            notes="Uses fluorophore combinations not in OMIP corpus"
        ))

    return test_cases[:n]


def generate_adversarial_cases(n: int = 10) -> list[PanelDesignTestCase]:
    """
    Generate cases where OMIP precedent and spectral physics disagree.

    Types:
    - OMIP uses pair with high similarity (>0.85) - retrieval says OK, physics says bad
    - Pair has low similarity but no OMIP uses it - retrieval has no answer, physics says OK

    These are the most diagnostic cases for MCP vs retrieval.
    """
    test_cases = []

    # Find high-overlap pairs actually used in OMIPs
    high_overlap_omip_pairs = []
    for panel_name, panel_data in OMIP_PANEL_DATA.items():
        fluors = list(panel_data["assignments"].values())
        for i, f1 in enumerate(fluors):
            for f2 in fluors[i+1:]:
                sim = get_known_overlap(f1, f2)
                if sim and sim > 0.80:
                    high_overlap_omip_pairs.append({
                        "pair": (f1, f2),
                        "similarity": sim,
                        "source": panel_name
                    })

    # Adversarial case 1: OMIP uses high-overlap pair
    # Claude should recognize this is suboptimal even though OMIP uses it
    if high_overlap_omip_pairs:
        for pair_info in high_overlap_omip_pairs[:3]:
            f1, f2 = pair_info["pair"]
            source = pair_info["source"]

            # Find markers that use these fluorophores
            source_data = OMIP_PANEL_DATA[source]
            m1 = [m for m, f in source_data["assignments"].items() if f == f1]
            m2 = [m for m, f in source_data["assignments"].items() if f == f2]

            if m1 and m2:
                test_cases.append(PanelDesignTestCase(
                    id=f"adv_high_overlap_{len(test_cases)}",
                    case_type=TestCaseType.ADVERSARIAL,
                    biological_question=f"Design a panel including {m1[0]} and {m2[0]} with minimal spectral overlap",
                    required_markers=[m1[0], m2[0]] + ["CD3", "Viability"],
                    marker_expression={m1[0]: "medium", m2[0]: "medium", "CD3": "high", "Viability": "high"},
                    candidate_fluorophores=_get_available_fluorophores(),
                    ground_truth_assignments={},  # No single ground truth - test spectral awareness
                    ground_truth_complexity_index=0.0,
                    notes=f"OMIP uses {f1}/{f2} (sim={pair_info['similarity']:.2f}) but better options exist"
                ))

    # Adversarial case 2: Good pairs that OMIPs don't use
    # These are spectrally optimal but retrieval won't find them
    good_unused_panels = [
        {
            "id": "adv_unused_good",
            "question": "Design a 5-color T cell panel with UV laser optimization",
            "markers": ["CD3", "CD4", "CD8", "CD45RA", "Viability"],
            "expression": {"CD3": "high", "CD4": "high", "CD8": "high", "CD45RA": "high", "Viability": "high"},
            "optimal_assignments": {
                "CD3": "BUV496",
                "CD4": "BUV563",
                "CD8": "BUV805",
                "CD45RA": "BV570",
                "Viability": "LIVE/DEAD Blue"
            },
            "notes": "UV-optimized panel not in OMIP corpus"
        },
        {
            "id": "adv_minimal_overlap",
            "question": "Design a 4-color lineage panel with absolutely minimal spectral overlap",
            "markers": ["CD3", "CD19", "CD56", "Viability"],
            "expression": {"CD3": "high", "CD19": "high", "CD56": "high", "Viability": "high"},
            "optimal_assignments": {
                "CD3": "BUV395",
                "CD19": "FITC",
                "CD56": "APC",
                "Viability": "PE-Cy7"
            },
            "notes": "Maximally separated fluorophores"
        },
    ]

    for panel in good_unused_panels:
        # Calculate comprehensive metrics (EasyPanel methodology)
        metrics = _calculate_panel_metrics(panel["optimal_assignments"])

        test_cases.append(PanelDesignTestCase(
            id=panel["id"],
            case_type=TestCaseType.ADVERSARIAL,
            biological_question=panel["question"],
            required_markers=panel["markers"],
            marker_expression=panel["expression"],
            candidate_fluorophores=_get_available_fluorophores(),
            ground_truth_assignments=panel["optimal_assignments"],
            ground_truth_complexity_index=metrics["total_similarity_score"],
            ground_truth_avg_similarity=metrics["avg_similarity"],
            ground_truth_n_critical_pairs=metrics["n_critical_pairs"],
            ground_truth_quality_rating=metrics["quality_rating"],
            notes=panel["notes"]
        ))

    # Adversarial case 3: Panel where naive pattern matching would fail
    tricky_panels = [
        {
            "id": "adv_tandem_trap",
            "question": "Design a panel avoiding PE tandem degradation issues",
            "markers": ["CD3", "CD4", "CD8", "CD25", "CD127", "Viability"],
            "expression": {"CD3": "high", "CD4": "high", "CD8": "high", "CD25": "medium", "CD127": "medium", "Viability": "high"},
            "notes": "Should prefer non-tandem dyes or stable tandems over PE-Cy7/APC-Cy7"
        },
        {
            "id": "adv_brightness_mismatch",
            "question": "Design a panel for low-expression markers requiring bright fluorophores",
            "markers": ["CD39", "CXCR3", "CCR6", "CXCR5", "CD3", "Viability"],
            "expression": {"CD39": "low", "CXCR3": "low", "CCR6": "low", "CXCR5": "low", "CD3": "high", "Viability": "high"},
            "notes": "Low-expression markers need brightest fluorophores, not just minimal overlap"
        },
    ]

    for panel in tricky_panels:
        if len(test_cases) >= n:
            break

        # Generate optimal assignment
        assignments = {}
        used = []
        all_fluors = _get_available_fluorophores()

        for marker in panel["markers"]:
            exp = panel["expression"].get(marker, "medium")

            best_fluor = None
            best_score = -1

            for fluor_name in all_fluors:
                if fluor_name in used:
                    continue

                fluor = get_fluorophore(fluor_name)
                if not fluor:
                    continue

                # Score based on brightness and overlap
                brightness_score = fluor.relative_brightness / 100

                max_sim = 0
                for used_f in used:
                    sim = get_known_overlap(fluor_name, used_f) or 0
                    max_sim = max(max_sim, sim)

                overlap_penalty = max_sim

                # Low expression needs brightness
                if exp == "low":
                    score = brightness_score * 2 - overlap_penalty
                else:
                    score = brightness_score - overlap_penalty

                if score > best_score:
                    best_score = score
                    best_fluor = fluor_name

            if best_fluor:
                assignments[marker] = best_fluor
                used.append(best_fluor)

        # Calculate comprehensive metrics (EasyPanel methodology)
        metrics = _calculate_panel_metrics(assignments)

        test_cases.append(PanelDesignTestCase(
            id=panel["id"],
            case_type=TestCaseType.ADVERSARIAL,
            biological_question=panel["question"],
            required_markers=panel["markers"],
            marker_expression=panel["expression"],
            candidate_fluorophores=all_fluors,
            ground_truth_assignments=assignments,
            ground_truth_complexity_index=metrics["total_similarity_score"],
            ground_truth_avg_similarity=metrics["avg_similarity"],
            ground_truth_n_critical_pairs=metrics["n_critical_pairs"],
            ground_truth_quality_rating=metrics["quality_rating"],
            notes=panel["notes"]
        ))

    return test_cases[:n]


def build_ablation_test_suite(
    n_in_dist: int = 20,
    n_near_dist: int = 10,
    n_out_dist: int = 20,
    n_adversarial: int = 10
) -> TestSuite:
    """Build complete test suite for ablation study.

    Args:
        n_in_dist: Number of in-distribution cases (OMIP-derived)
        n_near_dist: Number of near-distribution cases (same markers, ~50% different fluorophores)
        n_out_dist: Number of out-of-distribution cases (novel marker combinations)
        n_adversarial: Number of adversarial cases (challenging edge cases)

    Returns:
        TestSuite with all test cases
    """
    return TestSuite(
        name="mcp_ablation_v2",
        test_cases=(
            generate_in_distribution_cases(n_in_dist) +
            generate_near_distribution_cases(n_near_dist) +
            generate_out_of_distribution_cases(n_out_dist) +
            generate_adversarial_cases(n_adversarial)
        )
    )


if __name__ == "__main__":
    # Test generation with all case types
    suite = build_ablation_test_suite(
        n_in_dist=5,
        n_near_dist=3,
        n_out_dist=5,
        n_adversarial=3
    )

    print(f"Generated {len(suite.test_cases)} test cases:")
    print(f"  In-distribution: {len(suite.filter_by_type(TestCaseType.IN_DISTRIBUTION))}")
    print(f"  Near-distribution: {len(suite.filter_by_type(TestCaseType.NEAR_DISTRIBUTION))}")
    print(f"  Out-of-distribution: {len(suite.filter_by_type(TestCaseType.OUT_OF_DISTRIBUTION))}")
    print(f"  Adversarial: {len(suite.filter_by_type(TestCaseType.ADVERSARIAL))}")

    print("\nSample test cases by type:")
    for case_type in TestCaseType:
        cases = suite.filter_by_type(case_type)
        if cases:
            tc = cases[0]
            print(f"\n{case_type.value.upper()}:")
            print(f"  ID: {tc.id}")
            print(f"  Question: {tc.biological_question}")
            print(f"  Markers: {tc.required_markers}")
            print(f"  Total Similarity: {tc.ground_truth_complexity_index:.4f}")
            print(f"  Avg Similarity: {tc.ground_truth_avg_similarity:.4f}")
            print(f"  Critical Pairs: {tc.ground_truth_n_critical_pairs}")
            print(f"  Quality: {tc.ground_truth_quality_rating}")
