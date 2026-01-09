"""
Evaluator for reasoning benchmark.

Evaluates model responses based on:
1. Structural correctness (gate order, prerequisites)
2. Reasoning quality (concepts mentioned, justifications)
3. Coverage score (for panel subset tests)
4. Human-in-the-loop validation hooks
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    ReasoningTestCase,
    ReasoningTestType,
    ReasoningQuality,
    ReasoningResult,
)


@dataclass
class CoverageScore:
    """Score for panel subset coverage tests."""

    # Population coverage
    populations_required: list[str] = field(default_factory=list)
    populations_covered: list[str] = field(default_factory=list)
    populations_missing: list[str] = field(default_factory=list)
    population_coverage_rate: float = 0.0

    # Marker efficiency
    markers_selected: list[str] = field(default_factory=list)
    markers_essential: list[str] = field(default_factory=list)  # Required markers
    markers_redundant: list[str] = field(default_factory=list)  # Duplicative
    marker_efficiency_score: float = 0.0  # Coverage / markers used

    # Justification quality (0-1, human validated)
    justification_score: float = 0.0
    justification_notes: str = ""

    def overall_score(self) -> float:
        """Compute weighted overall score."""
        return (
            0.5 * self.population_coverage_rate
            + 0.3 * self.marker_efficiency_score
            + 0.2 * self.justification_score
        )


@dataclass
class HumanValidation:
    """Human-in-the-loop validation data."""

    validator_id: str = ""
    validation_timestamp: str = ""

    # Overall assessment
    quality_rating: str = ""  # fail, partial, pass
    is_biologically_correct: bool = False
    demonstrates_reasoning: bool = False

    # Detailed feedback
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)

    # Scores (0-10)
    biological_accuracy_score: int = 0
    reasoning_depth_score: int = 0
    practical_utility_score: int = 0

    # Free-form notes
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validator_id": self.validator_id,
            "validation_timestamp": self.validation_timestamp,
            "quality_rating": self.quality_rating,
            "is_biologically_correct": self.is_biologically_correct,
            "demonstrates_reasoning": self.demonstrates_reasoning,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "critical_errors": self.critical_errors,
            "biological_accuracy_score": self.biological_accuracy_score,
            "reasoning_depth_score": self.reasoning_depth_score,
            "practical_utility_score": self.practical_utility_score,
            "notes": self.notes,
        }


class ReasoningEvaluator:
    """
    Evaluates reasoning benchmark responses.

    Combines automated scoring with human-in-the-loop validation.
    """

    def __init__(self):
        """Initialize evaluator."""
        self.pending_human_validations: list[dict] = []

    def evaluate(
        self,
        test_case: ReasoningTestCase,
        response: str,
        parsed_hierarchy: dict | None = None,
    ) -> ReasoningResult:
        """
        Evaluate a model response.

        Args:
            test_case: The test case
            response: Raw model response text
            parsed_hierarchy: Optionally pre-parsed hierarchy

        Returns:
            ReasoningResult with scores
        """
        result = ReasoningResult(
            test_id=test_case.test_id,
            test_type=test_case.test_type,
            quality=ReasoningQuality.FAIL,
            model_response=response,
            parsed_hierarchy=parsed_hierarchy,
        )

        # Normalize response for pattern matching
        response_lower = response.lower()

        # Check for failure indicators (pattern matching behavior)
        result.failure_indicators_found = self._find_patterns(
            response_lower, test_case.criteria.failure_indicators
        )

        # Check for required reasoning concepts
        result.reasoning_concepts_found = self._find_patterns(
            response_lower, test_case.criteria.required_reasoning_concepts
        )
        result.reasoning_concepts_missing = [
            c
            for c in test_case.criteria.required_reasoning_concepts
            if c.lower() not in response_lower
        ]

        # Check for bonus concepts
        result.bonus_concepts_found = self._find_patterns(
            response_lower, test_case.criteria.bonus_concepts
        )

        # Check gate order if hierarchy provided
        if parsed_hierarchy:
            result.gates_correct, result.order_correct = self._check_gate_structure(
                parsed_hierarchy, test_case.criteria
            )

        # Check for exclusion gates mentioned
        result.exclusions_present = self._check_exclusions(
            response_lower, test_case.criteria.exclusion_gates
        )

        # Compute scores
        result.reasoning_score = self._compute_reasoning_score(result, test_case)
        result.structure_score = self._compute_structure_score(result, test_case)

        # Determine quality level
        result.quality = self._determine_quality(result, test_case)

        return result

    def _find_patterns(self, text: str, patterns: list[str]) -> list[str]:
        """Find which patterns appear in text."""
        found = []
        for pattern in patterns:
            # Simple substring match or regex
            if pattern.lower() in text:
                found.append(pattern)
            else:
                # Try as regex
                try:
                    if re.search(pattern.lower(), text):
                        found.append(pattern)
                except re.error:
                    pass
        return found

    def _check_gate_structure(
        self,
        hierarchy: dict,
        criteria: Any,
    ) -> tuple[bool, bool]:
        """Check if gate structure matches requirements."""
        # Extract gate names from hierarchy
        gates = self._extract_gates(hierarchy)
        gate_order = list(gates.keys())

        # Check required gates present
        required = criteria.required_gates_in_order
        gates_present = all(
            any(r.lower() in g.lower() for g in gate_order) for r in required
        )

        # Check order (simplified)
        order_correct = True
        for i, req in enumerate(required[:-1]):
            req_idx = next(
                (j for j, g in enumerate(gate_order) if req.lower() in g.lower()),
                -1,
            )
            next_idx = next(
                (
                    j
                    for j, g in enumerate(gate_order)
                    if required[i + 1].lower() in g.lower()
                ),
                -1,
            )
            if req_idx >= 0 and next_idx >= 0 and req_idx > next_idx:
                order_correct = False
                break

        return gates_present, order_correct

    def _extract_gates(self, node: dict, depth: int = 0) -> dict[str, int]:
        """Extract gates with their depth."""
        gates = {}
        if "name" in node:
            gates[node["name"]] = depth
        for child in node.get("children", []):
            gates.update(self._extract_gates(child, depth + 1))
        return gates

    def _check_exclusions(self, text: str, exclusion_gates: list[str]) -> bool:
        """Check if exclusion/negative gating is mentioned."""
        exclusion_patterns = [
            r"exclude",
            r"negative",
            r"dump channel",
            r"lineage negative",
            r"lin-",
            r"gate out",
            r"remove",
        ]

        has_exclusion_language = any(
            re.search(p, text) for p in exclusion_patterns
        )

        # Check if specific markers mentioned with negative
        for gate in exclusion_gates:
            if f"{gate.lower()}-" in text or f"{gate.lower()} negative" in text:
                return True

        return has_exclusion_language

    def _compute_reasoning_score(
        self,
        result: ReasoningResult,
        test_case: ReasoningTestCase,
    ) -> float:
        """Compute reasoning quality score (0-1)."""
        required = test_case.criteria.required_reasoning_concepts
        if not required:
            return 1.0 if not result.failure_indicators_found else 0.5

        found_count = len(result.reasoning_concepts_found)
        required_count = len(required)

        base_score = found_count / required_count if required_count > 0 else 0.0

        # Bonus for extra concepts
        bonus = min(len(result.bonus_concepts_found) * 0.1, 0.2)

        # Penalty for failure indicators
        penalty = min(len(result.failure_indicators_found) * 0.2, 0.5)

        return max(0.0, min(1.0, base_score + bonus - penalty))

    def _compute_structure_score(
        self,
        result: ReasoningResult,
        test_case: ReasoningTestCase,
    ) -> float:
        """Compute structural correctness score (0-1)."""
        if result.parsed_hierarchy is None:
            return 0.5  # Can't evaluate structure without hierarchy

        score = 0.0

        if result.gates_correct:
            score += 0.5
        if result.order_correct:
            score += 0.3
        if result.exclusions_present:
            score += 0.2

        return score

    def _determine_quality(
        self,
        result: ReasoningResult,
        test_case: ReasoningTestCase,
    ) -> ReasoningQuality:
        """Determine overall quality level."""
        # Fail if failure indicators present and few reasoning concepts
        if (
            len(result.failure_indicators_found) > 0
            and result.reasoning_score < 0.3
        ):
            return ReasoningQuality.FAIL

        # Pass if high reasoning score and no major failures
        if (
            result.reasoning_score >= 0.7
            and len(result.failure_indicators_found) == 0
        ):
            return ReasoningQuality.PASS

        # Partial otherwise
        return ReasoningQuality.PARTIAL

    # =========================================================================
    # Panel Subset Evaluation (with human-in-the-loop)
    # =========================================================================

    def evaluate_panel_subset(
        self,
        test_case: ReasoningTestCase,
        response: str,
        selected_markers: list[str],
        justifications: dict[str, str],
    ) -> CoverageScore:
        """
        Evaluate a panel subset design response.

        Args:
            test_case: The panel subset test case
            response: Full response text
            selected_markers: Markers the model selected
            justifications: Marker -> justification mapping

        Returns:
            CoverageScore for human review
        """
        constraints = test_case.constraints
        required_populations = constraints.get("required_populations", [])

        score = CoverageScore(
            markers_selected=selected_markers,
            populations_required=required_populations,
        )

        # Check which populations can be identified with selected markers
        score.populations_covered, score.populations_missing = (
            self._check_population_coverage(selected_markers, required_populations)
        )

        score.population_coverage_rate = (
            len(score.populations_covered) / len(required_populations)
            if required_populations
            else 1.0
        )

        # Check marker efficiency
        essential_markers = self._get_essential_markers(required_populations)
        score.markers_essential = [m for m in selected_markers if m in essential_markers]
        score.markers_redundant = self._find_redundant_markers(selected_markers)

        target_colors = constraints.get("target_colors", len(selected_markers))
        if len(selected_markers) <= target_colors:
            score.marker_efficiency_score = score.population_coverage_rate
        else:
            # Penalty for exceeding target
            excess = len(selected_markers) - target_colors
            score.marker_efficiency_score = max(
                0, score.population_coverage_rate - (excess * 0.1)
            )

        # Justification quality requires human validation
        score.justification_notes = "Pending human validation"

        return score

    def _check_population_coverage(
        self,
        markers: list[str],
        populations: list[str],
    ) -> tuple[list[str], list[str]]:
        """Check which populations can be identified with given markers."""
        # Marker requirements for each population
        population_markers = {
            "naive B cells": ["CD19", "IgD", "CD27"],
            "memory B cells": ["CD19", "IgD", "CD27"],
            "class-switched B cells": ["CD19", "IgG", "IgA", "IgD"],
            "plasmablasts": ["CD19", "CD38", "CD27"],
            "transitional B cells": ["CD19", "CD24", "CD38"],
            "naive T cells": ["CD3", "CD45RA", "CCR7"],
            "memory T cells": ["CD3", "CD45RA", "CCR7"],
            "T helper cells": ["CD3", "CD4"],
            "cytotoxic T cells": ["CD3", "CD8"],
            "Tregs": ["CD4", "CD25", "FoxP3"],
            "NK cells": ["CD56", "CD3"],  # CD3 negative
            "monocytes": ["CD14", "CD16"],
        }

        markers_lower = {m.lower() for m in markers}
        covered = []
        missing = []

        for pop in populations:
            pop_lower = pop.lower()
            required = None

            for known_pop, req_markers in population_markers.items():
                if known_pop.lower() in pop_lower or pop_lower in known_pop.lower():
                    required = req_markers
                    break

            if required is None:
                # Unknown population, assume covered if any marker mentioned
                covered.append(pop)
            else:
                # Check if at least essential markers present
                req_lower = {m.lower() for m in required}
                if len(req_lower & markers_lower) >= len(required) - 1:
                    covered.append(pop)
                else:
                    missing.append(pop)

        return covered, missing

    def _get_essential_markers(self, populations: list[str]) -> set[str]:
        """Get markers essential for the required populations."""
        essential = {"Viability"}  # Always essential

        for pop in populations:
            pop_lower = pop.lower()
            if "b cell" in pop_lower:
                essential.update(["CD19", "CD20"])
            if "t cell" in pop_lower or "t helper" in pop_lower:
                essential.update(["CD3", "CD4", "CD8"])
            if "memory" in pop_lower:
                essential.update(["CD45RA", "CCR7", "CD27"])
            if "nk" in pop_lower:
                essential.update(["CD56", "CD3"])
            if "plasmablast" in pop_lower:
                essential.update(["CD38", "CD27"])

        return essential

    def _find_redundant_markers(self, markers: list[str]) -> list[str]:
        """Find markers that are redundant (same information)."""
        redundant = []

        # CD19 and CD20 both mark B cells
        if "CD19" in markers and "CD20" in markers:
            redundant.append("CD20 (redundant with CD19)")

        return redundant

    # =========================================================================
    # Human Validation Interface
    # =========================================================================

    def queue_for_human_validation(
        self,
        test_case: ReasoningTestCase,
        result: ReasoningResult,
        model_name: str,
    ) -> str:
        """
        Queue a result for human validation.

        Returns a validation ID.
        """
        validation_id = f"val_{test_case.test_id}_{model_name}_{len(self.pending_human_validations)}"

        self.pending_human_validations.append({
            "validation_id": validation_id,
            "test_case": test_case.to_dict(),
            "result": result.to_dict(),
            "model_name": model_name,
            "status": "pending",
        })

        return validation_id

    def apply_human_validation(
        self,
        validation_id: str,
        validation: HumanValidation,
    ) -> bool:
        """
        Apply human validation to a pending result.

        Returns True if validation was applied.
        """
        for item in self.pending_human_validations:
            if item["validation_id"] == validation_id:
                item["human_validation"] = validation.to_dict()
                item["status"] = "validated"
                return True
        return False

    def export_for_human_review(self, output_path: str) -> None:
        """Export pending validations to a file for human review."""
        import json

        with open(output_path, "w") as f:
            json.dump(self.pending_human_validations, f, indent=2)


# =============================================================================
# Evaluation Metrics Summary
# =============================================================================

@dataclass
class ReasoningBenchmarkMetrics:
    """Aggregate metrics for reasoning benchmark."""

    # By test type
    metrics_by_type: dict[str, dict] = field(default_factory=dict)

    # Overall
    total_tests: int = 0
    pass_rate: float = 0.0
    partial_rate: float = 0.0
    fail_rate: float = 0.0

    # Detailed
    avg_reasoning_score: float = 0.0
    avg_structure_score: float = 0.0

    # Pattern matching detection
    pattern_matching_detected: int = 0
    genuine_reasoning_detected: int = 0

    def compute_from_results(self, results: list[ReasoningResult]) -> None:
        """Compute metrics from results."""
        if not results:
            return

        self.total_tests = len(results)

        # Count quality levels
        pass_count = sum(1 for r in results if r.quality == ReasoningQuality.PASS)
        partial_count = sum(1 for r in results if r.quality == ReasoningQuality.PARTIAL)
        fail_count = sum(1 for r in results if r.quality == ReasoningQuality.FAIL)

        self.pass_rate = pass_count / self.total_tests
        self.partial_rate = partial_count / self.total_tests
        self.fail_rate = fail_count / self.total_tests

        # Averages
        self.avg_reasoning_score = sum(r.reasoning_score for r in results) / self.total_tests
        self.avg_structure_score = sum(r.structure_score for r in results) / self.total_tests

        # Pattern matching detection
        self.pattern_matching_detected = sum(
            1 for r in results if len(r.failure_indicators_found) > 0
        )
        self.genuine_reasoning_detected = sum(
            1 for r in results
            if len(r.failure_indicators_found) == 0 and r.reasoning_score >= 0.7
        )

        # By type
        from collections import defaultdict
        by_type = defaultdict(list)
        for r in results:
            by_type[r.test_type.value].append(r)

        for test_type, type_results in by_type.items():
            self.metrics_by_type[test_type] = {
                "n": len(type_results),
                "pass_rate": sum(1 for r in type_results if r.quality == ReasoningQuality.PASS) / len(type_results),
                "avg_reasoning_score": sum(r.reasoning_score for r in type_results) / len(type_results),
            }

    def format_summary(self) -> str:
        """Format human-readable summary."""
        lines = [
            "=" * 70,
            "REASONING BENCHMARK RESULTS",
            "=" * 70,
            "",
            f"Total tests: {self.total_tests}",
            f"Pass rate: {self.pass_rate:.1%}",
            f"Partial rate: {self.partial_rate:.1%}",
            f"Fail rate: {self.fail_rate:.1%}",
            "",
            f"Avg reasoning score: {self.avg_reasoning_score:.3f}",
            f"Avg structure score: {self.avg_structure_score:.3f}",
            "",
            f"Pattern matching detected: {self.pattern_matching_detected}/{self.total_tests}",
            f"Genuine reasoning detected: {self.genuine_reasoning_detected}/{self.total_tests}",
            "",
            "By Test Type:",
        ]

        for test_type, metrics in self.metrics_by_type.items():
            lines.append(
                f"  {test_type}: {metrics['pass_rate']:.1%} pass "
                f"(n={metrics['n']}, reasoning={metrics['avg_reasoning_score']:.2f})"
            )

        return "\n".join(lines)
