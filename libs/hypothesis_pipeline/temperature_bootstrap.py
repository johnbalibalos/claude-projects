"""
Temperature Bootstrap: Best-of-N Selection with LLM Judge.

Runs the same trial at multiple temperatures and uses an LLM judge
to select the best output, potentially finding better reasoning
through controlled randomness.

Key insight: Temperature introduces variance that may escape local
optima in reasoning, and the judge can identify when higher temperature
produces genuinely better outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .llm_judge import (
    EvaluationRubric,
    JudgeModel,
    JudgmentResult,
    LLMJudge,
    PairwiseComparisonResult,
    PairwiseJudge,
)
from .models import ExperimentResults, HypothesisCondition, TrialResult


@dataclass
class TemperatureCandidate:
    """A single candidate from temperature sampling."""

    temperature: float
    response: str
    trial_result: TrialResult
    scores: dict[str, float]


@dataclass
class BestOfNResult:
    """Result of best-of-N temperature selection."""

    trial_id: str
    base_condition: str  # Condition without temperature suffix
    n_candidates: int
    temperatures_tested: list[float]

    # The selected best candidate
    best_candidate: TemperatureCandidate
    best_temperature: float

    # Comparison details
    comparisons: list[PairwiseComparisonResult]
    selection_rationale: str

    # Analysis
    temperature_impact: Literal["high", "medium", "low", "none"]
    score_variance: float
    best_vs_deterministic_delta: float  # Score improvement over t=0

    # All candidates for debugging/analysis
    all_candidates: list[TemperatureCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "base_condition": self.base_condition,
            "n_candidates": self.n_candidates,
            "temperatures_tested": self.temperatures_tested,
            "best_temperature": self.best_temperature,
            "temperature_impact": self.temperature_impact,
            "score_variance": self.score_variance,
            "best_vs_deterministic_delta": self.best_vs_deterministic_delta,
            "selection_rationale": self.selection_rationale,
            "best_response": self.best_candidate.response[:500],
            "best_scores": self.best_candidate.scores,
        }


class BestOfNSelector:
    """
    Selects the best output from multiple temperature runs.

    Uses tournament-style pairwise comparison to find the best
    candidate, with position bias mitigation.
    """

    def __init__(
        self,
        judge_model: JudgeModel,
        rubric: EvaluationRubric | None = None,
        use_pairwise: bool = True,
        debias: bool = True,
    ):
        """
        Initialize selector.

        Args:
            judge_model: Model to use for judging
            rubric: Evaluation rubric (default: scientific analysis)
            use_pairwise: Use pairwise comparison (more robust but slower)
            debias: Mitigate position bias in pairwise comparison
        """
        self.judge_model = judge_model
        self.rubric = rubric or EvaluationRubric.scientific_analysis_rubric()
        self.use_pairwise = use_pairwise
        self.debias = debias

        if use_pairwise:
            self.pairwise_judge = PairwiseJudge(judge_model, debias=debias)
        else:
            self.single_judge = LLMJudge(judge_model, rubric=self.rubric)

    def select_best(
        self,
        candidates: list[TemperatureCandidate],
        question: str,
        ground_truth: str | None = None,
    ) -> BestOfNResult:
        """
        Select the best candidate from multiple temperature runs.

        Args:
            candidates: List of candidates at different temperatures
            question: The original question/prompt
            ground_truth: Optional reference answer

        Returns:
            BestOfNResult with the best candidate and analysis
        """
        if len(candidates) == 1:
            return self._single_candidate_result(candidates[0])

        # Sort by temperature for consistent ordering
        candidates = sorted(candidates, key=lambda c: c.temperature)

        if self.use_pairwise:
            best, comparisons = self._tournament_selection(
                candidates, question, ground_truth
            )
        else:
            best, comparisons = self._direct_scoring_selection(
                candidates, question, ground_truth
            )

        # Compute analysis metrics
        temperatures = [c.temperature for c in candidates]
        all_scores = [
            sum(c.scores.values()) / len(c.scores) if c.scores else 0
            for c in candidates
        ]
        score_variance = self._compute_variance(all_scores)

        # Find deterministic (t=0) candidate for comparison
        deterministic = next((c for c in candidates if c.temperature == 0.0), None)
        if deterministic and deterministic.scores:
            det_score = sum(deterministic.scores.values()) / len(deterministic.scores)
            best_score = sum(best.scores.values()) / len(best.scores) if best.scores else 0
            delta = best_score - det_score
        else:
            delta = 0.0

        # Determine temperature impact
        if score_variance > 0.2:
            impact = "high"
        elif score_variance > 0.1:
            impact = "medium"
        elif score_variance > 0.05:
            impact = "low"
        else:
            impact = "none"

        # Build rationale
        rationale = self._build_selection_rationale(
            best, candidates, comparisons, impact
        )

        # Extract base condition name (remove temperature suffix)
        base_condition = candidates[0].trial_result.condition_name
        if "_t0." in base_condition or "_t1." in base_condition:
            base_condition = "_".join(base_condition.split("_")[:-1])

        return BestOfNResult(
            trial_id=candidates[0].trial_result.trial_id,
            base_condition=base_condition,
            n_candidates=len(candidates),
            temperatures_tested=temperatures,
            best_candidate=best,
            best_temperature=best.temperature,
            comparisons=comparisons,
            selection_rationale=rationale,
            temperature_impact=impact,
            score_variance=score_variance,
            best_vs_deterministic_delta=delta,
            all_candidates=candidates,
        )

    def _tournament_selection(
        self,
        candidates: list[TemperatureCandidate],
        question: str,
        ground_truth: str | None,
    ) -> tuple[TemperatureCandidate, list[PairwiseComparisonResult]]:
        """Run tournament-style pairwise comparisons."""
        comparisons = []
        remaining = candidates.copy()

        while len(remaining) > 1:
            next_round = []

            for i in range(0, len(remaining), 2):
                if i + 1 >= len(remaining):
                    # Odd one out advances
                    next_round.append(remaining[i])
                    continue

                a, b = remaining[i], remaining[i + 1]
                result = self.pairwise_judge.compare(
                    question=question,
                    response_a=a.response,
                    response_b=b.response,
                    ground_truth=ground_truth,
                )
                comparisons.append(result)

                if result.winner == "A":
                    next_round.append(a)
                elif result.winner == "B":
                    next_round.append(b)
                else:
                    # Tie: prefer lower temperature for reproducibility
                    next_round.append(a if a.temperature <= b.temperature else b)

            remaining = next_round

        return remaining[0], comparisons

    def _direct_scoring_selection(
        self,
        candidates: list[TemperatureCandidate],
        question: str,
        ground_truth: str | None,
    ) -> tuple[TemperatureCandidate, list[PairwiseComparisonResult]]:
        """Score each candidate directly and pick highest."""
        scored = []

        for candidate in candidates:
            result = self.single_judge.evaluate(
                question=question,
                response=candidate.response,
                ground_truth=ground_truth,
            )
            scored.append((candidate, result.normalized_score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0], []

    def _compute_variance(self, scores: list[float]) -> float:
        """Compute variance of scores."""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return variance ** 0.5  # Return std dev

    def _build_selection_rationale(
        self,
        best: TemperatureCandidate,
        candidates: list[TemperatureCandidate],
        comparisons: list[PairwiseComparisonResult],
        impact: str,
    ) -> str:
        """Build human-readable selection rationale."""
        lines = [
            f"Selected temperature {best.temperature:.1f} from {len(candidates)} candidates.",
            f"Temperature impact on output quality: {impact}.",
        ]

        if comparisons:
            n_biased = sum(1 for c in comparisons if c.position_bias_detected)
            if n_biased:
                lines.append(f"Position bias detected in {n_biased}/{len(comparisons)} comparisons.")

        if best.temperature > 0:
            lines.append(
                "Higher temperature produced better reasoning, "
                "suggesting deterministic decoding may have missed better paths."
            )
        else:
            lines.append(
                "Deterministic output (t=0) was judged best, "
                "indicating stable reasoning without need for exploration."
            )

        return " ".join(lines)

    def _single_candidate_result(
        self,
        candidate: TemperatureCandidate,
    ) -> BestOfNResult:
        """Handle case with only one candidate."""
        return BestOfNResult(
            trial_id=candidate.trial_result.trial_id,
            base_condition=candidate.trial_result.condition_name,
            n_candidates=1,
            temperatures_tested=[candidate.temperature],
            best_candidate=candidate,
            best_temperature=candidate.temperature,
            comparisons=[],
            selection_rationale="Only one candidate available.",
            temperature_impact="none",
            score_variance=0.0,
            best_vs_deterministic_delta=0.0,
            all_candidates=[candidate],
        )


def group_trials_by_base_condition(
    results: ExperimentResults,
) -> dict[tuple[str, str], list[TrialResult]]:
    """
    Group trial results by (trial_id, base_condition).

    Returns mapping from (trial_id, base_condition) to list of
    trial results at different temperatures.
    """
    groups: dict[tuple[str, str], list[TrialResult]] = {}

    for trial in results.trials:
        # Extract base condition (remove temperature suffix)
        condition = trial.condition_name
        if "_t0." in condition or "_t1." in condition:
            base = "_".join(condition.split("_")[:-1])
        else:
            base = condition

        key = (trial.trial_id, base)
        if key not in groups:
            groups[key] = []
        groups[key].append(trial)

    return groups


def analyze_temperature_bootstrap(
    results: ExperimentResults,
    judge_model: JudgeModel,
    questions: dict[str, str],  # trial_id -> question text
    ground_truths: dict[str, str] | None = None,  # trial_id -> ground truth
) -> dict[str, Any]:
    """
    Analyze temperature bootstrap results with LLM judge.

    Args:
        results: Experiment results with multiple temperature runs
        judge_model: Model for judging
        questions: Mapping of trial_id to question text
        ground_truths: Optional mapping of trial_id to ground truth

    Returns:
        Analysis summary with best selections and statistics
    """
    selector = BestOfNSelector(judge_model, use_pairwise=True, debias=True)
    grouped = group_trials_by_base_condition(results)

    selections: list[BestOfNResult] = []
    ground_truths = ground_truths or {}

    for (trial_id, base_condition), trials in grouped.items():
        if len(trials) < 2:
            continue

        # Build candidates
        candidates = []
        for trial in trials:
            # Extract temperature from condition name
            temp = 0.0
            if "_t" in trial.condition_name:
                try:
                    temp_str = trial.condition_name.split("_t")[-1]
                    temp = float(temp_str)
                except ValueError:
                    pass

            candidates.append(TemperatureCandidate(
                temperature=temp,
                response=trial.raw_response or "",
                trial_result=trial,
                scores=trial.scores or {},
            ))

        question = questions.get(trial_id, "")
        ground_truth = ground_truths.get(trial_id)

        selection = selector.select_best(candidates, question, ground_truth)
        selections.append(selection)

    # Aggregate statistics
    if not selections:
        return {"error": "No multi-temperature trials found"}

    best_temps = [s.best_temperature for s in selections]
    deltas = [s.best_vs_deterministic_delta for s in selections]
    impacts = [s.temperature_impact for s in selections]

    return {
        "n_trials_analyzed": len(selections),
        "best_temperature_distribution": {
            t: best_temps.count(t) for t in sorted(set(best_temps))
        },
        "mean_improvement_over_deterministic": sum(deltas) / len(deltas),
        "temperature_impact_counts": {
            i: impacts.count(i) for i in ["high", "medium", "low", "none"]
        },
        "high_impact_rate": impacts.count("high") / len(impacts),
        "selections": [s.to_dict() for s in selections],
    }


# =============================================================================
# GATING-SPECIFIC RUBRIC
# =============================================================================


def create_gating_rubric() -> EvaluationRubric:
    """Create a rubric specifically for gating hierarchy evaluation."""
    from .llm_judge import RubricCriterion, RubricLevel

    return EvaluationRubric(
        name="Gating Hierarchy Evaluation",
        description="Evaluates the quality of predicted flow cytometry gating hierarchies.",
        criteria=[
            RubricCriterion(
                name="Structural Accuracy",
                description="Is the hierarchical structure correct (parent-child relationships)?",
                weight=0.35,
                levels=[
                    RubricLevel(3, "Perfect", "All relationships correct"),
                    RubricLevel(2, "Minor Errors", "1-2 incorrect relationships"),
                    RubricLevel(1, "Significant Errors", "Multiple structural errors"),
                    RubricLevel(0, "Incorrect", "Structure is fundamentally wrong"),
                ],
            ),
            RubricCriterion(
                name="Gate Completeness",
                description="Are all expected gates present?",
                weight=0.25,
                levels=[
                    RubricLevel(3, "Complete", "All gates present"),
                    RubricLevel(2, "Mostly Complete", "1-2 gates missing"),
                    RubricLevel(1, "Incomplete", "Several gates missing"),
                    RubricLevel(0, "Very Incomplete", "Most gates missing"),
                ],
            ),
            RubricCriterion(
                name="Biological Plausibility",
                description="Does the hierarchy follow biological logic?",
                weight=0.25,
                levels=[
                    RubricLevel(3, "Expert-level", "Follows standard immunology conventions"),
                    RubricLevel(2, "Reasonable", "Generally biologically sound"),
                    RubricLevel(1, "Questionable", "Some biologically implausible elements"),
                    RubricLevel(0, "Implausible", "Major biological errors"),
                ],
            ),
            RubricCriterion(
                name="Reasoning Quality",
                description="Is the reasoning for the hierarchy well-explained?",
                weight=0.15,
                levels=[
                    RubricLevel(3, "Clear", "Well-reasoned with clear logic"),
                    RubricLevel(2, "Adequate", "Reasoning present but brief"),
                    RubricLevel(1, "Weak", "Minimal or unclear reasoning"),
                    RubricLevel(0, "None", "No reasoning provided"),
                ],
            ),
        ],
    )
