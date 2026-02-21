"""
LLM-as-Judge evaluation module.

Provides best practices for using LLMs to evaluate other LLM outputs:
- Structured rubric-based evaluation
- Position bias mitigation
- Inter-judge agreement metrics
- Multi-aspect evaluation
- Calibrated scoring

This module re-exports classes from focused submodules:
- judge_common: JSON extraction, protocols
- judge_rubric: Rubric definitions
- judge_agreement: Inter-judge agreement metrics (kappa)
- judge_calibration: Judge calibration against human scores
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# Re-export from focused modules for backward compatibility
from .judge_common import JudgeModel, extract_json_from_response
from .judge_rubric import EvaluationRubric

logger = logging.getLogger(__name__)


# JUDGE EVALUATION
# =============================================================================


@dataclass
class JudgmentScore:
    """Score for a single criterion."""

    criterion: str
    score: int
    max_score: int
    rationale: str


@dataclass
class JudgmentResult:
    """Complete judgment result."""

    question: str
    response: str
    ground_truth: str | None
    criterion_scores: list[JudgmentScore]
    total_score: float
    max_total_score: float
    normalized_score: float  # 0-1
    overall_rationale: str
    judge_model: str
    raw_judgment: str = field(repr=False, default="")

    def summary(self) -> str:
        lines = [
            f"Total Score: {self.total_score:.1f}/{self.max_total_score:.1f} ({self.normalized_score:.1%})",
            "",
            "Criterion Scores:",
        ]
        for cs in self.criterion_scores:
            lines.append(f"  - {cs.criterion}: {cs.score}/{cs.max_score}")
            lines.append(f"    {cs.rationale[:100]}...")

        lines.append("")
        lines.append(f"Overall: {self.overall_rationale[:200]}...")

        return "\n".join(lines)


class LLMJudge:
    """
    LLM-based evaluation with best practices.

    Features:
    - Structured rubric-based evaluation
    - Position bias mitigation
    - Consistent scoring
    """

    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of a response.

{rubric}

## Task

Evaluate the following response to a question.

**Question:** {question}

{ground_truth_section}

**Response to Evaluate:**
{response}

## Instructions

1. Evaluate the response against each criterion in the rubric
2. Provide a score for each criterion with a brief rationale
3. Be objective and consistent in your scoring
4. Consider the ground truth (if provided) as the reference answer

## Output Format

Provide your evaluation as JSON:
```json
{{
  "criterion_scores": [
    {{
      "criterion": "criterion_name",
      "score": <int>,
      "rationale": "brief explanation"
    }}
  ],
  "overall_rationale": "summary of overall assessment",
  "total_score": <float>
}}
```

Evaluate now:"""

    def __init__(
        self,
        judge_model: JudgeModel,
        rubric: EvaluationRubric | None = None,
        model_name: str = "unknown",
    ):
        """
        Initialize LLM judge.

        Args:
            judge_model: Model client for judging
            rubric: Evaluation rubric (default: Q&A rubric)
            model_name: Name of judge model for logging
        """
        self.model = judge_model
        self.rubric = rubric or EvaluationRubric.default_qa_rubric()
        self.model_name = model_name

    def evaluate(
        self,
        question: str,
        response: str,
        ground_truth: str | None = None,
    ) -> JudgmentResult:
        """
        Evaluate a response using the rubric.

        Args:
            question: The original question
            response: The response to evaluate
            ground_truth: Optional reference answer

        Returns:
            JudgmentResult with scores and rationale
        """
        # Build prompt
        ground_truth_section = ""
        if ground_truth:
            ground_truth_section = f"**Reference Answer (Ground Truth):**\n{ground_truth}\n"

        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            rubric=self.rubric.to_prompt_string(),
            question=question,
            ground_truth_section=ground_truth_section,
            response=response,
        )

        # Get judgment
        raw_judgment = self.model.generate(prompt, max_tokens=2048)

        # Parse judgment
        return self._parse_judgment(
            raw_judgment, question, response, ground_truth
        )

    def _parse_judgment(
        self,
        raw_judgment: str,
        question: str,
        response: str,
        ground_truth: str | None,
    ) -> JudgmentResult:
        """Parse the judge's raw output into structured result."""
        # Extract JSON using the shared utility
        extraction = extract_json_from_response(raw_judgment)
        data = extraction.data

        # Track parsing failures for diagnostics
        parse_error = extraction.error if not extraction.success else None
        if parse_error:
            logger.warning(f"Judge response parsing failed: {parse_error}")

        # Build criterion scores
        criterion_scores = []
        total_score = 0.0
        max_total = 0.0

        for cs_data in data.get("criterion_scores", []):
            criterion_name = cs_data.get("criterion", "Unknown")
            score = cs_data.get("score", 0)

            # Find max score for this criterion
            max_score = 3  # Default
            for crit in self.rubric.criteria:
                if crit.name.lower() == criterion_name.lower():
                    max_score = max(level.score for level in crit.levels)
                    weight = crit.weight
                    break
            else:
                weight = 1.0 / len(self.rubric.criteria)

            criterion_scores.append(JudgmentScore(
                criterion=criterion_name,
                score=score,
                max_score=max_score,
                rationale=cs_data.get("rationale", "No rationale provided"),
            ))

            total_score += score * weight
            max_total += max_score * weight

        # Handle case where no scores were parsed
        if not criterion_scores:
            max_total = sum(
                max(level.score for level in c.levels) * c.weight
                for c in self.rubric.criteria
            )

        normalized_score = total_score / max_total if max_total > 0 else 0.0

        return JudgmentResult(
            question=question,
            response=response,
            ground_truth=ground_truth,
            criterion_scores=criterion_scores,
            total_score=total_score,
            max_total_score=max_total,
            normalized_score=normalized_score,
            overall_rationale=data.get("overall_rationale", "No overall rationale provided"),
            judge_model=self.model_name,
            raw_judgment=raw_judgment,
        )


# =============================================================================
# POSITION BIAS MITIGATION
# =============================================================================


@dataclass
class PairwiseComparisonResult:
    """Result of pairwise comparison with position debiasing."""

    response_a: str
    response_b: str
    winner: Literal["A", "B", "tie"]
    confidence: float
    score_a: float  # Debiased score for A
    score_b: float  # Debiased score for B
    position_bias_detected: bool
    raw_judgments: dict[str, str] = field(repr=False, default_factory=dict)


class PairwiseJudge:
    """
    Pairwise comparison judge with position bias mitigation.

    Runs comparisons in both orders and averages to reduce position bias.
    """

    COMPARISON_PROMPT = """You are comparing two responses to determine which is better.

**Question:** {question}

{ground_truth_section}

**Response {label_a}:**
{response_a}

**Response {label_b}:**
{response_b}

## Instructions

Compare the two responses and determine which is better based on:
- Accuracy and correctness
- Completeness
- Clarity and organization
- Relevance to the question

## Output Format

Provide your judgment as JSON:
```json
{{
  "winner": "A" or "B" or "tie",
  "score_a": <float 0-10>,
  "score_b": <float 0-10>,
  "rationale": "explanation of your decision"
}}
```

Your judgment:"""

    def __init__(
        self,
        judge_model: JudgeModel,
        debias: bool = True,
    ):
        """
        Initialize pairwise judge.

        Args:
            judge_model: Model client for judging
            debias: Whether to run both orderings for debiasing
        """
        self.model = judge_model
        self.debias = debias

    def compare(
        self,
        question: str,
        response_a: str,
        response_b: str,
        ground_truth: str | None = None,
    ) -> PairwiseComparisonResult:
        """
        Compare two responses with optional position debiasing.

        Args:
            question: The original question
            response_a: First response
            response_b: Second response
            ground_truth: Optional reference answer

        Returns:
            PairwiseComparisonResult with debiased scores
        """
        ground_truth_section = ""
        if ground_truth:
            ground_truth_section = f"**Reference Answer:**\n{ground_truth}\n"

        # First comparison: A then B
        prompt_ab = self.COMPARISON_PROMPT.format(
            question=question,
            ground_truth_section=ground_truth_section,
            label_a="A",
            response_a=response_a,
            label_b="B",
            response_b=response_b,
        )
        judgment_ab = self.model.generate(prompt_ab, max_tokens=1024)
        scores_ab = self._parse_comparison(judgment_ab)

        if not self.debias:
            return PairwiseComparisonResult(
                response_a=response_a,
                response_b=response_b,
                winner=self._determine_winner(scores_ab["score_a"], scores_ab["score_b"]),
                confidence=abs(scores_ab["score_a"] - scores_ab["score_b"]) / 10,
                score_a=scores_ab["score_a"],
                score_b=scores_ab["score_b"],
                position_bias_detected=False,
                raw_judgments={"ab": judgment_ab},
            )

        # Second comparison: B then A (swapped positions)
        prompt_ba = self.COMPARISON_PROMPT.format(
            question=question,
            ground_truth_section=ground_truth_section,
            label_a="A",
            response_a=response_b,  # B is now first
            label_b="B",
            response_b=response_a,  # A is now second
        )
        judgment_ba = self.model.generate(prompt_ba, max_tokens=1024)
        scores_ba = self._parse_comparison(judgment_ba)

        # Debias by averaging
        # In AB ordering: score_a is for response_a, score_b is for response_b
        # In BA ordering: score_a is for response_b, score_b is for response_a
        debiased_a = (scores_ab["score_a"] + scores_ba["score_b"]) / 2
        debiased_b = (scores_ab["score_b"] + scores_ba["score_a"]) / 2

        # Detect position bias
        # If scores flip based on position, there's position bias
        ab_winner = self._determine_winner(scores_ab["score_a"], scores_ab["score_b"])
        ba_winner = self._determine_winner(scores_ba["score_b"], scores_ba["score_a"])  # Note: swapped
        position_bias = ab_winner != ba_winner

        return PairwiseComparisonResult(
            response_a=response_a,
            response_b=response_b,
            winner=self._determine_winner(debiased_a, debiased_b),
            confidence=abs(debiased_a - debiased_b) / 10,
            score_a=debiased_a,
            score_b=debiased_b,
            position_bias_detected=position_bias,
            raw_judgments={"ab": judgment_ab, "ba": judgment_ba},
        )

    def _parse_comparison(self, judgment: str) -> dict[str, float]:
        """Parse comparison judgment."""
        extraction = extract_json_from_response(judgment)

        if not extraction.success:
            logger.warning(f"Comparison parse failed: {extraction.error}, using defaults")
            return {"score_a": 5.0, "score_b": 5.0}

        try:
            return {
                "score_a": float(extraction.data.get("score_a", 5)),
                "score_b": float(extraction.data.get("score_b", 5)),
            }
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid score values in comparison: {e}")
            return {"score_a": 5.0, "score_b": 5.0}

    def _determine_winner(self, score_a: float, score_b: float) -> Literal["A", "B", "tie"]:
        """Determine winner from scores."""
        diff = score_a - score_b
        if abs(diff) < 0.5:  # Tie threshold
            return "tie"
        return "A" if diff > 0 else "B"


# =============================================================================
# MULTI-JUDGE ENSEMBLE
# =============================================================================


@dataclass
class EnsembleJudgmentResult:
    """Result from ensemble of judges."""

    individual_results: list[JudgmentResult]
    ensemble_score: float
    score_std: float
    agreement_level: str  # "high", "medium", "low"
    consensus_rationale: str


class EnsembleJudge:
    """
    Ensemble of multiple judges for more reliable evaluation.

    Combines judgments from multiple judges (models or prompts)
    to produce a more robust evaluation.
    """

    def __init__(
        self,
        judges: list[LLMJudge],
        aggregation: Literal["mean", "median", "majority"] = "mean",
    ):
        """
        Initialize ensemble judge.

        Args:
            judges: List of LLMJudge instances
            aggregation: How to combine scores
        """
        self.judges = judges
        self.aggregation = aggregation

    def evaluate(
        self,
        question: str,
        response: str,
        ground_truth: str | None = None,
    ) -> EnsembleJudgmentResult:
        """
        Evaluate using all judges and aggregate.

        Args:
            question: The original question
            response: Response to evaluate
            ground_truth: Optional reference answer

        Returns:
            EnsembleJudgmentResult with aggregated score
        """
        # Collect individual judgments
        results = []
        for judge in self.judges:
            result = judge.evaluate(question, response, ground_truth)
            results.append(result)

        # Aggregate scores
        scores = [r.normalized_score for r in results]

        if self.aggregation == "mean":
            ensemble_score = float(np.mean(scores))
        elif self.aggregation == "median":
            ensemble_score = float(np.median(scores))
        else:  # majority
            # Round to nearest 0.1 and take mode
            rounded = [round(s, 1) for s in scores]
            from collections import Counter
            most_common = Counter(rounded).most_common(1)[0][0]
            ensemble_score = most_common

        score_std = float(np.std(scores))

        # Determine agreement level
        if score_std < 0.1:
            agreement = "high"
        elif score_std < 0.2:
            agreement = "medium"
        else:
            agreement = "low"

        # Combine rationales
        rationales = [r.overall_rationale for r in results]
        consensus = f"Ensemble of {len(self.judges)} judges. " + " | ".join(rationales[:3])

        return EnsembleJudgmentResult(
            individual_results=results,
            ensemble_score=ensemble_score,
            score_std=score_std,
            agreement_level=agreement,
            consensus_rationale=consensus[:500],
        )


# =============================================================================
# BIAS-AWARE JUDGE
# =============================================================================


@dataclass
class BiasAwareJudgeConfig:
    """Configuration for bias-aware evaluation.

    Based on recommendations from "Justice or Prejudice? Quantifying Biases
    in LLM-as-a-Judge" (https://arxiv.org/abs/2410.02736).
    """

    # Position bias
    debias_position: bool = True

    # Verbosity bias
    debias_verbosity: bool = True
    verbosity_penalty_factor: float = 0.1
    verbosity_max_penalty: float = 0.2

    # Self-enhancement bias
    check_self_enhancement: bool = True

    # Authority bias
    strip_authority_markers: bool = True

    # Bandwagon bias
    strip_bandwagon_markers: bool = True

    # Multi-judge ensemble (mitigates multiple biases)
    use_multi_judge: bool = False

    # Thresholds for warnings
    verbosity_ratio_warning: float = 1.5
    verbosity_ratio_severe: float = 2.5
    judge_disagreement_threshold: float = 0.15


@dataclass
class BiasAwareJudgmentResult:
    """Result from bias-aware evaluation."""

    # Core results
    score: float
    normalized_score: float
    individual_scores: list[float]

    # Bias analysis
    bias_report: Any  # BiasReport from bias_detection module
    score_adjustments: dict[str, float]

    # Raw data
    individual_results: list[JudgmentResult]
    prepared_response: str
    original_response: str

    # Metadata
    config_used: BiasAwareJudgeConfig
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Score: {self.normalized_score:.2%}",
            f"Individual scores: {[f'{s:.2%}' for s in self.individual_scores]}",
        ]

        if self.score_adjustments:
            lines.append("Score adjustments:")
            for name, adj in self.score_adjustments.items():
                lines.append(f"  - {name}: {adj:+.2%}")

        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)


class BiasAwareJudge:
    """
    LLM Judge with comprehensive bias mitigation.

    Implements bias detection and mitigation based on the CALM framework from:
    "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge"
    https://arxiv.org/abs/2410.02736

    Mitigated biases:
    - Position bias: via pairwise comparison in both orders
    - Verbosity bias: via length penalty/normalization
    - Self-enhancement bias: via multi-model ensemble and warnings
    - Authority bias: via stripping authority markers
    - Bandwagon bias: via stripping popularity claims
    - Sentiment bias: via detection and reporting

    Usage:
        from hypothesis_pipeline.llm_judge import BiasAwareJudge, BiasAwareJudgeConfig

        config = BiasAwareJudgeConfig(
            debias_position=True,
            debias_verbosity=True,
            strip_authority_markers=True,
        )

        judge = BiasAwareJudge(
            judges=[judge1, judge2],  # Multiple judges for ensemble
            config=config,
        )

        result = judge.evaluate(
            question="What is X?",
            response="Response text...",
            ground_truth="Reference answer...",
            response_source_model="gpt-4",  # For self-enhancement detection
        )

        print(result.normalized_score)
        print(result.bias_report.summary())
    """

    def __init__(
        self,
        judges: list[LLMJudge],
        config: BiasAwareJudgeConfig | None = None,
    ):
        """
        Initialize bias-aware judge.

        Args:
            judges: List of LLMJudge instances (multiple for ensemble)
            config: Configuration for bias mitigation
        """
        if not judges:
            raise ValueError("At least one judge is required")

        self.judges = judges
        self.config = config or BiasAwareJudgeConfig()

        # Create pairwise judge from first model for position debiasing
        self._pairwise_judge = PairwiseJudge(
            judges[0].model,
            debias=self.config.debias_position,
        )

    def evaluate(
        self,
        question: str,
        response: str,
        ground_truth: str | None = None,
        response_source_model: str | None = None,
    ) -> BiasAwareJudgmentResult:
        """
        Evaluate a response with comprehensive bias mitigation.

        Args:
            question: The original question
            response: The response to evaluate
            ground_truth: Optional reference answer
            response_source_model: Model that generated the response (for
                                   self-enhancement bias detection)

        Returns:
            BiasAwareJudgmentResult with scores, bias analysis, and adjustments
        """
        # Import here to avoid circular imports
        from hypothesis_pipeline.bias_detection import (
            analyze_response_for_biases,
            apply_verbosity_penalty,
            compute_verbosity_metrics,
            prepare_response_for_evaluation,
        )

        warnings = []
        score_adjustments = {}

        # 1. Analyze response for biases
        bias_report = analyze_response_for_biases(
            response=response,
            reference=ground_truth,
            judge_model=self.judges[0].model_name,
            response_source_model=response_source_model,
        )

        # Collect warnings from bias report
        warnings.extend(bias_report.bias_warnings)

        # 2. Prepare response (strip biasing elements)
        prepared_response, modifications = prepare_response_for_evaluation(
            response=response,
            strip_authority=self.config.strip_authority_markers,
            strip_bandwagon=self.config.strip_bandwagon_markers,
            normalize_formatting=False,  # Don't normalize by default
        )
        bias_report.authority_markers_stripped = modifications["authority_markers_removed"] > 0

        # 3. Run evaluation with all judges
        individual_results = []
        for judge in self.judges:
            result = judge.evaluate(
                question=question,
                response=prepared_response,
                ground_truth=ground_truth,
            )
            individual_results.append(result)

        # 4. Aggregate scores
        raw_scores = [r.normalized_score for r in individual_results]

        if len(raw_scores) > 1:
            ensemble_score = float(np.mean(raw_scores))
            score_std = float(np.std(raw_scores))

            # Check for judge disagreement
            if score_std > self.config.judge_disagreement_threshold:
                warnings.append(
                    f"High judge disagreement (std={score_std:.2f})"
                )
        else:
            ensemble_score = raw_scores[0]
            score_std = 0.0

        # 5. Apply verbosity penalty if configured
        final_score = ensemble_score
        if self.config.debias_verbosity and ground_truth:
            verbosity_metrics = compute_verbosity_metrics(response, ground_truth)

            if verbosity_metrics.get("word_ratio", 1.0) > self.config.verbosity_ratio_warning:
                adjusted_score, penalty = apply_verbosity_penalty(
                    score=final_score,
                    verbosity_metrics=verbosity_metrics,
                    penalty_factor=self.config.verbosity_penalty_factor,
                    max_penalty=self.config.verbosity_max_penalty,
                )

                if penalty > 0:
                    score_adjustments["verbosity_penalty"] = -penalty
                    final_score = adjusted_score

        return BiasAwareJudgmentResult(
            score=final_score,
            normalized_score=final_score,
            individual_scores=raw_scores,
            bias_report=bias_report,
            score_adjustments=score_adjustments,
            individual_results=individual_results,
            prepared_response=prepared_response,
            original_response=response,
            config_used=self.config,
            warnings=warnings,
        )

    def compare_pairwise(
        self,
        question: str,
        response_a: str,
        response_b: str,
        ground_truth: str | None = None,
        source_model_a: str | None = None,
        source_model_b: str | None = None,
    ) -> PairwiseComparisonResult:
        """
        Compare two responses with position bias mitigation.

        Args:
            question: The original question
            response_a: First response
            response_b: Second response
            ground_truth: Optional reference answer
            source_model_a: Model that generated response A
            source_model_b: Model that generated response B

        Returns:
            PairwiseComparisonResult with debiased scores
        """
        from hypothesis_pipeline.bias_detection import prepare_response_for_evaluation

        # Prepare both responses
        prep_a, _ = prepare_response_for_evaluation(
            response_a,
            strip_authority=self.config.strip_authority_markers,
            strip_bandwagon=self.config.strip_bandwagon_markers,
        )
        prep_b, _ = prepare_response_for_evaluation(
            response_b,
            strip_authority=self.config.strip_authority_markers,
            strip_bandwagon=self.config.strip_bandwagon_markers,
        )

        # Run position-debiased comparison
        return self._pairwise_judge.compare(
            question=question,
            response_a=prep_a,
            response_b=prep_b,
            ground_truth=ground_truth,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_default_judge(
    model_client: JudgeModel,
    rubric_type: Literal["qa", "scientific"] = "qa",
) -> LLMJudge:
    """Create a judge with default rubric."""
    if rubric_type == "scientific":
        rubric = EvaluationRubric.scientific_analysis_rubric()
    else:
        rubric = EvaluationRubric.default_qa_rubric()

    return LLMJudge(model_client, rubric)


def quick_evaluate(
    model_client: JudgeModel,
    question: str,
    response: str,
    ground_truth: str | None = None,
) -> float:
    """Quick evaluation returning just a normalized score."""
    judge = create_default_judge(model_client)
    result = judge.evaluate(question, response, ground_truth)
    return result.normalized_score


# =============================================================================
# PLUGGABLE JUDGE (For domain-specific prompts)
# =============================================================================


@dataclass
class PluggableJudgeConfig:
    """Configuration for pluggable judge."""

    max_tokens: int = 2048
    temperature: float = 0.0
    parallel_workers: int = 4
    delay_seconds: float = 0.0


@dataclass
class PluggableJudgeResult:
    """Generic result from pluggable judge.

    The parsed_data dict contains whatever the response_parser returns.
    Domain-specific code can interpret this as needed.
    """

    item_id: str
    parsed_data: dict[str, Any]
    raw_prompt: str
    raw_response: str
    success: bool
    error: str | None = None


class PluggableJudge:
    """
    LLM judge with pluggable prompt building and response parsing.

    Separates the domain-specific logic (prompts, parsing) from the
    infrastructure (parallel execution, retry, rate limiting).

    Usage:
        from hypothesis_pipeline.llm_judge import PluggableJudge, PluggableJudgeConfig

        # Define domain-specific prompt builder
        def my_prompt_builder(item: MyItem, **kwargs) -> str:
            return f"Evaluate: {item.text}"

        # Define domain-specific response parser
        def my_response_parser(response: str) -> dict | None:
            # Parse response, return dict or None on failure
            return {"score": parse_score(response)}

        judge = PluggableJudge(
            model=my_model_client,
            prompt_builder=my_prompt_builder,
            response_parser=my_response_parser,
        )

        results = judge.evaluate_batch(items)
    """

    def __init__(
        self,
        model: JudgeModel,
        prompt_builder: Callable[..., str],
        response_parser: Callable[[str], dict[str, Any] | None],
        config: PluggableJudgeConfig | None = None,
        model_name: str = "unknown",
    ):
        """
        Initialize pluggable judge.

        Args:
            model: Model client implementing JudgeModel protocol
            prompt_builder: Callable that builds prompts from items
            response_parser: Callable that parses model responses to dicts
            config: Optional configuration
            model_name: Name of model for logging
        """
        self.model = model
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.config = config or PluggableJudgeConfig()
        self.model_name = model_name

    def evaluate_one(
        self,
        item: Any,
        item_id: str,
        **prompt_kwargs: Any,
    ) -> PluggableJudgeResult:
        """
        Evaluate a single item.

        Args:
            item: The item to evaluate (passed to prompt_builder)
            item_id: Unique identifier for this item
            **prompt_kwargs: Additional kwargs passed to prompt_builder

        Returns:
            PluggableJudgeResult with parsed data or error
        """
        try:
            # Build prompt
            prompt = self.prompt_builder(item, **prompt_kwargs)

            # Call model
            raw_response = self.model.generate(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            # Parse response
            parsed = self.response_parser(raw_response)

            if parsed is not None:
                return PluggableJudgeResult(
                    item_id=item_id,
                    parsed_data=parsed,
                    raw_prompt=prompt,
                    raw_response=raw_response,
                    success=True,
                )
            else:
                return PluggableJudgeResult(
                    item_id=item_id,
                    parsed_data={},
                    raw_prompt=prompt,
                    raw_response=raw_response,
                    success=False,
                    error="Failed to parse response",
                )

        except (ValueError, TypeError) as e:
            # Prompt builder or response parser errors
            logger.warning(f"Evaluation failed for {item_id}: {type(e).__name__}: {e}")
            return PluggableJudgeResult(
                item_id=item_id,
                parsed_data={},
                raw_prompt=prompt if 'prompt' in locals() else "",
                raw_response="",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )
        except (ConnectionError, TimeoutError, OSError) as e:
            # Network/API errors
            logger.error(f"API error for {item_id}: {type(e).__name__}: {e}")
            return PluggableJudgeResult(
                item_id=item_id,
                parsed_data={},
                raw_prompt=prompt if 'prompt' in locals() else "",
                raw_response="",
                success=False,
                error=f"API error: {type(e).__name__}: {e}",
            )

    def evaluate_batch(
        self,
        items: Sequence[tuple[Any, str]],
        progress_callback: Callable[[int, int, PluggableJudgeResult], None] | None = None,
        **prompt_kwargs: Any,
    ) -> list[PluggableJudgeResult]:
        """
        Evaluate a batch of items in parallel.

        Args:
            items: Sequence of (item, item_id) tuples
            progress_callback: Optional callback(current, total, result)
            **prompt_kwargs: Additional kwargs passed to prompt_builder

        Returns:
            List of PluggableJudgeResult objects
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        total = len(items)

        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all tasks
            futures = {}
            for item, item_id in items:
                future = executor.submit(
                    self._evaluate_with_delay,
                    item,
                    item_id,
                    prompt_kwargs,
                )
                futures[future] = item_id

            # Collect results
            for i, future in enumerate(as_completed(futures), 1):
                item_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    # Log and continue - we want to collect all results even if some fail
                    # This is a legitimate use of broad exception handling for thread results
                    logger.error(f"Worker failed for {item_id}: {type(e).__name__}: {e}")
                    result = PluggableJudgeResult(
                        item_id=item_id,
                        parsed_data={},
                        raw_prompt="",
                        raw_response="",
                        success=False,
                        error=f"{type(e).__name__}: {e}",
                    )

                results.append(result)

                if progress_callback:
                    progress_callback(i, total, result)

        return results

    def _evaluate_with_delay(
        self,
        item: Any,
        item_id: str,
        prompt_kwargs: dict[str, Any],
    ) -> PluggableJudgeResult:
        """Evaluate with optional rate limit delay."""
        import time

        if self.config.delay_seconds > 0:
            time.sleep(self.config.delay_seconds)

        return self.evaluate_one(item, item_id, **prompt_kwargs)
