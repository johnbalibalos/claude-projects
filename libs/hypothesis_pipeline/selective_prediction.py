"""
Selective prediction module for LLM evaluation.

Allows models to abstain from predictions when uncertain:
- Coverage vs accuracy tradeoffs
- Risk-coverage curves
- Optimal threshold selection
- Abstention strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike


# =============================================================================
# SELECTIVE PREDICTION METRICS
# =============================================================================


@dataclass
class SelectivePredictionResult:
    """Result of selective prediction evaluation."""

    # Core metrics
    coverage: float  # Fraction of samples model attempts
    selective_accuracy: float  # Accuracy on attempted samples
    rejection_rate: float  # Fraction of samples rejected

    # Risk metrics
    selective_risk: float  # Error rate on attempted samples (1 - accuracy)
    overall_risk: float  # Error rate including abstentions as errors

    # Threshold info
    threshold: float
    n_total: int
    n_attempted: int
    n_rejected: int
    n_correct: int
    n_incorrect: int

    def __str__(self) -> str:
        return (
            f"Selective Prediction (threshold={self.threshold:.2f}):\n"
            f"  Coverage: {self.coverage:.1%} ({self.n_attempted}/{self.n_total})\n"
            f"  Selective Accuracy: {self.selective_accuracy:.1%}\n"
            f"  Rejection Rate: {self.rejection_rate:.1%}"
        )


def selective_prediction_metrics(
    confidences: ArrayLike,
    correctness: ArrayLike,
    threshold: float = 0.5,
) -> SelectivePredictionResult:
    """
    Compute selective prediction metrics at a given confidence threshold.

    Model only makes predictions when confidence >= threshold.

    Args:
        confidences: Model confidence scores (0-1)
        correctness: Binary correctness labels
        threshold: Confidence threshold for prediction

    Returns:
        SelectivePredictionResult with coverage and accuracy metrics

    Example:
        >>> conf = [0.9, 0.7, 0.4, 0.8, 0.3]
        >>> correct = [1, 1, 0, 0, 1]
        >>> result = selective_prediction_metrics(conf, correct, threshold=0.6)
        >>> print(f"Coverage: {result.coverage:.1%}, Accuracy: {result.selective_accuracy:.1%}")
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    n_total = len(confidences)

    # Filter by threshold
    mask = confidences >= threshold
    n_attempted = np.sum(mask)
    n_rejected = n_total - n_attempted

    if n_attempted == 0:
        return SelectivePredictionResult(
            coverage=0.0,
            selective_accuracy=0.0,
            rejection_rate=1.0,
            selective_risk=1.0,
            overall_risk=1.0,
            threshold=threshold,
            n_total=n_total,
            n_attempted=0,
            n_rejected=n_rejected,
            n_correct=0,
            n_incorrect=0,
        )

    # Compute metrics on attempted samples
    attempted_correctness = correctness[mask]
    n_correct = int(np.sum(attempted_correctness))
    n_incorrect = n_attempted - n_correct

    coverage = n_attempted / n_total
    selective_accuracy = n_correct / n_attempted
    selective_risk = 1 - selective_accuracy

    # Overall risk: treat rejections as errors
    overall_risk = (n_incorrect + n_rejected) / n_total

    return SelectivePredictionResult(
        coverage=float(coverage),
        selective_accuracy=float(selective_accuracy),
        rejection_rate=float(n_rejected / n_total),
        selective_risk=float(selective_risk),
        overall_risk=float(overall_risk),
        threshold=threshold,
        n_total=n_total,
        n_attempted=int(n_attempted),
        n_rejected=int(n_rejected),
        n_correct=n_correct,
        n_incorrect=n_incorrect,
    )


# =============================================================================
# RISK-COVERAGE CURVES
# =============================================================================


@dataclass
class RiskCoverageCurve:
    """Risk-coverage curve data for selective prediction."""

    coverages: list[float]
    risks: list[float]  # Selective risk at each coverage
    thresholds: list[float]
    auc_rc: float  # Area under risk-coverage curve
    optimal_threshold: float  # Threshold that minimizes risk at 80% coverage
    optimal_coverage: float
    optimal_risk: float


def compute_risk_coverage_curve(
    confidences: ArrayLike,
    correctness: ArrayLike,
    n_thresholds: int = 100,
) -> RiskCoverageCurve:
    """
    Compute risk-coverage curve by varying confidence threshold.

    The curve shows the tradeoff between coverage (fraction of predictions made)
    and risk (error rate on predictions made).

    Args:
        confidences: Model confidence scores (0-1)
        correctness: Binary correctness labels
        n_thresholds: Number of threshold points to compute

    Returns:
        RiskCoverageCurve with curve data and optimal threshold

    Example:
        >>> conf = [0.9, 0.7, 0.4, 0.8, 0.3, 0.95, 0.6]
        >>> correct = [1, 1, 0, 0, 1, 1, 0]
        >>> curve = compute_risk_coverage_curve(conf, correct)
        >>> print(f"AUC-RC: {curve.auc_rc:.3f}")
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    # Sort by confidence (descending)
    sorted_indices = np.argsort(-confidences)
    sorted_conf = confidences[sorted_indices]
    sorted_correct = correctness[sorted_indices]

    # Compute curve at various thresholds
    thresholds = np.linspace(0, 1, n_thresholds)
    coverages = []
    risks = []
    valid_thresholds = []

    for thresh in thresholds:
        result = selective_prediction_metrics(confidences, correctness, thresh)
        if result.coverage > 0:
            coverages.append(result.coverage)
            risks.append(result.selective_risk)
            valid_thresholds.append(thresh)

    # Compute AUC (area under risk-coverage curve)
    # Lower AUC is better (less risk at each coverage level)
    if len(coverages) > 1:
        # Sort by coverage for proper integration
        sorted_pairs = sorted(zip(coverages, risks))
        sorted_cov = [p[0] for p in sorted_pairs]
        sorted_risk = [p[1] for p in sorted_pairs]
        # Use trapezoid (numpy >= 2.0) or trapz (older numpy)
        try:
            auc_rc = float(np.trapezoid(sorted_risk, sorted_cov))
        except AttributeError:
            auc_rc = float(np.trapz(sorted_risk, sorted_cov))
    else:
        auc_rc = 1.0

    # Find optimal threshold for ~80% coverage
    target_coverage = 0.8
    optimal_idx = -1
    min_coverage_diff = float('inf')

    for i, cov in enumerate(coverages):
        diff = abs(cov - target_coverage)
        if diff < min_coverage_diff:
            min_coverage_diff = diff
            optimal_idx = i

    if optimal_idx >= 0:
        optimal_threshold = valid_thresholds[optimal_idx]
        optimal_coverage = coverages[optimal_idx]
        optimal_risk = risks[optimal_idx]
    else:
        optimal_threshold = 0.5
        optimal_coverage = 1.0
        optimal_risk = 1.0 - float(np.mean(correctness))

    return RiskCoverageCurve(
        coverages=coverages,
        risks=risks,
        thresholds=valid_thresholds,
        auc_rc=auc_rc,
        optimal_threshold=optimal_threshold,
        optimal_coverage=optimal_coverage,
        optimal_risk=optimal_risk,
    )


def plot_risk_coverage_ascii(curve: RiskCoverageCurve, width: int = 60) -> str:
    """Generate ASCII plot of risk-coverage curve."""
    if not curve.coverages:
        return "No data to plot"

    lines = [
        "Risk-Coverage Curve",
        "=" * width,
        "Risk",
    ]

    height = 10
    grid = [[" " for _ in range(width - 6)] for _ in range(height)]

    # Plot curve points
    for cov, risk in zip(curve.coverages, curve.risks):
        x_idx = int(cov * (width - 7))
        y_idx = int(risk * (height - 1))
        if 0 <= x_idx < width - 6 and 0 <= y_idx < height:
            grid[height - 1 - y_idx][x_idx] = "●"

    # Mark optimal point
    x_opt = int(curve.optimal_coverage * (width - 7))
    y_opt = int(curve.optimal_risk * (height - 1))
    if 0 <= x_opt < width - 6 and 0 <= y_opt < height:
        grid[height - 1 - y_opt][x_opt] = "★"

    # Build output
    for i, row in enumerate(grid):
        y_val = 1.0 - i / (height - 1)
        if i == height // 2:
            lines.append(f"{y_val:.1f} |" + "".join(row) + "|")
        else:
            lines.append("    |" + "".join(row) + "|")

    lines.append("0.0 |" + "_" * (width - 6) + "|")
    lines.append("    0.0" + " " * (width - 14) + "1.0")
    lines.append(" " * ((width - 8) // 2) + "Coverage")
    lines.append("")
    lines.append(f"AUC-RC: {curve.auc_rc:.4f} (lower is better)")
    lines.append(f"Optimal: threshold={curve.optimal_threshold:.2f}, "
                 f"coverage={curve.optimal_coverage:.1%}, risk={curve.optimal_risk:.1%}")
    lines.append("Legend: ● (curve points)  ★ (optimal at ~80% coverage)")

    return "\n".join(lines)


# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================


@dataclass
class ThresholdOptimizationResult:
    """Result of threshold optimization."""

    optimal_threshold: float
    optimal_coverage: float
    optimal_accuracy: float
    optimization_criterion: str
    all_thresholds: list[float] = field(repr=False, default_factory=list)
    all_scores: list[float] = field(repr=False, default_factory=list)


def optimize_threshold(
    confidences: ArrayLike,
    correctness: ArrayLike,
    criterion: Literal["accuracy", "f1", "coverage_accuracy", "risk_coverage"] = "accuracy",
    min_coverage: float = 0.5,
    n_thresholds: int = 100,
) -> ThresholdOptimizationResult:
    """
    Find optimal confidence threshold for selective prediction.

    Args:
        confidences: Model confidence scores (0-1)
        correctness: Binary correctness labels
        criterion: Optimization criterion
            - "accuracy": Maximize selective accuracy
            - "f1": Maximize F1 of predictions (precision=accuracy, recall=coverage)
            - "coverage_accuracy": Maximize coverage * accuracy
            - "risk_coverage": Minimize risk at given coverage
        min_coverage: Minimum required coverage
        n_thresholds: Number of thresholds to try

    Returns:
        ThresholdOptimizationResult with optimal threshold
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    thresholds = np.linspace(0, 1, n_thresholds)
    scores = []
    valid_thresholds = []

    for thresh in thresholds:
        result = selective_prediction_metrics(confidences, correctness, thresh)

        # Skip if coverage is too low
        if result.coverage < min_coverage:
            continue

        if criterion == "accuracy":
            score = result.selective_accuracy
        elif criterion == "f1":
            # F1 with precision=accuracy, recall=coverage
            if result.selective_accuracy + result.coverage > 0:
                score = 2 * result.selective_accuracy * result.coverage / \
                        (result.selective_accuracy + result.coverage)
            else:
                score = 0
        elif criterion == "coverage_accuracy":
            score = result.coverage * result.selective_accuracy
        elif criterion == "risk_coverage":
            # Minimize risk (so negate for maximization)
            score = -result.selective_risk
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        scores.append(score)
        valid_thresholds.append(thresh)

    if not scores:
        # No threshold meets min_coverage requirement
        return ThresholdOptimizationResult(
            optimal_threshold=0.0,
            optimal_coverage=1.0,
            optimal_accuracy=float(np.mean(correctness)),
            optimization_criterion=criterion,
        )

    # Find best threshold
    best_idx = np.argmax(scores)
    best_threshold = valid_thresholds[best_idx]

    result = selective_prediction_metrics(confidences, correctness, best_threshold)

    return ThresholdOptimizationResult(
        optimal_threshold=float(best_threshold),
        optimal_coverage=result.coverage,
        optimal_accuracy=result.selective_accuracy,
        optimization_criterion=criterion,
        all_thresholds=valid_thresholds,
        all_scores=scores,
    )


# =============================================================================
# ABSTENTION STRATEGIES
# =============================================================================


class AbstentionStrategy:
    """Base class for abstention strategies."""

    def should_abstain(
        self,
        confidence: float,
        response: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Determine whether to abstain from prediction."""
        raise NotImplementedError


class ConfidenceThresholdStrategy(AbstentionStrategy):
    """Simple confidence threshold abstention."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def should_abstain(self, confidence: float, **kwargs: Any) -> bool:
        return confidence < self.threshold


class AdaptiveThresholdStrategy(AbstentionStrategy):
    """Adaptive threshold based on running accuracy."""

    def __init__(
        self,
        initial_threshold: float = 0.5,
        learning_rate: float = 0.1,
        target_accuracy: float = 0.9,
    ):
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.target_accuracy = target_accuracy
        self._history: list[tuple[float, bool]] = []

    def should_abstain(self, confidence: float, **kwargs: Any) -> bool:
        return confidence < self.threshold

    def update(self, confidence: float, was_correct: bool) -> None:
        """Update threshold based on prediction outcome."""
        self._history.append((confidence, was_correct))

        # Only update after enough history
        if len(self._history) < 10:
            return

        # Compute recent accuracy above current threshold
        recent = self._history[-50:]
        above_threshold = [(c, w) for c, w in recent if c >= self.threshold]

        if len(above_threshold) < 5:
            return

        accuracy = sum(w for _, w in above_threshold) / len(above_threshold)

        # Adjust threshold
        if accuracy < self.target_accuracy:
            # Accuracy too low, raise threshold
            self.threshold = min(0.99, self.threshold + self.learning_rate)
        else:
            # Can afford to lower threshold for more coverage
            self.threshold = max(0.01, self.threshold - self.learning_rate * 0.5)


class UncertaintyBasedStrategy(AbstentionStrategy):
    """Abstain based on uncertainty indicators in response."""

    UNCERTAINTY_PHRASES = [
        "i'm not sure",
        "i don't know",
        "uncertain",
        "unsure",
        "might be",
        "possibly",
        "could be",
        "not certain",
        "hard to say",
        "difficult to determine",
        "unclear",
        "ambiguous",
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        uncertainty_weight: float = 0.3,
    ):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_weight = uncertainty_weight

    def should_abstain(
        self,
        confidence: float,
        response: str | None = None,
        **kwargs: Any,
    ) -> bool:
        # Base confidence check
        if confidence < self.confidence_threshold:
            return True

        # Check for uncertainty phrases in response
        if response:
            response_lower = response.lower()
            uncertainty_count = sum(
                1 for phrase in self.UNCERTAINTY_PHRASES
                if phrase in response_lower
            )
            # Penalize confidence for uncertainty phrases
            adjusted_confidence = confidence - (uncertainty_count * self.uncertainty_weight)
            if adjusted_confidence < self.confidence_threshold:
                return True

        return False


class EnsembleDisagreementStrategy(AbstentionStrategy):
    """Abstain when ensemble members disagree."""

    def __init__(
        self,
        agreement_threshold: float = 0.7,
        min_confidence: float = 0.3,
    ):
        self.agreement_threshold = agreement_threshold
        self.min_confidence = min_confidence

    def should_abstain(
        self,
        confidence: float,
        ensemble_predictions: list[Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        if confidence < self.min_confidence:
            return True

        if ensemble_predictions:
            # Check agreement
            unique_predictions = set(str(p) for p in ensemble_predictions)
            agreement = 1 - (len(unique_predictions) - 1) / len(ensemble_predictions)
            if agreement < self.agreement_threshold:
                return True

        return False


# =============================================================================
# SELECTIVE PREDICTION EVALUATOR
# =============================================================================


@dataclass
class SelectiveEvaluationResult:
    """Complete selective prediction evaluation."""

    # Basic metrics
    overall_accuracy: float  # Without selection
    selective_accuracy: float  # With selection
    coverage: float

    # Risk-coverage analysis
    risk_coverage_curve: RiskCoverageCurve

    # Threshold analysis
    threshold_optimization: ThresholdOptimizationResult

    # Comparison at different thresholds
    metrics_by_threshold: dict[float, SelectivePredictionResult]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "SELECTIVE PREDICTION EVALUATION",
            "=" * 60,
            f"Overall Accuracy (no selection): {self.overall_accuracy:.1%}",
            f"Selective Accuracy (optimized): {self.selective_accuracy:.1%}",
            f"Coverage at optimal threshold: {self.coverage:.1%}",
            f"Optimal threshold: {self.threshold_optimization.optimal_threshold:.2f}",
            "",
            f"AUC-RC: {self.risk_coverage_curve.auc_rc:.4f}",
            "",
            "Metrics at different thresholds:",
            "| Threshold | Coverage | Accuracy | Risk |",
            "|-----------|----------|----------|------|",
        ]

        for thresh in [0.3, 0.5, 0.7, 0.9]:
            if thresh in self.metrics_by_threshold:
                m = self.metrics_by_threshold[thresh]
                lines.append(
                    f"| {thresh:.1f} | {m.coverage:.1%} | "
                    f"{m.selective_accuracy:.1%} | {m.selective_risk:.1%} |"
                )

        return "\n".join(lines)


def evaluate_selective_prediction(
    confidences: ArrayLike,
    correctness: ArrayLike,
    optimization_criterion: str = "coverage_accuracy",
    min_coverage: float = 0.5,
) -> SelectiveEvaluationResult:
    """
    Complete evaluation of selective prediction.

    Args:
        confidences: Model confidence scores (0-1)
        correctness: Binary correctness labels
        optimization_criterion: Criterion for threshold optimization
        min_coverage: Minimum coverage requirement

    Returns:
        SelectiveEvaluationResult with comprehensive analysis
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    # Overall accuracy
    overall_accuracy = float(np.mean(correctness))

    # Risk-coverage curve
    rc_curve = compute_risk_coverage_curve(confidences, correctness)

    # Threshold optimization
    thresh_opt = optimize_threshold(
        confidences, correctness,
        criterion=optimization_criterion,
        min_coverage=min_coverage,
    )

    # Metrics at different thresholds
    metrics_by_threshold = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics_by_threshold[thresh] = selective_prediction_metrics(
            confidences, correctness, thresh
        )

    return SelectiveEvaluationResult(
        overall_accuracy=overall_accuracy,
        selective_accuracy=thresh_opt.optimal_accuracy,
        coverage=thresh_opt.optimal_coverage,
        risk_coverage_curve=rc_curve,
        threshold_optimization=thresh_opt,
        metrics_by_threshold=metrics_by_threshold,
    )


# =============================================================================
# COST-SENSITIVE SELECTIVE PREDICTION
# =============================================================================


@dataclass
class CostSensitiveResult:
    """Result of cost-sensitive selective prediction."""

    optimal_threshold: float
    expected_cost: float
    coverage: float
    accuracy: float
    abstention_cost_paid: float
    error_cost_paid: float


def cost_sensitive_selective_prediction(
    confidences: ArrayLike,
    correctness: ArrayLike,
    error_cost: float = 1.0,
    abstention_cost: float = 0.5,
    n_thresholds: int = 100,
) -> CostSensitiveResult:
    """
    Optimize selective prediction with different costs for errors and abstentions.

    Args:
        confidences: Model confidence scores
        correctness: Binary correctness labels
        error_cost: Cost of making an incorrect prediction
        abstention_cost: Cost of abstaining
        n_thresholds: Number of thresholds to try

    Returns:
        CostSensitiveResult with cost-optimal threshold
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)
    n = len(confidences)

    best_threshold = 0.0
    min_cost = float('inf')
    best_result = None

    for thresh in np.linspace(0, 1, n_thresholds):
        mask = confidences >= thresh
        n_attempted = np.sum(mask)
        n_abstained = n - n_attempted

        if n_attempted > 0:
            # Use == 0 instead of ~ to avoid bitwise NOT on integers
            n_errors = np.sum(correctness[mask] == 0)
        else:
            n_errors = 0

        total_cost = n_errors * error_cost + n_abstained * abstention_cost

        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = thresh
            best_result = {
                "n_attempted": n_attempted,
                "n_abstained": n_abstained,
                "n_errors": n_errors,
            }

    if best_result is None:
        return CostSensitiveResult(
            optimal_threshold=0.0,
            expected_cost=n * error_cost,
            coverage=1.0,
            accuracy=float(np.mean(correctness)),
            abstention_cost_paid=0.0,
            error_cost_paid=float((1 - np.mean(correctness)) * n * error_cost),
        )

    coverage = best_result["n_attempted"] / n
    accuracy = 1 - (best_result["n_errors"] / best_result["n_attempted"]) \
        if best_result["n_attempted"] > 0 else 0.0

    return CostSensitiveResult(
        optimal_threshold=float(best_threshold),
        expected_cost=float(min_cost / n),  # Per-sample cost
        coverage=float(coverage),
        accuracy=float(accuracy),
        abstention_cost_paid=float(best_result["n_abstained"] * abstention_cost / n),
        error_cost_paid=float(best_result["n_errors"] * error_cost / n),
    )
