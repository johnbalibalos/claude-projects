"""
Calibration metrics for LLM evaluation.

Measures whether model confidence scores align with actual accuracy:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagrams
- Confidence parsing from model outputs
- Selective prediction metrics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike


# =============================================================================
# CONFIDENCE PARSING
# =============================================================================


@dataclass
class ParsedConfidence:
    """Parsed confidence information from model output."""

    confidence_score: float | None  # 0-1 normalized confidence
    raw_confidence: str | None  # Original text
    uncertainty_notes: str | None  # Any stated uncertainty
    parse_method: str  # How confidence was extracted


def parse_confidence_from_response(
    response: str,
    methods: Sequence[str] | None = None,
) -> ParsedConfidence:
    """
    Parse confidence score from model response.

    Tries multiple methods to extract confidence:
    1. JSON field extraction
    2. Explicit confidence statement
    3. Percentage mention
    4. Verbal confidence indicators

    Args:
        response: Raw model response text
        methods: Specific methods to try (default: all)

    Returns:
        ParsedConfidence with extracted score and metadata
    """
    if methods is None:
        methods = ["json", "explicit", "percentage", "verbal"]

    for method in methods:
        if method == "json":
            result = _parse_json_confidence(response)
        elif method == "explicit":
            result = _parse_explicit_confidence(response)
        elif method == "percentage":
            result = _parse_percentage_confidence(response)
        elif method == "verbal":
            result = _parse_verbal_confidence(response)
        else:
            continue

        if result.confidence_score is not None:
            return result

    return ParsedConfidence(
        confidence_score=None,
        raw_confidence=None,
        uncertainty_notes=None,
        parse_method="none",
    )


def _parse_json_confidence(response: str) -> ParsedConfidence:
    """Parse confidence from JSON-formatted response."""
    import json

    # Try to find JSON block
    json_patterns = [
        r'```json\s*([\s\S]*?)```',
        r'\{[\s\S]*?"confidence"[\s\S]*?\}',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                if not match.strip().startswith("{"):
                    match = "{" + match + "}"
                data = json.loads(match)

                confidence = data.get("confidence")
                if confidence is not None:
                    # Normalize to 0-1
                    if isinstance(confidence, (int, float)):
                        if confidence > 1:
                            confidence = confidence / 100
                        confidence = max(0, min(1, confidence))

                    return ParsedConfidence(
                        confidence_score=float(confidence),
                        raw_confidence=str(data.get("confidence")),
                        uncertainty_notes=data.get("uncertainty_notes") or data.get("uncertainty"),
                        parse_method="json",
                    )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    return ParsedConfidence(None, None, None, "json")


def _parse_explicit_confidence(response: str) -> ParsedConfidence:
    """Parse explicit confidence statements."""
    patterns = [
        r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%?',
        r'(\d+(?:\.\d+)?)\s*%?\s*confident',
        r'certainty[:\s]+(\d+(?:\.\d+)?)\s*%?',
        r'(\d+(?:\.\d+)?)\s*%?\s*certain',
        r'confidence\s*(?:score|level)?[:\s]*(\d+(?:\.\d+)?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            value = float(match.group(1))
            # Normalize to 0-1
            if value > 1:
                value = value / 100
            value = max(0, min(1, value))

            return ParsedConfidence(
                confidence_score=value,
                raw_confidence=match.group(0),
                uncertainty_notes=None,
                parse_method="explicit",
            )

    return ParsedConfidence(None, None, None, "explicit")


def _parse_percentage_confidence(response: str) -> ParsedConfidence:
    """Parse percentage mentions that might indicate confidence."""
    # Look for percentages in confidence-related context
    patterns = [
        r'(\d+(?:\.\d+)?)\s*%\s*(?:sure|certain|confident|probability|likely)',
        r'(?:sure|certain|confident|probability|likely)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*%',
    ]

    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            value = float(match.group(1)) / 100
            value = max(0, min(1, value))

            return ParsedConfidence(
                confidence_score=value,
                raw_confidence=match.group(0),
                uncertainty_notes=None,
                parse_method="percentage",
            )

    return ParsedConfidence(None, None, None, "percentage")


def _parse_verbal_confidence(response: str) -> ParsedConfidence:
    """Parse verbal confidence indicators."""
    response_lower = response.lower()

    # Map verbal expressions to confidence scores
    confidence_map = {
        # High confidence
        ("definitely", "certainly", "absolutely", "clearly", "obviously"): 0.95,
        ("very confident", "highly confident", "very sure", "highly certain"): 0.90,
        ("confident", "sure", "certain"): 0.85,
        # Medium confidence
        ("probably", "likely", "most likely"): 0.75,
        ("think", "believe", "seems"): 0.65,
        ("fairly confident", "reasonably sure"): 0.70,
        # Low confidence
        ("possibly", "might", "could be"): 0.50,
        ("uncertain", "not sure", "unsure"): 0.40,
        ("guess", "speculation"): 0.30,
        # Very low confidence
        ("unlikely", "doubt", "doubtful"): 0.25,
        ("very uncertain", "highly uncertain"): 0.20,
    }

    # Check for negations that flip confidence
    negation_words = ["not", "don't", "doesn't", "isn't", "aren't", "no"]
    has_negation = any(neg in response_lower for neg in negation_words)

    for phrases, base_score in confidence_map.items():
        for phrase in phrases:
            if phrase in response_lower:
                # Check if negated
                score = base_score
                if has_negation:
                    # Find if negation is near the phrase
                    phrase_idx = response_lower.find(phrase)
                    for neg in negation_words:
                        neg_idx = response_lower.find(neg)
                        if neg_idx != -1 and abs(neg_idx - phrase_idx) < 20:
                            score = 1 - score
                            break

                return ParsedConfidence(
                    confidence_score=score,
                    raw_confidence=phrase,
                    uncertainty_notes=None,
                    parse_method="verbal",
                )

    return ParsedConfidence(None, None, None, "verbal")


# =============================================================================
# PROMPT TEMPLATES FOR CONFIDENCE ELICITATION
# =============================================================================


CONFIDENCE_ELICITATION_PROMPTS = {
    "numeric": """
After your answer, provide a confidence score from 0-100 indicating how confident you are in your answer.

Format your response as:
```json
{
  "answer": "your answer here",
  "confidence": 85,
  "uncertainty_notes": "brief explanation of any uncertainty"
}
```
""",

    "verbal": """
After your answer, describe your confidence level using one of these categories:
- Very High (90-100%): Almost certain this is correct
- High (70-90%): Confident but with some uncertainty
- Medium (50-70%): Somewhat confident, notable uncertainty
- Low (30-50%): Uncertain, this is my best guess
- Very Low (0-30%): Highly uncertain, largely guessing
""",

    "calibrated": """
After your answer, provide a probability estimate (0-100%) representing how likely your answer is correct.

Important: Your probability should be calibrated - if you say 80% confident, you should be correct about 80% of the time on questions where you give that confidence level.

Format:
Answer: [your answer]
Confidence: [0-100]%
Reasoning for confidence: [brief explanation]
""",

    "comparative": """
After your answer, compare your confidence to these reference points:
- As confident as knowing 2+2=4 (99%+)
- As confident as knowing the capital of France (95%)
- As confident as remembering what you had for breakfast yesterday (80%)
- As confident as predicting tomorrow's weather (60%)
- As confident as guessing a coin flip (50%)

State which reference point best matches your confidence.
""",
}


def get_confidence_prompt(style: str = "numeric") -> str:
    """Get a prompt template for confidence elicitation."""
    return CONFIDENCE_ELICITATION_PROMPTS.get(style, CONFIDENCE_ELICITATION_PROMPTS["numeric"])


# =============================================================================
# CALIBRATION METRICS
# =============================================================================


@dataclass
class CalibrationBin:
    """Statistics for a single calibration bin."""

    bin_lower: float
    bin_upper: float
    bin_center: float
    n_samples: int
    mean_confidence: float
    accuracy: float
    calibration_error: float  # |accuracy - mean_confidence|


@dataclass
class CalibrationResult:
    """Complete calibration analysis result."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Average Calibration Error (unweighted)
    overconfidence_rate: float  # Fraction of bins where confidence > accuracy
    underconfidence_rate: float  # Fraction of bins where confidence < accuracy
    bins: list[CalibrationBin]
    n_samples: int
    n_bins: int

    # Per-sample data for plotting
    confidences: list[float] = field(repr=False, default_factory=list)
    correctness: list[bool] = field(repr=False, default_factory=list)

    def summary(self) -> str:
        lines = [
            "Calibration Analysis",
            "=" * 40,
            f"ECE (Expected Calibration Error): {self.ece:.4f}",
            f"MCE (Maximum Calibration Error): {self.mce:.4f}",
            f"ACE (Average Calibration Error): {self.ace:.4f}",
            f"Overconfidence rate: {self.overconfidence_rate:.1%}",
            f"Underconfidence rate: {self.underconfidence_rate:.1%}",
            f"Samples: {self.n_samples}, Bins: {self.n_bins}",
            "",
            "Reliability Diagram Data:",
            "| Bin | Confidence | Accuracy | Error | N |",
            "|-----|------------|----------|-------|---|",
        ]

        for b in self.bins:
            if b.n_samples > 0:
                lines.append(
                    f"| {b.bin_center:.2f} | {b.mean_confidence:.3f} | "
                    f"{b.accuracy:.3f} | {b.calibration_error:.3f} | {b.n_samples} |"
                )

        return "\n".join(lines)

    def interpretation(self) -> str:
        """Provide interpretation of calibration results."""
        if self.ece < 0.05:
            quality = "excellent"
            advice = "Model confidence scores are well-calibrated."
        elif self.ece < 0.10:
            quality = "good"
            advice = "Model is reasonably well-calibrated with minor deviations."
        elif self.ece < 0.20:
            quality = "moderate"
            advice = "Model shows some calibration issues. Consider temperature scaling."
        else:
            quality = "poor"
            advice = "Model is poorly calibrated. Confidence scores are unreliable."

        direction = ""
        if self.overconfidence_rate > 0.6:
            direction = "Model tends to be overconfident (confidence > accuracy)."
        elif self.underconfidence_rate > 0.6:
            direction = "Model tends to be underconfident (confidence < accuracy)."

        return f"Calibration quality: {quality}. {advice} {direction}"


def expected_calibration_error(
    confidences: ArrayLike,
    correctness: ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> CalibrationResult:
    """
    Compute Expected Calibration Error and related metrics.

    ECE measures how well confidence scores align with actual accuracy.
    A well-calibrated model has ECE close to 0.

    Args:
        confidences: Model confidence scores (0-1)
        correctness: Binary correctness labels (0 or 1, or bool)
        n_bins: Number of bins for calibration
        strategy: "uniform" for equal-width bins, "quantile" for equal-count bins

    Returns:
        CalibrationResult with ECE, MCE, and per-bin statistics

    Example:
        >>> confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        >>> correct = [1, 1, 0, 1, 0]
        >>> result = expected_calibration_error(confidences, correct)
        >>> print(f"ECE: {result.ece:.4f}")
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)

    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have same length")

    n_samples = len(confidences)

    if n_samples == 0:
        return CalibrationResult(
            ece=0.0, mce=0.0, ace=0.0,
            overconfidence_rate=0.0, underconfidence_rate=0.0,
            bins=[], n_samples=0, n_bins=n_bins,
        )

    # Determine bin boundaries
    if strategy == "uniform":
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_boundaries = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
        bin_boundaries[0] = 0
        bin_boundaries[-1] = 1

    bins = []
    ece = 0.0
    mce = 0.0
    calibration_errors = []
    overconfident_bins = 0
    underconfident_bins = 0
    non_empty_bins = 0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        bin_center = (bin_lower + bin_upper) / 2

        # Select samples in this bin
        if i == n_bins - 1:
            # Include upper boundary for last bin
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_confidences = confidences[mask]
        bin_correct = correctness[mask]
        n_in_bin = len(bin_confidences)

        if n_in_bin > 0:
            mean_confidence = float(np.mean(bin_confidences))
            accuracy = float(np.mean(bin_correct))
            calibration_error = abs(accuracy - mean_confidence)

            ece += (n_in_bin / n_samples) * calibration_error
            mce = max(mce, calibration_error)
            calibration_errors.append(calibration_error)

            if mean_confidence > accuracy:
                overconfident_bins += 1
            elif mean_confidence < accuracy:
                underconfident_bins += 1

            non_empty_bins += 1
        else:
            mean_confidence = bin_center
            accuracy = 0.0
            calibration_error = 0.0

        bins.append(CalibrationBin(
            bin_lower=bin_lower,
            bin_upper=bin_upper,
            bin_center=bin_center,
            n_samples=n_in_bin,
            mean_confidence=mean_confidence,
            accuracy=accuracy,
            calibration_error=calibration_error,
        ))

    # Average calibration error (unweighted)
    ace = float(np.mean(calibration_errors)) if calibration_errors else 0.0

    # Rates
    overconfidence_rate = overconfident_bins / non_empty_bins if non_empty_bins > 0 else 0.0
    underconfidence_rate = underconfident_bins / non_empty_bins if non_empty_bins > 0 else 0.0

    return CalibrationResult(
        ece=float(ece),
        mce=float(mce),
        ace=ace,
        overconfidence_rate=overconfidence_rate,
        underconfidence_rate=underconfidence_rate,
        bins=bins,
        n_samples=n_samples,
        n_bins=n_bins,
        confidences=confidences.tolist(),
        correctness=correctness.astype(bool).tolist(),
    )


def static_calibration_error(
    confidences: ArrayLike,
    correctness: ArrayLike,
) -> float:
    """
    Compute Static Calibration Error (class-independent ECE).

    Alternative to ECE that doesn't require binning.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)

    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    sorted_conf = confidences[sorted_indices]
    sorted_correct = correctness[sorted_indices]

    # Compute cumulative accuracy
    cumsum = np.cumsum(sorted_correct)
    n = np.arange(1, len(sorted_correct) + 1)
    cumulative_acc = cumsum / n

    # SCE is the average absolute difference
    return float(np.mean(np.abs(sorted_conf - cumulative_acc)))


# =============================================================================
# TEMPERATURE SCALING
# =============================================================================


@dataclass
class TemperatureScalingResult:
    """Result of temperature scaling calibration."""

    optimal_temperature: float
    ece_before: float
    ece_after: float
    improvement: float

    def apply(self, logits: ArrayLike) -> np.ndarray:
        """Apply temperature scaling to logits."""
        logits = np.asarray(logits)
        return logits / self.optimal_temperature


def find_optimal_temperature(
    confidences: ArrayLike,
    correctness: ArrayLike,
    temperature_range: tuple[float, float] = (0.1, 5.0),
    n_steps: int = 50,
) -> TemperatureScalingResult:
    """
    Find optimal temperature for calibration.

    Temperature scaling is a post-hoc calibration method that divides
    logits by a learned temperature parameter.

    Args:
        confidences: Original model confidences (0-1)
        correctness: Binary correctness labels
        temperature_range: Range of temperatures to search
        n_steps: Number of temperatures to try

    Returns:
        TemperatureScalingResult with optimal temperature
    """
    confidences = np.asarray(confidences)
    correctness = np.asarray(correctness)

    # Convert confidences to pseudo-logits
    # Clip to avoid log(0)
    confidences_clipped = np.clip(confidences, 1e-7, 1 - 1e-7)
    logits = np.log(confidences_clipped / (1 - confidences_clipped))

    # Original ECE
    ece_before = expected_calibration_error(confidences, correctness).ece

    # Search for optimal temperature
    temperatures = np.linspace(temperature_range[0], temperature_range[1], n_steps)
    best_temp = 1.0
    best_ece = ece_before

    for temp in temperatures:
        # Apply temperature scaling
        scaled_logits = logits / temp
        scaled_conf = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid

        # Compute ECE
        ece = expected_calibration_error(scaled_conf, correctness).ece

        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    return TemperatureScalingResult(
        optimal_temperature=float(best_temp),
        ece_before=ece_before,
        ece_after=best_ece,
        improvement=ece_before - best_ece,
    )


# =============================================================================
# BRIER SCORE
# =============================================================================


@dataclass
class BrierScoreResult:
    """Brier score decomposition result."""

    brier_score: float  # Overall Brier score (lower is better)
    reliability: float  # Calibration component
    resolution: float  # Discrimination component
    uncertainty: float  # Base rate uncertainty

    def summary(self) -> str:
        return (
            f"Brier Score: {self.brier_score:.4f}\n"
            f"  Reliability (calibration): {self.reliability:.4f}\n"
            f"  Resolution (discrimination): {self.resolution:.4f}\n"
            f"  Uncertainty (base rate): {self.uncertainty:.4f}"
        )


def brier_score(
    confidences: ArrayLike,
    correctness: ArrayLike,
    decompose: bool = True,
) -> BrierScoreResult | float:
    """
    Compute Brier score and optionally decompose it.

    Brier score measures both calibration and discrimination.
    Lower is better (0 = perfect, 1 = worst).

    Args:
        confidences: Model confidence scores (0-1)
        correctness: Binary correctness labels
        decompose: Whether to return full decomposition

    Returns:
        BrierScoreResult if decompose=True, else float
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)

    # Basic Brier score: mean squared error
    bs = float(np.mean((confidences - correctness) ** 2))

    if not decompose:
        return bs

    # Decomposition requires binning
    n_bins = 10
    n = len(confidences)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    # Base rate
    base_rate = np.mean(correctness)
    uncertainty = base_rate * (1 - base_rate)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])

        n_k = np.sum(mask)
        if n_k > 0:
            o_k = np.mean(correctness[mask])  # Accuracy in bin
            f_k = np.mean(confidences[mask])  # Mean confidence in bin

            reliability += (n_k / n) * (f_k - o_k) ** 2
            resolution += (n_k / n) * (o_k - base_rate) ** 2

    return BrierScoreResult(
        brier_score=bs,
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
    )


# =============================================================================
# RELIABILITY DIAGRAM DATA
# =============================================================================


@dataclass
class ReliabilityDiagramData:
    """Data for plotting a reliability diagram."""

    bin_centers: list[float]
    accuracies: list[float]
    confidences: list[float]
    bin_counts: list[int]
    perfect_calibration: list[float]  # Diagonal reference line


def get_reliability_diagram_data(calibration_result: CalibrationResult) -> ReliabilityDiagramData:
    """
    Extract data for plotting a reliability diagram.

    A reliability diagram plots accuracy vs confidence for each bin.
    Perfect calibration would be a diagonal line.

    Args:
        calibration_result: Result from expected_calibration_error

    Returns:
        ReliabilityDiagramData for plotting
    """
    non_empty_bins = [b for b in calibration_result.bins if b.n_samples > 0]

    return ReliabilityDiagramData(
        bin_centers=[b.bin_center for b in non_empty_bins],
        accuracies=[b.accuracy for b in non_empty_bins],
        confidences=[b.mean_confidence for b in non_empty_bins],
        bin_counts=[b.n_samples for b in non_empty_bins],
        perfect_calibration=[b.bin_center for b in non_empty_bins],
    )


def plot_reliability_diagram_ascii(calibration_result: CalibrationResult, width: int = 60) -> str:
    """
    Generate ASCII reliability diagram.

    Args:
        calibration_result: Calibration analysis result
        width: Width of the diagram in characters

    Returns:
        ASCII art reliability diagram
    """
    data = get_reliability_diagram_data(calibration_result)

    if not data.bin_centers:
        return "No data to plot"

    lines = [
        "Reliability Diagram",
        "=" * width,
        "Acc",
        "1.0 |" + " " * (width - 6) + "|",
    ]

    # Create grid
    height = 10
    grid = [[" " for _ in range(width - 6)] for _ in range(height)]

    # Plot diagonal (perfect calibration)
    for i in range(width - 6):
        x = i / (width - 7)
        y_idx = int(x * (height - 1))
        if 0 <= y_idx < height:
            grid[height - 1 - y_idx][i] = "."

    # Plot actual calibration points
    for conf, acc, count in zip(data.confidences, data.accuracies, data.bin_counts):
        x_idx = int(conf * (width - 7))
        y_idx = int(acc * (height - 1))
        if 0 <= x_idx < width - 6 and 0 <= y_idx < height:
            # Use different symbols based on count
            if count >= 10:
                symbol = "●"
            elif count >= 5:
                symbol = "○"
            else:
                symbol = "·"
            grid[height - 1 - y_idx][x_idx] = symbol

    # Build output
    for i, row in enumerate(grid):
        y_val = 1.0 - i / (height - 1)
        if i == height // 2:
            lines.append(f"{y_val:.1f} |" + "".join(row) + "|")
        else:
            lines.append("    |" + "".join(row) + "|")

    lines.append("0.0 |" + "_" * (width - 6) + "|")
    lines.append("    0.0" + " " * (width - 14) + "1.0")
    lines.append(" " * ((width - 10) // 2) + "Confidence")
    lines.append("")
    lines.append(f"ECE: {calibration_result.ece:.4f}")
    lines.append("Legend: ● (n≥10)  ○ (n≥5)  · (n<5)  . (perfect)")

    return "\n".join(lines)


# =============================================================================
# AGGREGATE CALIBRATION ANALYSIS
# =============================================================================


@dataclass
class AggregateCalibrationAnalysis:
    """Aggregate calibration analysis across multiple conditions or models."""

    condition_results: dict[str, CalibrationResult]
    best_calibrated: str
    worst_calibrated: str
    average_ece: float
    ece_range: tuple[float, float]


def compare_calibration(
    results: dict[str, tuple[ArrayLike, ArrayLike]],
    n_bins: int = 10,
) -> AggregateCalibrationAnalysis:
    """
    Compare calibration across multiple conditions or models.

    Args:
        results: Dict mapping condition names to (confidences, correctness) tuples
        n_bins: Number of bins for ECE calculation

    Returns:
        AggregateCalibrationAnalysis with comparison
    """
    condition_results = {}
    eces = {}

    for name, (conf, correct) in results.items():
        cal_result = expected_calibration_error(conf, correct, n_bins=n_bins)
        condition_results[name] = cal_result
        eces[name] = cal_result.ece

    best = min(eces, key=eces.get)
    worst = max(eces, key=eces.get)

    return AggregateCalibrationAnalysis(
        condition_results=condition_results,
        best_calibrated=best,
        worst_calibrated=worst,
        average_ece=float(np.mean(list(eces.values()))),
        ece_range=(min(eces.values()), max(eces.values())),
    )
