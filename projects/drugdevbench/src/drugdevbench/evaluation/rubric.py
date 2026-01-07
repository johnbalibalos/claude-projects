"""Scoring rubric for evaluating model responses."""

import re
from dataclasses import dataclass
from typing import Any

from drugdevbench.data.schemas import QuestionType


@dataclass
class ScoringResult:
    """Result of scoring a response."""

    score: float  # 0.0 to 1.0
    rationale: str
    matched_answer: str | None = None
    metadata: dict[str, Any] | None = None


# Scoring criteria by question type
SCORING_RUBRIC = {
    QuestionType.FACTUAL_EXTRACTION: {
        "description": "Exact or near-exact match with gold answer",
        "scoring": {
            1.0: "Exact match or semantically equivalent",
            0.75: "Correct value with minor formatting difference",
            0.5: "Partially correct (right order of magnitude)",
            0.25: "Related but incorrect value",
            0.0: "Completely incorrect or no answer",
        },
    },
    QuestionType.VISUAL_ESTIMATION: {
        "description": "Estimate within acceptable tolerance of gold answer",
        "scoring": {
            1.0: "Within 10% of gold answer",
            0.75: "Within 25% of gold answer",
            0.5: "Within 50% of gold answer (right order of magnitude)",
            0.25: "Correct direction but >50% error",
            0.0: "Completely incorrect or no answer",
        },
    },
    QuestionType.QUALITY_ASSESSMENT: {
        "description": "Correct identification of quality issues",
        "scoring": {
            1.0: "Correctly identifies all relevant quality aspects",
            0.75: "Identifies main quality issue correctly",
            0.5: "Partially correct assessment",
            0.25: "Addresses quality but misses key points",
            0.0: "Incorrect assessment or no answer",
        },
    },
    QuestionType.INTERPRETATION: {
        "description": "Reasonable scientific interpretation",
        "scoring": {
            1.0: "Comprehensive, scientifically sound interpretation",
            0.75: "Correct main interpretation with minor gaps",
            0.5: "Partially correct interpretation",
            0.25: "Some relevant points but major gaps",
            0.0: "Incorrect interpretation or no answer",
        },
    },
    QuestionType.ERROR_DETECTION: {
        "description": "Correct identification of errors or issues",
        "scoring": {
            1.0: "Correctly identifies all significant issues",
            0.75: "Identifies main issues",
            0.5: "Identifies some issues but misses others",
            0.25: "Identifies minor issues only",
            0.0: "Fails to identify issues or false positives",
        },
    },
}


def _normalize_value(value: str) -> tuple[float | None, str]:
    """Extract numeric value and unit from a string.

    Args:
        value: String potentially containing a numeric value

    Returns:
        Tuple of (numeric_value, unit_string)
    """
    # Remove common formatting
    value = value.strip().lower()

    # Try to extract number and unit
    match = re.search(r"([\d.]+)\s*([a-zμµ/%]+)?", value)
    if match:
        try:
            num = float(match.group(1))
            unit = match.group(2) or ""
            return num, unit
        except ValueError:
            pass

    return None, value


def _normalize_unit(unit: str) -> str:
    """Normalize unit string for comparison.

    Args:
        unit: Unit string

    Returns:
        Normalized unit
    """
    unit = unit.lower().strip()
    # Handle common variations
    replacements = {
        "μm": "um",
        "µm": "um",
        "nm": "nm",
        "pm": "pm",
        "mm": "mm",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "min": "min",
        "mins": "min",
        "minute": "min",
        "minutes": "min",
        "kda": "kda",
        "ng/ml": "ng/ml",
        "ug/ml": "ug/ml",
        "μg/ml": "ug/ml",
        "µg/ml": "ug/ml",
    }
    return replacements.get(unit, unit)


def _compare_numeric_values(
    response_value: float,
    gold_value: float,
    tolerance_pct: float = 10.0,
) -> tuple[float, str]:
    """Compare two numeric values with tolerance.

    Args:
        response_value: Value from model response
        gold_value: Gold standard value
        tolerance_pct: Tolerance percentage for exact match

    Returns:
        Tuple of (score, rationale)
    """
    if gold_value == 0:
        # Avoid division by zero
        if response_value == 0:
            return 1.0, "Both values are zero"
        return 0.0, "Gold value is zero but response is not"

    pct_error = abs(response_value - gold_value) / abs(gold_value) * 100

    if pct_error <= tolerance_pct:
        return 1.0, f"Within {tolerance_pct}% tolerance ({pct_error:.1f}% error)"
    elif pct_error <= 25:
        return 0.75, f"Within 25% tolerance ({pct_error:.1f}% error)"
    elif pct_error <= 50:
        return 0.5, f"Within 50% tolerance ({pct_error:.1f}% error)"
    elif pct_error <= 100:
        return 0.25, f"Same order of magnitude ({pct_error:.1f}% error)"
    else:
        return 0.0, f"Large error ({pct_error:.1f}%)"


def _compare_boolean(response: str, gold: str) -> tuple[float, str]:
    """Compare boolean/yes-no type answers.

    Args:
        response: Model response
        gold: Gold answer

    Returns:
        Tuple of (score, rationale)
    """
    response_lower = response.lower()
    gold_lower = gold.lower()

    positive_terms = ["yes", "true", "correct", "present", "shown", "included"]
    negative_terms = ["no", "false", "incorrect", "absent", "not shown", "not included"]

    response_positive = any(term in response_lower for term in positive_terms)
    response_negative = any(term in response_lower for term in negative_terms)
    gold_positive = any(term in gold_lower for term in positive_terms)
    gold_negative = any(term in gold_lower for term in negative_terms)

    if (response_positive and gold_positive) or (response_negative and gold_negative):
        return 1.0, "Boolean match"
    elif response_positive != gold_positive:
        return 0.0, "Boolean mismatch"
    else:
        return 0.5, "Ambiguous boolean comparison"


def score_response(
    response_text: str,
    gold_answer: str,
    question_type: QuestionType,
) -> ScoringResult:
    """Score a model response against a gold answer.

    Args:
        response_text: The model's response
        gold_answer: The expected correct answer
        question_type: Type of question for scoring criteria

    Returns:
        ScoringResult with score and rationale
    """
    response_text = response_text.strip()
    gold_answer = gold_answer.strip()

    # Handle empty responses
    if not response_text or response_text.upper() == "ERROR":
        return ScoringResult(
            score=0.0,
            rationale="No response or error",
        )

    # Exact match (case-insensitive)
    if response_text.lower() == gold_answer.lower():
        return ScoringResult(
            score=1.0,
            rationale="Exact match",
            matched_answer=gold_answer,
        )

    # Check if gold answer appears in response
    if gold_answer.lower() in response_text.lower():
        return ScoringResult(
            score=1.0,
            rationale="Gold answer found in response",
            matched_answer=gold_answer,
        )

    # For factual extraction and visual estimation, try numeric comparison
    if question_type in (QuestionType.FACTUAL_EXTRACTION, QuestionType.VISUAL_ESTIMATION):
        response_num, response_unit = _normalize_value(response_text)
        gold_num, gold_unit = _normalize_value(gold_answer)

        if response_num is not None and gold_num is not None:
            # Check units match (if both have units)
            if response_unit and gold_unit:
                if _normalize_unit(response_unit) != _normalize_unit(gold_unit):
                    return ScoringResult(
                        score=0.25,
                        rationale=f"Unit mismatch: {response_unit} vs {gold_unit}",
                        metadata={"response_value": response_num, "gold_value": gold_num},
                    )

            # Compare numeric values
            tolerance = 10.0 if question_type == QuestionType.FACTUAL_EXTRACTION else 25.0
            score, rationale = _compare_numeric_values(response_num, gold_num, tolerance)
            return ScoringResult(
                score=score,
                rationale=rationale,
                matched_answer=str(response_num),
                metadata={"response_value": response_num, "gold_value": gold_num},
            )

    # For quality assessment, try boolean comparison
    if question_type == QuestionType.QUALITY_ASSESSMENT:
        score, rationale = _compare_boolean(response_text, gold_answer)
        if score > 0:
            return ScoringResult(score=score, rationale=rationale)

    # Default: partial credit for mentioning relevant terms
    gold_terms = set(gold_answer.lower().split())
    response_terms = set(response_text.lower().split())
    overlap = gold_terms & response_terms

    if len(overlap) > 0:
        overlap_ratio = len(overlap) / len(gold_terms)
        if overlap_ratio >= 0.5:
            return ScoringResult(
                score=0.5,
                rationale=f"Partial term overlap ({len(overlap)}/{len(gold_terms)} terms)",
                metadata={"overlapping_terms": list(overlap)},
            )
        else:
            return ScoringResult(
                score=0.25,
                rationale=f"Minimal term overlap ({len(overlap)}/{len(gold_terms)} terms)",
                metadata={"overlapping_terms": list(overlap)},
            )

    return ScoringResult(
        score=0.0,
        rationale="No match found",
    )
