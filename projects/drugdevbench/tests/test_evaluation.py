"""Tests for evaluation and scoring."""

import pytest

from drugdevbench.data.schemas import QuestionType
from drugdevbench.evaluation.rubric import (
    score_response,
    ScoringResult,
    _normalize_value,
    _compare_numeric_values,
)


class TestNormalizeValue:
    """Test value normalization helper."""

    def test_simple_number(self):
        """Should extract simple numbers."""
        num, unit = _normalize_value("42")
        assert num == 42.0
        assert unit == ""

    def test_number_with_unit(self):
        """Should extract number and unit."""
        num, unit = _normalize_value("2.5 nM")
        assert num == 2.5
        assert unit == "nm"

    def test_number_with_micromolar(self):
        """Should handle μM notation."""
        num, unit = _normalize_value("10 μM")
        assert num == 10.0

    def test_percentage(self):
        """Should handle percentages."""
        num, unit = _normalize_value("75%")
        assert num == 75.0
        assert "%" in unit

    def test_non_numeric(self):
        """Should return None for non-numeric strings."""
        num, unit = _normalize_value("yes")
        assert num is None


class TestCompareNumericValues:
    """Test numeric value comparison."""

    def test_exact_match(self):
        """Same values should score 1.0."""
        score, _ = _compare_numeric_values(2.5, 2.5)
        assert score == 1.0

    def test_within_10_percent(self):
        """Values within 10% should score 1.0."""
        score, _ = _compare_numeric_values(2.6, 2.5, tolerance_pct=10.0)
        assert score == 1.0

    def test_within_25_percent(self):
        """Values within 25% should score 0.75."""
        score, _ = _compare_numeric_values(3.0, 2.5, tolerance_pct=10.0)
        assert score == 0.75

    def test_within_50_percent(self):
        """Values within 50% should score 0.5."""
        score, _ = _compare_numeric_values(3.5, 2.5, tolerance_pct=10.0)
        assert score == 0.5

    def test_large_error(self):
        """Large errors should score 0.0."""
        score, _ = _compare_numeric_values(10.0, 2.5, tolerance_pct=10.0)
        assert score == 0.0


class TestScoreResponse:
    """Test response scoring."""

    def test_exact_match(self):
        """Exact match should score 1.0."""
        result = score_response(
            response_text="2.5 nM",
            gold_answer="2.5 nM",
            question_type=QuestionType.FACTUAL_EXTRACTION,
        )
        assert result.score == 1.0

    def test_case_insensitive_match(self):
        """Match should be case-insensitive."""
        result = score_response(
            response_text="YES",
            gold_answer="yes",
            question_type=QuestionType.QUALITY_ASSESSMENT,
        )
        assert result.score == 1.0

    def test_gold_in_response(self):
        """Gold answer found in response should score 1.0."""
        result = score_response(
            response_text="The IC50 is approximately 2.5 nM based on the curve.",
            gold_answer="2.5 nM",
            question_type=QuestionType.FACTUAL_EXTRACTION,
        )
        assert result.score == 1.0

    def test_numeric_close_match(self):
        """Close numeric values should get partial credit."""
        result = score_response(
            response_text="2.6 nM",
            gold_answer="2.5 nM",
            question_type=QuestionType.FACTUAL_EXTRACTION,
        )
        assert result.score >= 0.75

    def test_visual_estimation_tolerance(self):
        """Visual estimation should have higher tolerance."""
        result = score_response(
            response_text="3.0 hours",
            gold_answer="2.5 hours",
            question_type=QuestionType.VISUAL_ESTIMATION,
        )
        # 20% error should still score reasonably well for estimation
        assert result.score >= 0.5

    def test_empty_response(self):
        """Empty response should score 0.0."""
        result = score_response(
            response_text="",
            gold_answer="2.5 nM",
            question_type=QuestionType.FACTUAL_EXTRACTION,
        )
        assert result.score == 0.0

    def test_error_response(self):
        """Error response should score 0.0."""
        result = score_response(
            response_text="ERROR",
            gold_answer="2.5 nM",
            question_type=QuestionType.FACTUAL_EXTRACTION,
        )
        assert result.score == 0.0

    def test_boolean_yes_match(self):
        """Yes/no questions should match appropriately."""
        result = score_response(
            response_text="Yes, a loading control is shown.",
            gold_answer="Yes",
            question_type=QuestionType.QUALITY_ASSESSMENT,
        )
        assert result.score == 1.0

    def test_boolean_no_match(self):
        """No answers should match appropriately."""
        result = score_response(
            response_text="No, there is no loading control visible.",
            gold_answer="No",
            question_type=QuestionType.QUALITY_ASSESSMENT,
        )
        assert result.score == 1.0

    def test_wrong_unit(self):
        """Different units should be penalized."""
        result = score_response(
            response_text="2.5 μM",  # μM instead of nM
            gold_answer="2.5 nM",
            question_type=QuestionType.FACTUAL_EXTRACTION,
        )
        assert result.score < 1.0

    def test_completely_wrong(self):
        """Completely wrong answer should score 0.0."""
        result = score_response(
            response_text="The sky is blue",
            gold_answer="2.5 nM",
            question_type=QuestionType.FACTUAL_EXTRACTION,
        )
        assert result.score == 0.0

    def test_partial_term_overlap(self):
        """Partial term overlap should get some credit."""
        result = score_response(
            response_text="The half-life appears to be around 4 hours",
            gold_answer="half-life of 4.2 hours",
            question_type=QuestionType.FACTUAL_EXTRACTION,
        )
        # Should get some credit for mentioning half-life and hours
        assert result.score > 0.0


class TestScoringResult:
    """Test ScoringResult dataclass."""

    def test_scoring_result_creation(self):
        """ScoringResult should store all fields."""
        result = ScoringResult(
            score=0.75,
            rationale="Within 25% tolerance",
            matched_answer="2.6",
            metadata={"response_value": 2.6, "gold_value": 2.5},
        )
        assert result.score == 0.75
        assert result.rationale == "Within 25% tolerance"
        assert result.matched_answer == "2.6"
        assert result.metadata["response_value"] == 2.6
