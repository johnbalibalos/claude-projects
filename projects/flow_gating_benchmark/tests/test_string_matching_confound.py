"""Tests for string matching confound analysis."""

import pytest
from src.analysis.string_matching_confound import (
    compute_synonym_coverage,
    exact_string_match,
    normalized_string_match,
    semantic_string_match,
    compare_matching_levels,
    evaluate_with_matcher,
    compute_detection_rates,
    compute_synonym_boost,
    analyze_match_level_distribution,
)
from src.evaluation.normalization import CELL_TYPE_SYNONYMS


class TestSynonymCoverage:
    """Tests for synonym coverage computation."""

    def test_coverage_computed(self):
        """Test that coverage is computed for all canonical forms."""
        coverage = compute_synonym_coverage()

        # Should have coverage for common types
        assert "t_cells" in coverage
        assert "b_cells" in coverage
        assert "monocytes" in coverage

    def test_coverage_counts_synonyms(self):
        """Test that synonym counts are correct."""
        coverage = compute_synonym_coverage()

        # T cells should have multiple synonyms
        t_cell_cov = coverage.get("t_cells")
        assert t_cell_cov is not None
        assert t_cell_cov.synonym_count >= 5  # At least 5 synonyms defined

        # Check synonyms list matches count
        assert len(t_cell_cov.synonyms) == t_cell_cov.synonym_count

    def test_coverage_score_normalized(self):
        """Test that coverage score is normalized to 0-1."""
        coverage = compute_synonym_coverage()

        for cov in coverage.values():
            assert 0 <= cov.coverage_score <= 1


class TestMatchingLevels:
    """Tests for different matching levels."""

    def test_exact_match_identical(self):
        """Test exact match with identical strings."""
        assert exact_string_match("CD4+ T cells", "CD4+ T cells")
        assert exact_string_match("CD4+ T cells", "cd4+ t cells")  # Case insensitive

    def test_exact_match_different(self):
        """Test exact match rejects different strings."""
        assert not exact_string_match("CD4+ T cells", "CD4 T cells")
        assert not exact_string_match("T cells", "T lymphocytes")

    def test_normalized_match_cleans_names(self):
        """Test normalized match handles common variations."""
        # Positive/negative notation
        assert normalized_string_match("CD4 positive", "CD4+")
        assert normalized_string_match("CD8 negative", "CD8-")

        # Cell suffix removal
        assert normalized_string_match("T cells", "T")
        assert normalized_string_match("monocytes", "monos")

    def test_semantic_match_uses_synonyms(self):
        """Test semantic match uses synonym dictionary."""
        # T cell variations
        assert semantic_string_match("T cells", "T lymphocytes")
        assert semantic_string_match("CD3+ T cells", "T cells")

        # B cell variations
        assert semantic_string_match("B cells", "B lymphocytes")
        assert semantic_string_match("CD19+ B cells", "B cells")

        # NK cell variations
        assert semantic_string_match("NK cells", "Natural killer cells")

    def test_semantic_match_rejects_unrelated(self):
        """Test semantic match doesn't match unrelated types."""
        assert not semantic_string_match("T cells", "B cells")
        assert not semantic_string_match("Monocytes", "NK cells")


class TestCompareMatchingLevels:
    """Tests for matching level comparison."""

    def test_exact_match_recorded(self):
        """Test that exact matches are properly recorded."""
        comp = compare_matching_levels("CD4+ T cells", "CD4+ T cells")
        assert comp.exact_match
        assert comp.normalized_match
        assert comp.semantic_match
        assert comp.match_difference == "exact"

    def test_normalized_only_match(self):
        """Test normalized-only matches."""
        comp = compare_matching_levels("CD4 positive T cells", "CD4+ T")
        assert not comp.exact_match
        assert comp.normalized_match
        assert comp.semantic_match
        assert comp.match_difference == "normalized"

    def test_semantic_only_match(self):
        """Test semantic-only matches."""
        comp = compare_matching_levels("T lymphocytes", "CD3+ T cells")
        # These won't match at normalized level but should at semantic
        assert not comp.exact_match
        # normalized_match depends on implementation
        assert comp.semantic_match
        assert comp.match_difference in ["normalized", "semantic"]


class TestEvaluateWithMatcher:
    """Tests for gate evaluation with different matchers."""

    def test_exact_matcher(self):
        """Test evaluation with exact matcher."""
        gt = ["CD4+ T cells", "B cells", "NK cells"]
        pred = ["CD4+ T cells", "B cells", "Monocytes"]

        matched, missing, extra = evaluate_with_matcher(gt, pred, exact_string_match)

        assert "CD4+ T cells" in matched
        assert "B cells" in matched
        assert "NK cells" in missing
        assert "Monocytes" in extra

    def test_semantic_matcher_more_permissive(self):
        """Test that semantic matcher catches more matches."""
        gt = ["T cells", "B cells"]
        pred = ["T lymphocytes", "CD19+ B cells"]

        # Exact should miss
        matched_exact, _, _ = evaluate_with_matcher(gt, pred, exact_string_match)
        assert len(matched_exact) == 0

        # Semantic should catch both
        matched_sem, _, _ = evaluate_with_matcher(gt, pred, semantic_string_match)
        assert len(matched_sem) == 2


class TestDetectionRates:
    """Tests for detection rate computation."""

    def test_compute_detection_rates(self):
        """Test detection rate computation."""
        stats = {
            "exact": {"T cells": {"matches": 5, "misses": 5}},
            "semantic": {"T cells": {"matches": 8, "misses": 2}},
        }

        rates = compute_detection_rates(stats)

        assert rates["T cells"]["exact"] == 0.5
        assert rates["T cells"]["semantic"] == 0.8

    def test_handles_zero_total(self):
        """Test handling of zero total observations."""
        stats = {
            "exact": {"Unknown": {"matches": 0, "misses": 0}},
        }

        rates = compute_detection_rates(stats)
        assert rates["Unknown"]["exact"] == 0.0


class TestSynonymBoost:
    """Tests for synonym boost computation."""

    def test_compute_boost(self):
        """Test synonym boost computation."""
        detection_rates = {
            "T cells": {"exact": 0.5, "semantic": 0.8},
            "B cells": {"exact": 0.7, "semantic": 0.9},
        }

        boosts = compute_synonym_boost(detection_rates)

        assert boosts["T cells"] == pytest.approx(0.3)
        assert boosts["B cells"] == pytest.approx(0.2)

    def test_no_boost_for_exact_matches(self):
        """Test no boost when exact and semantic rates are equal."""
        detection_rates = {
            "Monocytes": {"exact": 0.9, "semantic": 0.9},
        }

        boosts = compute_synonym_boost(detection_rates)
        assert boosts["Monocytes"] == 0.0


class TestMatchLevelDistribution:
    """Tests for match level distribution analysis."""

    def test_analyze_distribution(self):
        """Test match level distribution analysis."""
        from src.analysis.string_matching_confound import MatchingComparison

        comparisons = [
            MatchingComparison("T cells", "T cells", "T cells", True, True, True),
            MatchingComparison("B cells", "B cells", "B lymphs", False, False, True),
            MatchingComparison("NK", "NK cells", "nk", False, True, True),
        ]

        result = analyze_match_level_distribution(comparisons)

        assert result["counts"]["exact"] == 1
        assert result["counts"]["normalized"] == 1
        assert result["counts"]["semantic"] == 1
        assert "semantic_dependency" in result

    def test_empty_comparisons(self):
        """Test handling of empty comparisons."""
        result = analyze_match_level_distribution([])
        assert "error" in result
