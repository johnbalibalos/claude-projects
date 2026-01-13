"""Tests for semantic similarity matching.

These tests require sentence-transformers to be installed.
Some tests use mocking to avoid loading the actual model.
"""

import pytest
import numpy as np

# Skip all tests in this module if sentence-transformers is not installed
pytest.importorskip("sentence_transformers")

from src.evaluation.semantic_similarity import (
    SemanticMatcher,
    SemanticMatch,
    SemanticMatchResult,
    compute_semantic_f1,
    compute_weighted_semantic_f1,
    are_semantically_equivalent,
    cosine_similarity,
    cosine_similarity_matrix,
)


class TestCosineSimilarity:
    """Tests for cosine similarity functions."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector(self):
        """Zero vector should return 0 similarity."""
        v1 = np.array([0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 1.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == 0.0

    def test_similarity_matrix(self):
        """Test pairwise similarity matrix computation."""
        a = np.array([[1, 0], [0, 1]], dtype=np.float32)
        b = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)

        sim_matrix = cosine_similarity_matrix(a, b)

        assert sim_matrix.shape == (2, 3)
        assert sim_matrix[0, 0] == pytest.approx(1.0)  # [1,0] vs [1,0]
        assert sim_matrix[1, 1] == pytest.approx(1.0)  # [0,1] vs [0,1]
        assert sim_matrix[0, 1] == pytest.approx(0.0)  # [1,0] vs [0,1]


class TestSemanticMatcher:
    """Tests for the SemanticMatcher class."""

    @pytest.fixture
    def matcher(self):
        """Create a SemanticMatcher instance."""
        return SemanticMatcher(high_threshold=0.85, medium_threshold=0.70)

    def test_exact_match(self, matcher):
        """Exact matches should have similarity 1.0."""
        sim = matcher.compute_similarity("T cells", "T cells")
        assert sim == 1.0

    def test_case_insensitive(self, matcher):
        """Matching should be case-insensitive."""
        sim = matcher.compute_similarity("T cells", "t cells")
        assert sim == 1.0

    def test_synonyms_high_similarity(self, matcher):
        """Known synonyms should have high similarity."""
        # These are semantically equivalent cell types
        sim = matcher.compute_similarity("T cells", "T lymphocytes")
        assert sim > 0.7

    def test_unrelated_low_similarity(self, matcher):
        """Unrelated terms should have lower similarity."""
        sim = matcher.compute_similarity("T cells", "hepatocytes")
        assert sim < 0.7

    def test_find_best_match(self, matcher):
        """Test finding best match from candidates."""
        candidates = ["B cells", "NK cells", "T lymphocytes", "Monocytes"]

        match, sim = matcher.find_best_match("T cells", candidates)

        # T lymphocytes should be the best match
        assert match == "T lymphocytes"
        assert sim > matcher.medium_threshold

    def test_find_best_match_no_match(self, matcher):
        """Test when no candidates match above threshold."""
        candidates = ["hepatocytes", "neurons", "keratinocytes"]

        match, sim = matcher.find_best_match("T cells", candidates)

        # None should match above threshold
        assert match is None or sim < matcher.medium_threshold

    def test_caching(self, matcher):
        """Test that embeddings are cached."""
        # First call
        _ = matcher.get_embedding("T cells")
        assert "T cells" in matcher._embedding_cache

        # Clear and verify
        matcher.clear_cache()
        assert "T cells" not in matcher._embedding_cache


class TestMatchGateSets:
    """Tests for matching gate sets."""

    @pytest.fixture
    def matcher(self):
        """Create matcher for testing."""
        return SemanticMatcher(high_threshold=0.85, medium_threshold=0.70)

    def test_exact_matches(self, matcher):
        """Test with exact matching gate names."""
        predicted = {"T cells", "B cells", "NK cells"}
        ground_truth = {"T cells", "B cells", "NK cells"}

        result = matcher.match_gate_sets(predicted, ground_truth)

        assert len(result.matches) == 3
        assert len(result.unmatched_predicted) == 0
        assert len(result.unmatched_ground_truth) == 0

    def test_semantic_matches(self, matcher):
        """Test with semantically equivalent gates."""
        predicted = {"T lymphocytes", "B lymphocytes"}
        ground_truth = {"T cells", "B cells"}

        result = matcher.match_gate_sets(predicted, ground_truth)

        # Should find matches due to semantic similarity
        assert len(result.matches) >= 1

    def test_partial_matches(self, matcher):
        """Test with partial overlap."""
        predicted = {"T cells", "Unknown population"}
        ground_truth = {"T cells", "B cells"}

        result = matcher.match_gate_sets(predicted, ground_truth)

        assert len(result.matches) >= 1  # T cells should match
        assert len(result.unmatched_predicted) >= 1
        assert len(result.unmatched_ground_truth) >= 1

    def test_empty_sets(self, matcher):
        """Test with empty sets."""
        result = matcher.match_gate_sets(set(), {"T cells"})

        assert len(result.matches) == 0
        assert len(result.unmatched_ground_truth) == 1

    def test_similarity_matrix(self, matcher):
        """Test that similarity matrix is computed."""
        predicted = {"A", "B"}
        ground_truth = {"X", "Y"}

        result = matcher.match_gate_sets(predicted, ground_truth)

        assert result.similarity_matrix is not None
        assert result.similarity_matrix.shape == (2, 2)


class TestComputeSemanticF1:
    """Tests for semantic F1 computation."""

    def test_perfect_f1(self):
        """Perfect matches should give F1 = 1.0."""
        predicted = {"T cells", "B cells"}
        ground_truth = {"T cells", "B cells"}

        f1, precision, recall, _ = compute_semantic_f1(predicted, ground_truth)

        assert f1 == pytest.approx(1.0)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)

    def test_zero_f1_no_overlap(self):
        """Completely different sets should have low F1."""
        predicted = {"Hepatocytes", "Neurons"}
        ground_truth = {"T cells", "B cells"}

        f1, _, _, _ = compute_semantic_f1(predicted, ground_truth)

        # Should be low (possibly 0)
        assert f1 < 0.5

    def test_f1_with_synonyms(self):
        """F1 should recognize semantic equivalents."""
        predicted = {"T lymphocytes", "B lymphocytes"}
        ground_truth = {"T cells", "B cells"}

        f1, _, _, result = compute_semantic_f1(predicted, ground_truth)

        # Should find at least some matches
        assert len(result.matches) > 0


class TestComputeWeightedSemanticF1:
    """Tests for weighted semantic F1."""

    def test_weighted_perfect(self):
        """Perfect matches should give weighted F1 = 1.0."""
        predicted = {"T cells", "B cells"}
        ground_truth = {"T cells", "B cells"}

        f1, _, _, _ = compute_weighted_semantic_f1(predicted, ground_truth)
        assert f1 == pytest.approx(1.0)

    def test_weighted_accounts_for_confidence(self):
        """Weighted F1 should account for match confidence."""
        # Create sets where some matches are more confident than others
        predicted = {"T cells", "CD3+ T lymphocytes"}
        ground_truth = {"T cells", "CD3+ cells"}

        _, _, _, result = compute_weighted_semantic_f1(predicted, ground_truth)

        # Matches should have varying similarities
        if len(result.matches) > 1:
            sims = [m.similarity for m in result.matches]
            # Not all similarities should be exactly 1.0
            assert not all(s == 1.0 for s in sims) or len(sims) == 1


class TestAreSemanticallySEquivalent:
    """Tests for the convenience function."""

    def test_identical(self):
        """Identical strings are equivalent."""
        assert are_semantically_equivalent("T cells", "T cells") is True

    def test_case_difference(self):
        """Case differences should be equivalent."""
        assert are_semantically_equivalent("T cells", "t cells") is True

    def test_custom_threshold(self):
        """Test with custom threshold."""
        # Very high threshold might exclude some matches
        result = are_semantically_equivalent(
            "T cells", "T lymphocytes", threshold=0.99
        )
        # Result depends on actual embeddings, just verify it returns bool
        assert isinstance(result, bool)


class TestMatchTypes:
    """Tests for match type classification."""

    @pytest.fixture
    def matcher(self):
        return SemanticMatcher()

    def test_exact_match_type(self, matcher):
        """Exact matches should be labeled 'exact'."""
        result = matcher.match_gate_sets({"T cells"}, {"T cells"})

        assert len(result.matches) == 1
        assert result.matches[0].match_type == "exact"

    def test_high_similarity_type(self, matcher):
        """High similarity matches should be labeled appropriately."""
        result = matcher.match_gate_sets({"T lymphocytes"}, {"T cells"})

        if result.matches:
            match = result.matches[0]
            assert match.match_type in ["exact", "high_similarity", "medium_similarity"]


class TestPreprocessing:
    """Tests for gate name preprocessing."""

    @pytest.fixture
    def matcher(self):
        return SemanticMatcher()

    def test_abbreviation_expansion(self, matcher):
        """Test that common abbreviations are expanded."""
        # "mono" should be treated similarly to "monocyte"
        sim = matcher.compute_similarity("mono", "monocyte")
        # The preprocessing adds context, so similarity should be reasonable
        assert sim > 0.5

    def test_context_addition(self, matcher):
        """Test that cell population context helps matching."""
        # These should be recognized as cell populations
        sim = matcher.compute_similarity("CD4+", "CD4 positive")
        assert sim > 0.7


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def matcher(self):
        return SemanticMatcher()

    def test_empty_string(self, matcher):
        """Empty string handling."""
        sim = matcher.compute_similarity("", "T cells")
        assert isinstance(sim, float)

    def test_special_characters(self, matcher):
        """Special characters in gate names."""
        sim = matcher.compute_similarity("CD3+/CD4+", "CD3+ CD4+")
        assert isinstance(sim, float)

    def test_unicode_characters(self, matcher):
        """Unicode characters in gate names."""
        sim = matcher.compute_similarity("γδ T cells", "gamma delta T cells")
        # Should handle unicode and recognize equivalence
        assert sim > 0.5

    def test_very_long_name(self, matcher):
        """Very long gate name."""
        long_name = "CD3+ CD4+ CD8- CD25+ FoxP3+ regulatory T cells"
        sim = matcher.compute_similarity(long_name, "Regulatory T cells")
        assert isinstance(sim, float)

    def test_single_character(self, matcher):
        """Single character gate names."""
        sim = matcher.compute_similarity("A", "B")
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0


class TestIntegrationWithFlowCytometry:
    """Integration tests with realistic flow cytometry data."""

    @pytest.fixture
    def matcher(self):
        return SemanticMatcher(high_threshold=0.80, medium_threshold=0.65)

    def test_realistic_gate_matching(self, matcher):
        """Test with realistic flow cytometry gate names."""
        predicted = {
            "All Events",
            "Singlets (FSC-A vs FSC-H)",
            "Live cells",
            "CD45+ leukocytes",
            "T lymphocytes",
            "CD4+ helper T cells",
            "CD8+ cytotoxic T cells",
            "Regulatory T cells",
        }

        ground_truth = {
            "All Events",
            "Singlets",
            "Live/Dead-",
            "Leukocytes",
            "T cells",
            "CD4+ T cells",
            "CD8+ T cells",
            "Tregs",
        }

        f1, precision, recall, result = compute_semantic_f1(
            predicted, ground_truth, matcher
        )

        # Should achieve reasonable matching
        assert len(result.matches) >= 5
        assert f1 > 0.5

    def test_monocyte_subsets(self, matcher):
        """Test matching monocyte subset naming variations."""
        predicted = {
            "Classical monocytes",
            "Non-classical monocytes",
            "Intermediate monocytes",
        }

        ground_truth = {
            "CD14++ CD16- monocytes",
            "CD14+ CD16++ monocytes",
            "CD14++ CD16+ monocytes",
        }

        result = matcher.match_gate_sets(predicted, ground_truth)

        # These are harder to match - verify we don't crash
        assert isinstance(result, SemanticMatchResult)
