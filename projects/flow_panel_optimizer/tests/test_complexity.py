"""Tests for complexity index calculations."""

import numpy as np
import pytest

from flow_panel_optimizer.spectral.complexity import (
    complexity_index,
    complexity_index_v2,
    total_similarity_score,
    pair_complexity_contribution,
    identify_complexity_drivers,
    estimate_panel_quality,
    count_critical_pairs,
)


class TestComplexityIndex:
    """Tests for the complexity_index function."""

    def test_returns_total_similarity_for_similarity_matrix(self):
        """Similarity matrix should return total similarity score."""
        matrix = np.array([
            [1.0, 0.3, 0.4],
            [0.3, 1.0, 0.5],
            [0.4, 0.5, 1.0],
        ])
        # Total similarity = 0.3 + 0.4 + 0.5 = 1.2
        ci = complexity_index(matrix)
        assert ci == pytest.approx(1.2, abs=0.01)

    def test_higher_similarity_higher_score(self):
        """Panel with higher similarities should have higher score."""
        matrix_low = np.array([
            [1.0, 0.3, 0.4],
            [0.3, 1.0, 0.5],
            [0.4, 0.5, 1.0],
        ])
        matrix_high = np.array([
            [1.0, 0.95, 0.92],
            [0.95, 1.0, 0.91],
            [0.92, 0.91, 1.0],
        ])
        ci_low = complexity_index(matrix_low)
        ci_high = complexity_index(matrix_high)
        assert ci_high > ci_low

    def test_increases_with_more_pairs(self):
        """Total similarity should increase with more high-similarity pairs."""
        # Lower similarities
        matrix1 = np.array([
            [1.0, 0.50, 0.50],
            [0.50, 1.0, 0.50],
            [0.50, 0.50, 1.0],
        ])
        # Higher similarities
        matrix2 = np.array([
            [1.0, 0.95, 0.92],
            [0.95, 1.0, 0.50],
            [0.92, 0.50, 1.0],
        ])

        ci1 = complexity_index(matrix1)
        ci2 = complexity_index(matrix2)
        assert ci2 > ci1

    def test_single_fluorophore_returns_zero(self):
        """Single fluorophore panel should have CI=0."""
        matrix = np.array([[1.0]])
        ci = complexity_index(matrix)
        assert ci == 0.0

    def test_two_fluorophores(self):
        """Two fluorophores should return their similarity."""
        matrix = np.array([
            [1.0, 0.75],
            [0.75, 1.0],
        ])
        ci = complexity_index(matrix)
        assert ci == pytest.approx(0.75, abs=0.01)


class TestTotalSimilarityScore:
    """Tests for total_similarity_score function."""

    def test_sums_all_pairs(self):
        """Should sum all unique pairwise similarities."""
        matrix = np.array([
            [1.0, 0.3, 0.4],
            [0.3, 1.0, 0.5],
            [0.4, 0.5, 1.0],
        ])
        score = total_similarity_score(matrix)
        assert score == pytest.approx(1.2, abs=0.01)  # 0.3 + 0.4 + 0.5

    def test_empty_for_single_fluorophore(self):
        """Single fluorophore should return 0."""
        matrix = np.array([[1.0]])
        assert total_similarity_score(matrix) == 0.0


class TestComplexityIndexV2:
    """Tests for the complexity_index_v2 function."""

    def test_adds_autofluorescence(self):
        """Should add autofluorescence impact."""
        matrix = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
        ])

        ci_no_af = complexity_index_v2(matrix, autofluorescence_impact=0.0)
        ci_with_af = complexity_index_v2(matrix, autofluorescence_impact=10.0)

        assert ci_with_af == ci_no_af + 10.0

    def test_critical_penalty(self):
        """Should add penalty for critical pairs."""
        # Above critical threshold (0.90)
        matrix_critical = np.array([
            [1.0, 0.95],
            [0.95, 1.0],
        ])
        # Below critical threshold
        matrix_below = np.array([
            [1.0, 0.85],
            [0.85, 1.0],
        ])

        ci_critical = complexity_index_v2(matrix_critical, critical_threshold=0.90)
        ci_below = complexity_index_v2(matrix_below, critical_threshold=0.90)

        # Critical should have extra penalty (5 points per critical pair)
        assert ci_critical > ci_below + 4.0  # At least the penalty amount


class TestPairComplexityContribution:
    """Tests for pair complexity contribution."""

    def test_all_pairs_contribute(self):
        """All pairs should contribute their similarity value."""
        # Below critical threshold - just the similarity
        contrib = pair_complexity_contribution(0.50, critical_threshold=0.90)
        assert contrib == pytest.approx(0.50, abs=0.01)

    def test_critical_pairs_get_penalty(self):
        """Pairs above critical threshold get extra penalty."""
        contrib = pair_complexity_contribution(0.95, critical_threshold=0.90)
        # Should be similarity + 5.0 penalty
        assert contrib == pytest.approx(5.95, abs=0.01)

    def test_at_threshold_no_penalty(self):
        """Exactly at threshold should not get penalty."""
        contrib = pair_complexity_contribution(0.90, critical_threshold=0.90)
        assert contrib == pytest.approx(0.90, abs=0.01)


class TestIdentifyComplexityDrivers:
    """Tests for identifying complexity drivers."""

    def test_finds_all_pairs(self):
        """Should identify all pairs (all contribute to total similarity)."""
        matrix = np.array([
            [1.0, 0.95, 0.50],
            [0.95, 1.0, 0.92],
            [0.50, 0.92, 1.0],
        ])
        names = ["A", "B", "C"]

        drivers = identify_complexity_drivers(matrix, names, critical_threshold=0.90)

        assert len(drivers) == 3  # All 3 pairs
        # Should be sorted by contribution (highest first)
        assert drivers[0]["contribution"] >= drivers[1]["contribution"]

    def test_marks_critical_pairs(self):
        """Should mark pairs above threshold as critical."""
        matrix = np.array([
            [1.0, 0.95, 0.50],
            [0.95, 1.0, 0.50],
            [0.50, 0.50, 1.0],
        ])
        names = ["A", "B", "C"]

        drivers = identify_complexity_drivers(matrix, names, critical_threshold=0.90)

        # A-B should be critical
        critical_pairs = [d for d in drivers if d["is_critical"]]
        assert len(critical_pairs) == 1
        assert critical_pairs[0]["similarity"] == pytest.approx(0.95, abs=0.01)

    def test_respects_top_n(self):
        """Should respect top_n limit."""
        matrix = np.array([
            [1.0, 0.95, 0.94, 0.93],
            [0.95, 1.0, 0.92, 0.91],
            [0.94, 0.92, 1.0, 0.90],
            [0.93, 0.91, 0.90, 1.0],
        ])
        names = ["A", "B", "C", "D"]

        drivers = identify_complexity_drivers(matrix, names, critical_threshold=0.89, top_n=3)
        assert len(drivers) <= 3


class TestCountCriticalPairs:
    """Tests for counting critical pairs."""

    def test_counts_pairs_above_threshold(self):
        """Should count pairs with similarity above threshold."""
        matrix = np.array([
            [1.0, 0.95, 0.50],
            [0.95, 1.0, 0.92],
            [0.50, 0.92, 1.0],
        ])
        count = count_critical_pairs(matrix, threshold=0.90)
        assert count == 2  # A-B and B-C

    def test_zero_for_low_similarity(self):
        """Should return 0 if no pairs above threshold."""
        matrix = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
        ])
        count = count_critical_pairs(matrix, threshold=0.90)
        assert count == 0


class TestEstimatePanelQuality:
    """Tests for panel quality estimation."""

    def test_excellent_quality(self):
        """Low average similarity should be excellent."""
        # 10 fluorophores = 45 pairs
        # avg similarity 0.20 -> complexity = 45 * 0.20 = 9.0
        assert estimate_panel_quality(9.0, 10) == "excellent"

    def test_good_quality(self):
        """Moderate average similarity should be good."""
        # avg similarity 0.30 -> complexity = 45 * 0.30 = 13.5
        assert estimate_panel_quality(13.5, 10) == "good"

    def test_acceptable_quality(self):
        """Higher average similarity should be acceptable."""
        # avg similarity 0.40 -> complexity = 45 * 0.40 = 18.0
        assert estimate_panel_quality(18.0, 10) == "acceptable"

    def test_poor_quality(self):
        """Very high average similarity should be poor."""
        # avg similarity 0.60 -> complexity = 45 * 0.60 = 27.0
        assert estimate_panel_quality(27.0, 10) == "poor"

    def test_scales_with_panel_size(self):
        """Quality thresholds should scale with panel size."""
        # For a 4-fluorophore panel (6 pairs), avg similarity = 0.30
        # complexity = 6 * 0.30 = 1.8 -> should be good
        assert estimate_panel_quality(1.8, 4) == "good"

        # For a 10-fluorophore panel (45 pairs), same avg similarity
        # complexity = 45 * 0.30 = 13.5 -> should also be good
        assert estimate_panel_quality(13.5, 10) == "good"
