"""Tests for spillover spreading matrix calculations."""

import numpy as np
import pytest

from flow_panel_optimizer.spectral.spreading import (
    theoretical_spreading,
    build_spreading_matrix,
    spreading_risk_level,
    find_high_spreading_pairs,
    estimate_spillover_from_similarity,
    total_panel_spreading,
)


class TestEstimateSpilloverFromSimilarity:
    """Tests for spillover estimation from similarity."""

    def test_zero_similarity_zero_spillover(self):
        """Zero similarity should give zero spillover."""
        assert estimate_spillover_from_similarity(0.0) == 0.0

    def test_perfect_similarity_high_spillover(self):
        """Perfect similarity should give high spillover."""
        spillover = estimate_spillover_from_similarity(1.0)
        assert spillover == 1.0

    def test_increases_with_similarity(self):
        """Spillover should increase with similarity."""
        low = estimate_spillover_from_similarity(0.5)
        high = estimate_spillover_from_similarity(0.9)
        assert high > low

    def test_power_parameter(self):
        """Power parameter should affect the curve."""
        linear = estimate_spillover_from_similarity(0.5, power=1.0)
        quadratic = estimate_spillover_from_similarity(0.5, power=2.0)
        assert linear > quadratic  # Higher power = more conservative


class TestTheoreticalSpreading:
    """Tests for theoretical spreading calculation."""

    def test_zero_spillover_zero_spread(self):
        """Zero spillover should give zero spread."""
        spread = theoretical_spreading(
            similarity=0.5,
            spillover_coefficient=0.0,
            stain_index_primary=100.0,
        )
        assert spread == 0.0

    def test_increases_with_stain_index(self):
        """Spread should increase with brighter signals."""
        low_si = theoretical_spreading(
            similarity=0.9,
            spillover_coefficient=0.5,
            stain_index_primary=50.0,
        )
        high_si = theoretical_spreading(
            similarity=0.9,
            spillover_coefficient=0.5,
            stain_index_primary=200.0,
        )
        assert high_si > low_si

    def test_increases_with_spillover(self):
        """Spread should increase with spillover coefficient."""
        low_spill = theoretical_spreading(
            similarity=0.9,
            spillover_coefficient=0.1,
            stain_index_primary=100.0,
        )
        high_spill = theoretical_spreading(
            similarity=0.9,
            spillover_coefficient=0.5,
            stain_index_primary=100.0,
        )
        assert high_spill > low_spill


class TestBuildSpreadingMatrix:
    """Tests for building spreading matrices."""

    def test_matrix_shape(self):
        """Matrix should be NxN."""
        sim_matrix = np.array([
            [1.0, 0.8, 0.5],
            [0.8, 1.0, 0.6],
            [0.5, 0.6, 1.0],
        ])
        ssm = build_spreading_matrix(sim_matrix)
        assert ssm.shape == (3, 3)

    def test_diagonal_is_zero(self):
        """Diagonal should be zero (no self-spreading)."""
        sim_matrix = np.array([
            [1.0, 0.8],
            [0.8, 1.0],
        ])
        ssm = build_spreading_matrix(sim_matrix)
        assert ssm[0, 0] == 0.0
        assert ssm[1, 1] == 0.0

    def test_uses_stain_indices(self):
        """Should use provided stain indices."""
        sim_matrix = np.array([
            [1.0, 0.9],
            [0.9, 1.0],
        ])

        # Different stain indices should give different results
        ssm_low = build_spreading_matrix(sim_matrix, stain_indices=[50, 50])
        ssm_high = build_spreading_matrix(sim_matrix, stain_indices=[200, 200])

        assert ssm_high[0, 1] > ssm_low[0, 1]

    def test_invalid_stain_indices_raises_error(self):
        """Should raise error if stain indices length doesn't match."""
        sim_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="must match"):
            build_spreading_matrix(sim_matrix, stain_indices=[100])


class TestSpreadingRiskLevel:
    """Tests for spreading risk level classification."""

    def test_critical_threshold(self):
        """High spread should be critical."""
        assert spreading_risk_level(25.0) == "critical"
        assert spreading_risk_level(20.0) == "critical"

    def test_high_threshold(self):
        """Moderate-high spread should be high."""
        assert spreading_risk_level(15.0) == "high"
        assert spreading_risk_level(10.0) == "high"

    def test_moderate_threshold(self):
        """Moderate spread should be moderate."""
        assert spreading_risk_level(7.0) == "moderate"
        assert spreading_risk_level(5.0) == "moderate"

    def test_low_threshold(self):
        """Low spread should be low."""
        assert spreading_risk_level(3.0) == "low"
        assert spreading_risk_level(2.0) == "low"

    def test_minimal_threshold(self):
        """Very low spread should be minimal."""
        assert spreading_risk_level(1.0) == "minimal"
        assert spreading_risk_level(0.0) == "minimal"


class TestFindHighSpreadingPairs:
    """Tests for finding high spreading pairs."""

    def test_finds_high_pairs(self):
        """Should find pairs above threshold."""
        ssm = np.array([
            [0.0, 15.0, 3.0],
            [14.0, 0.0, 5.0],
            [2.0, 4.0, 0.0],
        ])
        names = ["A", "B", "C"]

        pairs = find_high_spreading_pairs(ssm, names, threshold=10.0)

        # A->B (15.0) and B->A (14.0) should be found
        assert len(pairs) == 2
        assert any(p[0] == "A" and p[1] == "B" for p in pairs)

    def test_sorted_descending(self):
        """Results should be sorted by spread descending."""
        ssm = np.array([
            [0.0, 15.0, 12.0],
            [14.0, 0.0, 11.0],
            [10.0, 8.0, 0.0],
        ])
        names = ["A", "B", "C"]

        pairs = find_high_spreading_pairs(ssm, names, threshold=10.0)

        spreads = [s for _, _, s in pairs]
        assert spreads == sorted(spreads, reverse=True)

    def test_includes_both_directions(self):
        """Should include asymmetric spreading in both directions."""
        ssm = np.array([
            [0.0, 20.0],
            [5.0, 0.0],  # Asymmetric
        ])
        names = ["A", "B"]

        pairs = find_high_spreading_pairs(ssm, names, threshold=10.0)

        # Only A->B should be above threshold
        assert len(pairs) == 1
        assert pairs[0] == ("A", "B", 20.0)


class TestTotalPanelSpreading:
    """Tests for total panel spreading calculation."""

    def test_sums_all_off_diagonal(self):
        """Should sum all non-diagonal elements."""
        ssm = np.array([
            [0.0, 5.0, 3.0],
            [4.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ])
        total = total_panel_spreading(ssm)
        assert total == pytest.approx(5.0 + 3.0 + 4.0 + 2.0 + 3.0 + 2.0)

    def test_zero_for_zero_matrix(self):
        """Should return 0 for zero matrix."""
        ssm = np.zeros((3, 3))
        assert total_panel_spreading(ssm) == 0.0
