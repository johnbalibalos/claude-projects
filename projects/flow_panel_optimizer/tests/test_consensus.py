"""Tests for consensus checking and validation."""

import numpy as np
import pytest

from flow_panel_optimizer.validation.consensus import (
    RiskLevel,
    ConsensusResult,
    check_consensus,
    validate_panel_consensus,
    summarize_consensus,
)


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_ordering(self):
        """Risk levels should be orderable."""
        assert RiskLevel.MINIMAL < RiskLevel.LOW
        assert RiskLevel.LOW < RiskLevel.MODERATE
        assert RiskLevel.MODERATE < RiskLevel.HIGH
        assert RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_equality(self):
        """Same risk levels should be equal."""
        assert RiskLevel.HIGH == RiskLevel.HIGH
        assert not (RiskLevel.HIGH == RiskLevel.LOW)


class TestCheckConsensus:
    """Tests for the check_consensus function."""

    def test_all_metrics_agree_low_risk(self):
        """Low similarity and spread should give LOW risk consensus."""
        result = check_consensus(
            name_a="A",
            name_b="B",
            cosine_sim=0.50,
            complexity_contrib=0.0,
            spread_value=1.0,
        )

        assert result.consensus_risk == RiskLevel.MINIMAL
        assert result.metrics_agree

    def test_all_metrics_agree_high_risk(self):
        """High similarity and spread should give HIGH+ risk consensus."""
        result = check_consensus(
            name_a="A",
            name_b="B",
            cosine_sim=0.98,
            complexity_contrib=0.08,
            spread_value=15.0,
        )

        assert result.consensus_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_metrics_disagree(self):
        """Disagreeing metrics should be flagged."""
        result = check_consensus(
            name_a="A",
            name_b="B",
            cosine_sim=0.99,  # Critical
            complexity_contrib=0.09,
            spread_value=5.0,  # Moderate
        )

        assert not result.metrics_agree
        # Consensus should be highest risk
        assert result.consensus_risk == RiskLevel.CRITICAL

    def test_custom_thresholds(self):
        """Should respect custom thresholds."""
        thresholds = {
            "cosine_critical": 0.90,  # Lower than default
            "cosine_high": 0.85,
        }

        result = check_consensus(
            name_a="A",
            name_b="B",
            cosine_sim=0.92,  # Would be moderate with defaults
            complexity_contrib=0.02,
            spread_value=5.0,
            thresholds=thresholds,
        )

        # With custom thresholds, 0.92 should be critical
        assert result.similarity_risk == RiskLevel.CRITICAL

    def test_notes_added_for_high_similarity(self):
        """Should add notes for very high similarity."""
        result = check_consensus(
            name_a="A",
            name_b="B",
            cosine_sim=0.97,
            complexity_contrib=0.07,
            spread_value=12.0,
        )

        assert "high spectral similarity" in result.notes.lower()

    def test_result_contains_all_values(self):
        """Result should contain all input values."""
        result = check_consensus(
            name_a="FITC",
            name_b="BB515",
            cosine_sim=0.98,
            complexity_contrib=0.08,
            spread_value=10.0,
        )

        assert result.fluorophore_a == "FITC"
        assert result.fluorophore_b == "BB515"
        assert result.cosine_similarity == pytest.approx(0.98)
        assert result.complexity_contribution == pytest.approx(0.08)
        assert result.theoretical_spread == pytest.approx(10.0)


class TestValidatePanelConsensus:
    """Tests for panel-wide consensus validation."""

    @pytest.fixture
    def low_similarity_panel(self):
        """Panel with all low similarities and correspondingly low spreading."""
        names = ["A", "B", "C"]
        sim_matrix = np.array([
            [1.0, 0.3, 0.4],
            [0.3, 1.0, 0.5],
            [0.4, 0.5, 1.0],
        ])
        # Use spread values < 2.0 to match "minimal" risk level of low similarities
        spread_matrix = np.array([
            [0.0, 0.5, 0.8],
            [0.5, 0.0, 1.2],
            [0.8, 1.2, 0.0],
        ])
        return names, sim_matrix, spread_matrix

    @pytest.fixture
    def high_similarity_panel(self):
        """Panel with high similarities."""
        names = ["A", "B", "C"]
        sim_matrix = np.array([
            [1.0, 0.98, 0.95],
            [0.98, 1.0, 0.92],
            [0.95, 0.92, 1.0],
        ])
        spread_matrix = np.array([
            [0.0, 15.0, 10.0],
            [15.0, 0.0, 8.0],
            [10.0, 8.0, 0.0],
        ])
        return names, sim_matrix, spread_matrix

    def test_counts_total_pairs(self, low_similarity_panel):
        """Should count correct number of pairs."""
        names, sim_matrix, spread_matrix = low_similarity_panel

        result = validate_panel_consensus(
            "Test Panel", sim_matrix, spread_matrix, names
        )

        # N*(N-1)/2 = 3*2/2 = 3 pairs
        assert result["total_pairs"] == 3

    def test_agreement_rate_for_low_similarity(self, low_similarity_panel):
        """Low similarity panel should have high agreement."""
        names, sim_matrix, spread_matrix = low_similarity_panel

        result = validate_panel_consensus(
            "Test Panel", sim_matrix, spread_matrix, names
        )

        # All pairs should be minimal/low risk with agreement
        assert result["agreement_rate"] >= 0.5

    def test_identifies_high_risk_pairs(self, high_similarity_panel):
        """Should identify high risk pairs."""
        names, sim_matrix, spread_matrix = high_similarity_panel

        result = validate_panel_consensus(
            "Test Panel", sim_matrix, spread_matrix, names
        )

        assert len(result["high_risk_pairs"]) > 0

    def test_counts_critical_pairs(self, high_similarity_panel):
        """Should count critical pairs."""
        names, sim_matrix, spread_matrix = high_similarity_panel

        result = validate_panel_consensus(
            "Test Panel", sim_matrix, spread_matrix, names
        )

        # A-B with 0.98 similarity should be critical
        assert result["critical_pairs"] >= 1

    def test_risk_distribution_sums_to_total(self, low_similarity_panel):
        """Risk distribution should sum to total pairs."""
        names, sim_matrix, spread_matrix = low_similarity_panel

        result = validate_panel_consensus(
            "Test Panel", sim_matrix, spread_matrix, names
        )

        total = sum(result["by_risk_level"].values())
        assert total == result["total_pairs"]


class TestSummarizeConsensus:
    """Tests for consensus summary generation."""

    def test_includes_panel_name(self):
        """Summary should include panel name."""
        result = {
            "panel_name": "My Panel",
            "total_pairs": 3,
            "metrics_agree": 2,
            "agreement_rate": 0.667,
            "by_risk_level": {
                "minimal": 1,
                "low": 1,
                "moderate": 0,
                "high": 1,
                "critical": 0,
            },
            "high_risk_pairs": [
                {"fluor_a": "A", "fluor_b": "B", "risk": "high", "similarity": 0.95, "notes": "test"}
            ],
            "critical_pairs": 0,
        }

        summary = summarize_consensus(result)

        assert "My Panel" in summary

    def test_includes_agreement_rate(self):
        """Summary should include agreement rate."""
        result = {
            "panel_name": "Test",
            "total_pairs": 10,
            "metrics_agree": 8,
            "agreement_rate": 0.80,
            "by_risk_level": {
                "minimal": 5,
                "low": 3,
                "moderate": 1,
                "high": 1,
                "critical": 0,
            },
            "high_risk_pairs": [],
            "critical_pairs": 0,
        }

        summary = summarize_consensus(result)

        assert "80" in summary  # Should show 80% agreement

    def test_lists_high_risk_pairs(self):
        """Summary should list high risk pairs."""
        result = {
            "panel_name": "Test",
            "total_pairs": 3,
            "metrics_agree": 2,
            "agreement_rate": 0.667,
            "by_risk_level": {
                "minimal": 0,
                "low": 0,
                "moderate": 1,
                "high": 1,
                "critical": 1,
            },
            "high_risk_pairs": [
                {"fluor_a": "FITC", "fluor_b": "BB515", "risk": "critical", "similarity": 0.98, "notes": ""},
                {"fluor_a": "APC", "fluor_b": "Alexa647", "risk": "high", "similarity": 0.95, "notes": ""},
            ],
            "critical_pairs": 1,
        }

        summary = summarize_consensus(result)

        assert "FITC" in summary
        assert "BB515" in summary
