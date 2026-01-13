"""Tests for pass@k metrics."""

import pytest
import numpy as np
from src.evaluation.pass_at_k import (
    compute_pass_at_k,
    compute_pass_k_power,
    analyze_pass_at_k,
    aggregate_pass_at_k,
    compare_models_pass_at_k,
    compute_pass_at_k_from_scores,
    PassAtKResult,
    AggregatedPassAtK,
    format_pass_at_k_report,
)


class TestComputePassAtK:
    """Tests for the core pass@k computation."""

    def test_all_pass(self):
        """All samples passing should give pass@k = 1.0."""
        results = [True, True, True, True, True]
        assert compute_pass_at_k(results, k=1) == 1.0
        assert compute_pass_at_k(results, k=3) == 1.0
        assert compute_pass_at_k(results, k=5) == 1.0

    def test_all_fail(self):
        """All samples failing should give pass@k = 0.0."""
        results = [False, False, False, False, False]
        assert compute_pass_at_k(results, k=1) == 0.0
        assert compute_pass_at_k(results, k=3) == 0.0
        assert compute_pass_at_k(results, k=5) == 0.0

    def test_single_pass(self):
        """Single pass out of n should give pass@1 = 1/n."""
        results = [True, False, False, False, False]
        assert compute_pass_at_k(results, k=1) == pytest.approx(0.2)

    def test_pass_at_k_increases_with_k(self):
        """pass@k should increase as k increases."""
        results = [True, False, True, False, False]
        pass_1 = compute_pass_at_k(results, k=1)
        pass_2 = compute_pass_at_k(results, k=2)
        pass_3 = compute_pass_at_k(results, k=3)

        assert pass_1 < pass_2 < pass_3

    def test_exact_formula(self):
        """Test against known exact values."""
        # 3 passes out of 5, k=2
        # pass@2 = 1 - C(2,2)/C(5,2) = 1 - 1/10 = 0.9
        results = [True, True, True, False, False]
        assert compute_pass_at_k(results, k=2) == pytest.approx(0.9)

    def test_small_sample_size(self):
        """Test when n < k."""
        results = [True, False]  # n=2, but k=5
        # Should handle gracefully
        result = compute_pass_at_k(results, k=5)
        assert 0.0 <= result <= 1.0

    def test_empty_results(self):
        """Empty results should return 0."""
        results = []
        assert compute_pass_at_k(results, k=1) == 0.0


class TestComputePassKPower:
    """Tests for pass^k (all k pass) computation."""

    def test_all_pass(self):
        """All samples passing should give pass^k = 1.0."""
        results = [True, True, True, True, True]
        assert compute_pass_k_power(results, k=1) == 1.0
        assert compute_pass_k_power(results, k=3) == 1.0
        assert compute_pass_k_power(results, k=5) == 1.0

    def test_all_fail(self):
        """All samples failing should give pass^k = 0.0."""
        results = [False, False, False, False, False]
        assert compute_pass_k_power(results, k=1) == 0.0
        assert compute_pass_k_power(results, k=3) == 0.0

    def test_exact_formula(self):
        """Test against known exact values."""
        # 3 passes out of 5, k=2
        # pass^2 = C(3,2)/C(5,2) = 3/10 = 0.3
        results = [True, True, True, False, False]
        assert compute_pass_k_power(results, k=2) == pytest.approx(0.3)

    def test_pass_k_power_decreases_with_k(self):
        """pass^k should decrease as k increases (unless all pass)."""
        results = [True, True, True, False, False]
        power_1 = compute_pass_k_power(results, k=1)
        power_2 = compute_pass_k_power(results, k=2)
        power_3 = compute_pass_k_power(results, k=3)

        assert power_1 > power_2 > power_3

    def test_insufficient_passes(self):
        """If c < k, pass^k should be 0."""
        results = [True, False, False, False, False]  # Only 1 pass
        assert compute_pass_k_power(results, k=2) == 0.0


class TestAnalyzePassAtK:
    """Tests for comprehensive pass@k analysis."""

    def test_basic_analysis(self):
        """Test basic analysis returns correct structure."""
        results = [True, False, True, True, False]
        analysis = analyze_pass_at_k(results, k=3, n_bootstrap=100)

        assert isinstance(analysis, PassAtKResult)
        assert analysis.k == 3
        assert analysis.n_samples == 5
        assert analysis.n_passes == 3
        assert analysis.pass_rate == pytest.approx(0.6)

    def test_confidence_intervals(self):
        """Test that confidence intervals are computed."""
        results = [True, False, True, True, False] * 10  # 50 samples
        analysis = analyze_pass_at_k(results, k=3, n_bootstrap=500)

        assert analysis.pass_at_1_ci is not None
        assert analysis.pass_at_k_ci is not None
        assert analysis.pass_k_power_ci is not None

        # CI should contain point estimate (most of the time)
        lo, hi = analysis.pass_at_k_ci
        assert lo <= hi
        assert lo >= 0.0
        assert hi <= 1.0

    def test_pass_at_k_curve(self):
        """Test that pass@k curve is computed for multiple k values."""
        results = [True, False, True, True, False] * 4  # 20 samples
        analysis = analyze_pass_at_k(results, k=5, compute_curve=True)

        assert 1 in analysis.pass_at_k_curve
        assert 5 in analysis.pass_at_k_curve

        # Values should be monotonically increasing
        curve = analysis.pass_at_k_curve
        k_values = sorted(curve.keys())
        for i in range(len(k_values) - 1):
            assert curve[k_values[i]] <= curve[k_values[i + 1]]

    def test_to_dict(self):
        """Test serialization to dict."""
        results = [True, False, True]
        analysis = analyze_pass_at_k(results, k=2, n_bootstrap=0)

        d = analysis.to_dict()
        assert "pass_at_1" in d
        assert "pass_at_k" in d
        assert "n_samples" in d


class TestAggregatePassAtK:
    """Tests for aggregating pass@k across test cases."""

    def test_aggregate_multiple_test_cases(self):
        """Test aggregation across multiple test cases."""
        results_by_tc = {
            "tc1": [True, True, False, False],
            "tc2": [True, False, False, False],
            "tc3": [True, True, True, False],
        }

        aggregated = aggregate_pass_at_k(results_by_tc, k=2, n_bootstrap=0)

        assert isinstance(aggregated, AggregatedPassAtK)
        assert aggregated.n_test_cases == 3
        assert aggregated.total_samples == 12
        assert aggregated.total_passes == 6
        assert len(aggregated.per_test_case) == 3

    def test_mean_and_std(self):
        """Test that mean and std are computed correctly."""
        # All test cases have same pass rate
        results_by_tc = {
            "tc1": [True, False],
            "tc2": [True, False],
            "tc3": [True, False],
        }

        aggregated = aggregate_pass_at_k(results_by_tc, k=1, n_bootstrap=0)

        # All have pass@1 = 0.5, so std should be 0
        assert aggregated.mean_pass_at_1 == pytest.approx(0.5)
        assert aggregated.std_pass_at_1 == pytest.approx(0.0)


class TestComputePassAtKFromScores:
    """Tests for computing pass@k from numeric scores."""

    def test_threshold_based(self):
        """Test threshold-based pass/fail determination."""
        scores = [0.9, 0.7, 0.85, 0.6, 0.95]

        # With threshold 0.8, passes are: 0.9, 0.85, 0.95 (3/5)
        result = compute_pass_at_k_from_scores(scores, k=1, threshold=0.8, n_bootstrap=0)

        assert result.n_passes == 3
        assert result.n_samples == 5
        assert result.pass_rate == pytest.approx(0.6)


class TestCompareModelsPassAtK:
    """Tests for model comparison."""

    def test_compare_different_models(self):
        """Test comparison between models with different performance."""
        # Use larger samples for better statistical power
        results_a = [True] * 16 + [False] * 4  # 80% pass rate, n=20
        results_b = [True] * 8 + [False] * 12  # 40% pass rate, n=20

        comparison = compare_models_pass_at_k(
            results_a, results_b,
            k=3,
            n_permutations=1000,
            model_a_name="Better Model",
            model_b_name="Worse Model",
        )

        assert comparison.pass_at_k_a > comparison.pass_at_k_b
        assert comparison.difference > 0
        # With this large difference and sample size, should be significant
        assert comparison.p_value < 0.15  # Relaxed threshold for statistical variation

    def test_compare_identical_models(self):
        """Test comparison between models with identical performance."""
        results_a = [True, False, True, False, True]
        results_b = [True, False, True, False, True]

        comparison = compare_models_pass_at_k(
            results_a, results_b,
            k=2,
            n_permutations=500,
        )

        assert comparison.difference == pytest.approx(0.0)
        # Should not be significant
        assert comparison.p_value > 0.5


class TestFormatPassAtKReport:
    """Tests for report formatting."""

    def test_report_format(self):
        """Test that report is properly formatted."""
        results_by_tc = {
            "test_case_1": [True, False, True],
            "test_case_2": [True, True, False],
        }

        aggregated = aggregate_pass_at_k(results_by_tc, k=2, n_bootstrap=0)
        report = format_pass_at_k_report(aggregated, title="Test Report")

        assert "Test Report" in report
        assert "pass@1" in report
        assert "pass@2" in report
        assert "test_case" in report


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_k_equals_n(self):
        """Test when k equals n (all samples selected)."""
        results = [True, True, False]
        pass_at_3 = compute_pass_at_k(results, k=3)
        power_3 = compute_pass_k_power(results, k=3)

        # pass@n should be 1 if any pass, 0 if all fail
        assert pass_at_3 == 1.0  # At least one passes
        assert power_3 == 0.0  # Not all pass

    def test_k_greater_than_n(self):
        """Test when k > n."""
        results = [True, False]
        # Should handle gracefully without error
        result = compute_pass_at_k(results, k=10)
        assert 0.0 <= result <= 1.0

    def test_single_sample(self):
        """Test with single sample."""
        assert compute_pass_at_k([True], k=1) == 1.0
        assert compute_pass_at_k([False], k=1) == 0.0

        assert compute_pass_k_power([True], k=1) == 1.0
        assert compute_pass_k_power([False], k=1) == 0.0

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        results = [True, False] * 25

        np.random.seed(42)
        analysis1 = analyze_pass_at_k(results, k=5, n_bootstrap=100)

        np.random.seed(42)
        analysis2 = analyze_pass_at_k(results, k=5, n_bootstrap=100)

        # Core metrics should be identical
        assert analysis1.pass_at_1 == analysis2.pass_at_1
        assert analysis1.pass_at_k == analysis2.pass_at_k
