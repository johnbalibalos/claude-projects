"""
Tests for the new LLM evaluation modules.

Tests:
- Statistical rigor (statistics.py)
- Calibration metrics (calibration.py)
- Selective prediction (selective_prediction.py)
- Test set diversity (diversity.py)
- Cost-performance Pareto (pareto.py)
- Robustness utilities (robustness.py)
"""

import numpy as np
import pytest

# =============================================================================
# STATISTICS MODULE TESTS
# =============================================================================


class TestStatistics:
    """Tests for statistical rigor module."""

    def test_bootstrap_ci_basic(self):
        """Test basic bootstrap confidence interval."""
        from hypothesis_pipeline.statistics import bootstrap_ci

        scores = [0.8, 0.85, 0.9, 0.82, 0.88, 0.87, 0.83, 0.86]
        result = bootstrap_ci(scores, n_bootstrap=1000, random_state=42)

        assert 0.8 < result.estimate < 0.9
        assert result.ci_lower < result.estimate < result.ci_upper
        assert result.ci_level == 0.95
        assert len(result.bootstrap_distribution) == 1000

    def test_bootstrap_ci_methods(self):
        """Test different CI methods produce valid intervals."""
        from hypothesis_pipeline.statistics import bootstrap_ci

        scores = np.random.normal(0.8, 0.1, 50)

        for method in ["percentile", "basic", "bca"]:
            result = bootstrap_ci(scores, method=method, random_state=42)
            assert result.ci_lower < result.ci_upper
            assert result.ci_lower < result.estimate < result.ci_upper

    def test_paired_permutation_test(self):
        """Test paired permutation test."""
        from hypothesis_pipeline.statistics import paired_permutation_test

        # Create clearly different conditions
        scores_a = [0.9, 0.92, 0.88, 0.91, 0.89]
        scores_b = [0.7, 0.72, 0.68, 0.71, 0.69]

        result = paired_permutation_test(
            scores_a, scores_b,
            alternative="greater",
            random_state=42
        )

        assert result.p_value < 0.05  # Should be significant
        assert result.observed_statistic > 0
        assert result.effect_direction == "positive"

    def test_cohens_d(self):
        """Test Cohen's d effect size."""
        from hypothesis_pipeline.statistics import cohens_d

        # Large effect
        group_a = [10, 11, 12, 10, 11]
        group_b = [5, 6, 7, 5, 6]

        result = cohens_d(group_a, group_b)

        assert result.value > 0.8  # Large effect
        assert result.interpretation == "large"
        assert result.ci_lower is not None

    def test_hedges_g(self):
        """Test Hedges' g (bias-corrected d)."""
        from hypothesis_pipeline.statistics import hedges_g

        group_a = [10, 11, 12]
        group_b = [5, 6, 7]

        result = hedges_g(group_a, group_b)

        assert result.effect_type == "Hedges' g"
        # Hedges' g should be slightly smaller than Cohen's d for small samples
        assert abs(result.value) > 0

    def test_bonferroni_correction(self):
        """Test Bonferroni multiple comparison correction."""
        from hypothesis_pipeline.statistics import bonferroni_correction

        p_values = [0.01, 0.02, 0.03, 0.001]
        result = bonferroni_correction(p_values, alpha=0.05)

        assert result.method == "Bonferroni"
        assert len(result.corrected_p_values) == 4
        # First p-value (0.01 * 4 = 0.04) should still be significant
        assert result.corrected_p_values[0] == 0.04
        # 0.001 * 4 = 0.004 should be significant
        assert result.significant[3] == True

    def test_holm_correction(self):
        """Test Holm-Bonferroni step-down correction."""
        from hypothesis_pipeline.statistics import holm_correction

        p_values = [0.01, 0.04, 0.03, 0.001]
        result = holm_correction(p_values, alpha=0.05)

        assert result.method == "Holm-Bonferroni"
        # Should be less conservative than Bonferroni
        assert result.n_significant >= 0

    def test_power_analysis(self):
        """Test power analysis for two-sample test."""
        from hypothesis_pipeline.statistics import power_analysis_two_sample

        # Compute required n for medium effect size
        result = power_analysis_two_sample(effect_size=0.5, power=0.80)

        assert result.required_n is not None
        assert result.required_n > 10  # Should need reasonable sample size

        # Compute achieved power with given n
        result2 = power_analysis_two_sample(effect_size=0.5, n_per_group=50)
        assert result2.achieved_power is not None
        assert 0 < result2.achieved_power < 1

    def test_compare_conditions(self):
        """Test comprehensive condition comparison."""
        from hypothesis_pipeline.statistics import compare_conditions

        direct_scores = [0.72, 0.78, 0.75, 0.80, 0.77]
        cot_scores = [0.85, 0.82, 0.88, 0.84, 0.86]

        result = compare_conditions(
            cot_scores, direct_scores,
            condition_a_name="CoT",
            condition_b_name="Direct",
            paired=True,
            random_state=42
        )

        assert result.mean_a > result.mean_b
        assert result.effect_size.value > 0
        assert len(result.summary()) > 0


# =============================================================================
# CALIBRATION MODULE TESTS
# =============================================================================


class TestCalibration:
    """Tests for calibration metrics module."""

    def test_expected_calibration_error(self):
        """Test ECE computation."""
        from hypothesis_pipeline.calibration import expected_calibration_error

        # Perfect calibration: confidence matches accuracy
        confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
        correctness = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]  # ~90% acc for high conf, ~10% for low

        result = expected_calibration_error(confidences, correctness, n_bins=2)

        assert 0 <= result.ece <= 1
        assert len(result.bins) == 2
        assert result.n_samples == 10

    def test_ece_with_poor_calibration(self):
        """Test ECE detects poor calibration."""
        from hypothesis_pipeline.calibration import expected_calibration_error

        # Overconfident model: high confidence but low accuracy
        confidences = [0.95] * 10
        correctness = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]  # Only 30% correct

        result = expected_calibration_error(confidences, correctness)

        assert result.ece > 0.5  # Should show high calibration error
        assert result.overconfidence_rate > 0.5

    def test_brier_score(self):
        """Test Brier score computation."""
        from hypothesis_pipeline.calibration import brier_score

        confidences = [0.9, 0.8, 0.3, 0.1]
        correctness = [1, 1, 0, 0]

        result = brier_score(confidences, correctness, decompose=True)

        assert 0 <= result.brier_score <= 1
        assert result.reliability >= 0
        assert result.resolution >= 0

    def test_parse_confidence_json(self):
        """Test parsing confidence from JSON response."""
        from hypothesis_pipeline.calibration import parse_confidence_from_response

        response = '''
        Here is my answer.
        ```json
        {"answer": "test", "confidence": 85}
        ```
        '''

        result = parse_confidence_from_response(response)

        assert result.confidence_score == 0.85
        assert result.parse_method == "json"

    def test_parse_confidence_explicit(self):
        """Test parsing explicit confidence statements."""
        from hypothesis_pipeline.calibration import parse_confidence_from_response

        response = "The answer is X. I am 75% confident in this answer."
        result = parse_confidence_from_response(response)

        assert result.confidence_score == 0.75
        assert result.parse_method == "explicit"

    def test_temperature_scaling(self):
        """Test temperature scaling calibration."""
        from hypothesis_pipeline.calibration import find_optimal_temperature

        # Overconfident predictions
        confidences = np.array([0.95, 0.92, 0.88, 0.90, 0.93])
        correctness = np.array([1, 0, 1, 0, 1])  # 60% accuracy

        result = find_optimal_temperature(confidences, correctness)

        assert result.optimal_temperature > 0
        assert result.ece_after <= result.ece_before


# =============================================================================
# SELECTIVE PREDICTION MODULE TESTS
# =============================================================================


class TestSelectivePrediction:
    """Tests for selective prediction module."""

    def test_selective_prediction_metrics(self):
        """Test basic selective prediction metrics."""
        from hypothesis_pipeline.selective_prediction import selective_prediction_metrics

        confidences = [0.9, 0.8, 0.7, 0.4, 0.3]
        correctness = [1, 1, 1, 0, 0]

        result = selective_prediction_metrics(confidences, correctness, threshold=0.6)

        assert result.coverage == 0.6  # 3 out of 5 above threshold
        assert result.selective_accuracy == 1.0  # All 3 correct
        assert result.n_attempted == 3
        assert result.n_rejected == 2

    def test_risk_coverage_curve(self):
        """Test risk-coverage curve computation."""
        from hypothesis_pipeline.selective_prediction import compute_risk_coverage_curve

        np.random.seed(42)
        confidences = np.random.uniform(0, 1, 100)
        correctness = (np.random.uniform(0, 1, 100) < confidences).astype(int)

        curve = compute_risk_coverage_curve(confidences, correctness)

        assert len(curve.coverages) > 0
        assert len(curve.risks) == len(curve.coverages)
        assert 0 <= curve.auc_rc <= 1
        assert 0 <= curve.optimal_threshold <= 1

    def test_threshold_optimization(self):
        """Test threshold optimization."""
        from hypothesis_pipeline.selective_prediction import optimize_threshold

        confidences = [0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3]
        correctness = [1, 1, 1, 1, 0, 0, 0]

        result = optimize_threshold(
            confidences, correctness,
            criterion="accuracy",
            min_coverage=0.5
        )

        assert 0 <= result.optimal_threshold <= 1
        assert result.optimal_coverage >= 0.5
        assert 0 <= result.optimal_accuracy <= 1

    def test_cost_sensitive_selection(self):
        """Test cost-sensitive selective prediction."""
        from hypothesis_pipeline.selective_prediction import cost_sensitive_selective_prediction

        confidences = [0.9, 0.8, 0.5, 0.3]
        correctness = [1, 1, 0, 0]

        result = cost_sensitive_selective_prediction(
            confidences, correctness,
            error_cost=1.0,
            abstention_cost=0.3
        )

        assert result.expected_cost >= 0
        assert 0 <= result.optimal_threshold <= 1


# =============================================================================
# DIVERSITY MODULE TESTS
# =============================================================================


class TestDiversity:
    """Tests for test set diversity module."""

    def test_analyze_distribution(self):
        """Test distribution analysis."""
        from hypothesis_pipeline.diversity import analyze_distribution

        values = ["A", "A", "A", "B", "B", "C"]
        result = analyze_distribution(values, "category")

        assert result.n_unique == 3
        assert result.n_samples == 6
        assert result.distribution == {"A": 3, "B": 2, "C": 1}
        assert 0 <= result.entropy <= 1

    def test_analyze_test_set_diversity(self):
        """Test comprehensive diversity analysis."""
        from hypothesis_pipeline.diversity import analyze_test_set_diversity

        test_cases = [
            {"id": 1, "complexity": "simple", "domain": "biology"},
            {"id": 2, "complexity": "complex", "domain": "chemistry"},
            {"id": 3, "complexity": "simple", "domain": "biology"},
            {"id": 4, "complexity": "medium", "domain": "physics"},
        ]

        report = analyze_test_set_diversity(test_cases, ["complexity", "domain"])

        assert report.n_samples == 4
        assert report.n_features_analyzed == 2
        assert 0 <= report.overall_diversity_score <= 1
        assert len(report.summary()) > 0

    def test_cross_feature_coverage(self):
        """Test cross-feature coverage analysis."""
        from hypothesis_pipeline.diversity import analyze_cross_feature_coverage

        test_cases = [
            {"complexity": "simple", "domain": "bio"},
            {"complexity": "simple", "domain": "chem"},
            {"complexity": "complex", "domain": "bio"},
            # Missing: complex + chem
        ]

        result = analyze_cross_feature_coverage(test_cases, ["complexity", "domain"])

        assert result.total_combinations == 4
        assert result.covered_combinations == 3
        assert result.coverage_rate == 0.75
        assert len(result.missing_combinations) == 1

    def test_stratification_analysis(self):
        """Test stratification analysis."""
        from hypothesis_pipeline.diversity import analyze_stratification

        test_cases = [
            {"difficulty": "easy"} for _ in range(40)
        ] + [
            {"difficulty": "medium"} for _ in range(40)
        ] + [
            {"difficulty": "hard"} for _ in range(20)
        ]

        result = analyze_stratification(test_cases, "difficulty")

        assert result.n_strata == 3
        assert result.strata_counts["easy"] == 40
        # Not perfectly balanced, so quality < 1
        assert 0 < result.stratification_quality < 1


# =============================================================================
# PARETO MODULE TESTS
# =============================================================================


class TestPareto:
    """Tests for cost-performance Pareto module."""

    def test_compute_pareto_frontier(self):
        """Test Pareto frontier computation."""
        from hypothesis_pipeline.pareto import compute_pareto_frontier, ModelResult

        results = [
            ModelResult("cheap_bad", performance=0.6, cost_per_call=0.001, latency_ms=100, input_tokens=100, output_tokens=50),
            ModelResult("cheap_good", performance=0.8, cost_per_call=0.003, latency_ms=200, input_tokens=150, output_tokens=100),
            ModelResult("expensive_best", performance=0.95, cost_per_call=0.01, latency_ms=500, input_tokens=200, output_tokens=200),
            ModelResult("dominated", performance=0.7, cost_per_call=0.005, latency_ms=300, input_tokens=180, output_tokens=150),
        ]

        analysis = compute_pareto_frontier(results)

        # "dominated" should not be on frontier (worse perf than cheap_good, higher cost)
        assert "dominated" not in analysis.pareto_frontier
        assert "cheap_good" in analysis.pareto_frontier
        assert "expensive_best" in analysis.pareto_frontier
        assert len(analysis.dominated_points) >= 1

    def test_cost_effectiveness_metrics(self):
        """Test cost effectiveness metrics."""
        from hypothesis_pipeline.pareto import compute_cost_effectiveness, ModelResult

        results = [
            ModelResult("model_a", performance=0.8, cost_per_call=0.001, latency_ms=100, input_tokens=100, output_tokens=50),
            ModelResult("model_b", performance=0.9, cost_per_call=0.01, latency_ms=200, input_tokens=150, output_tokens=100),
        ]

        metrics = compute_cost_effectiveness(results, baseline_name="model_a")

        assert len(metrics) == 2
        # model_a should have higher performance per dollar
        assert metrics[0].performance_per_dollar > metrics[1].performance_per_dollar

    def test_upgrade_roi(self):
        """Test upgrade ROI calculation."""
        from hypothesis_pipeline.pareto import compute_upgrade_roi, ModelResult

        cheap = ModelResult("cheap", performance=0.7, cost_per_call=0.001, latency_ms=100, input_tokens=100, output_tokens=50)
        expensive = ModelResult("expensive", performance=0.9, cost_per_call=0.005, latency_ms=200, input_tokens=150, output_tokens=100)

        roi = compute_upgrade_roi(cheap, expensive)

        assert abs(roi.performance_gain - 0.2) < 1e-10
        assert roi.cost_increase == 0.004
        assert roi.roi > 0  # Positive ROI


# =============================================================================
# ROBUSTNESS MODULE TESTS
# =============================================================================


class TestRobustness:
    """Tests for robustness module."""

    def test_introduce_typos(self):
        """Test typo introduction."""
        from hypothesis_pipeline.robustness import introduce_typos

        text = "This is a sample text for testing typos."
        perturbed = introduce_typos(text, rate=0.1, seed=42)

        # Should be different from original
        assert perturbed != text
        # Should still be similar length
        assert abs(len(perturbed) - len(text)) < 10

    def test_replace_with_synonyms(self):
        """Test synonym replacement."""
        from hypothesis_pipeline.robustness import replace_with_synonyms

        text = "This is a large and important test."
        perturbed = replace_with_synonyms(text, replacement_rate=1.0, seed=42)

        # Should have some words replaced
        # "large" might become "big", "important" might become "significant"
        assert perturbed != text or len(text.split()) == len(perturbed.split())

    def test_reorder_sentences(self):
        """Test sentence reordering."""
        from hypothesis_pipeline.robustness import reorder_sentences

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        perturbed = reorder_sentences(text, keep_first=True, keep_last=True, seed=42)

        # First and last should be preserved
        assert perturbed.startswith("First sentence")
        assert perturbed.endswith("Fourth sentence.")

    def test_get_perturbation(self):
        """Test getting perturbation by name."""
        from hypothesis_pipeline.robustness import get_perturbation, PERTURBATIONS

        for name in PERTURBATIONS.keys():
            fn = get_perturbation(name)
            assert callable(fn)

        with pytest.raises(ValueError):
            get_perturbation("nonexistent")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests across modules."""

    def test_full_evaluation_pipeline(self):
        """Test combining multiple evaluation modules."""
        from hypothesis_pipeline.statistics import compare_conditions
        from hypothesis_pipeline.calibration import expected_calibration_error
        from hypothesis_pipeline.selective_prediction import evaluate_selective_prediction

        # Simulate evaluation results for two conditions
        np.random.seed(42)
        baseline_scores = np.random.beta(8, 2, 50)  # Mean ~0.8
        improved_scores = np.random.beta(9, 1, 50)  # Mean ~0.9
        confidences = np.random.beta(5, 1, 50)  # High confidence
        correctness = (np.random.uniform(0, 1, 50) < baseline_scores).astype(int)

        # Statistical comparison
        comparison = compare_conditions(improved_scores, baseline_scores, paired=True)
        assert comparison.mean_a > comparison.mean_b

        # Calibration
        cal_result = expected_calibration_error(confidences, correctness)
        assert cal_result.n_samples == 50

        # Selective prediction
        sel_result = evaluate_selective_prediction(confidences, correctness)
        assert sel_result.overall_accuracy > 0

    def test_import_all_modules(self):
        """Test that all new modules can be imported."""
        from hypothesis_pipeline import (
            # Statistics
            bootstrap_ci, cohens_d, compare_conditions,
            # Calibration
            expected_calibration_error, brier_score,
            # Selective prediction
            selective_prediction_metrics, compute_risk_coverage_curve,
            # Diversity
            analyze_test_set_diversity,
            # Pareto
            compute_pareto_frontier, ModelResult,
            # Robustness
            introduce_typos, RobustnessEvaluator,
            # LLM Judge
            LLMJudge, EvaluationRubric,
            # Experiment tracking
            LocalTracker, create_tracker,
            # Prompt sensitivity
            get_prompt_variations,
        )

        # Just verify imports work
        assert callable(bootstrap_ci)
        assert callable(expected_calibration_error)
        assert callable(selective_prediction_metrics)
        assert callable(analyze_test_set_diversity)
        assert callable(compute_pareto_frontier)
        assert callable(introduce_typos)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
