"""Integration tests for the evaluation runner."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from flow_panel_optimizer.evaluation.runner import (
    AblationRunner,
    TrialResult,
    ExperimentResults,
)
from flow_panel_optimizer.evaluation.test_cases import (
    PanelDesignTestCase,
    TestCaseType,
    TestSuite,
    build_ablation_test_suite,
)
from flow_panel_optimizer.evaluation.conditions import (
    ExperimentalCondition,
    RetrievalMode,
    CONDITIONS,
    CORE_CONDITIONS,
)


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = TrialResult(
            condition_name="baseline",
            test_case_id="test_001",
            test_case_type="in_distribution",
            raw_response="Test response",
            extracted_assignments={"CD3": "BV421"},
            tool_calls_made=[],
            assignment_accuracy=0.75,
            complexity_index=2.5,
            ground_truth_ci=3.0,
            ci_improvement=0.167,
            latency_seconds=5.2,
            input_tokens=100,
            output_tokens=50,
        )

        d = result.to_dict()

        assert d["condition_name"] == "baseline"
        assert d["test_case_id"] == "test_001"
        assert d["assignment_accuracy"] == 0.75
        assert d["tool_calls_count"] == 0

    def test_response_truncation(self):
        """Test that long responses are truncated in to_dict."""
        long_response = "x" * 1000
        result = TrialResult(
            condition_name="test",
            test_case_id="test",
            test_case_type="test",
            raw_response=long_response,
            extracted_assignments={},
            tool_calls_made=[],
            assignment_accuracy=0.0,
            complexity_index=0.0,
            ground_truth_ci=0.0,
            ci_improvement=0.0,
            latency_seconds=0.0,
            input_tokens=0,
            output_tokens=0,
        )

        d = result.to_dict()
        assert len(d["raw_response"]) == 503  # 500 + "..."


class TestExperimentResults:
    """Tests for ExperimentResults aggregation."""

    @pytest.fixture
    def sample_trials(self):
        """Create sample trial results."""
        return [
            TrialResult(
                condition_name="baseline",
                test_case_id="test_001",
                test_case_type="in_distribution",
                raw_response="",
                extracted_assignments={},
                tool_calls_made=[],
                assignment_accuracy=0.5,
                complexity_index=3.0,
                ground_truth_ci=2.5,
                ci_improvement=-0.2,
                latency_seconds=5.0,
                input_tokens=100,
                output_tokens=50,
            ),
            TrialResult(
                condition_name="baseline",
                test_case_id="test_002",
                test_case_type="out_of_distribution",
                raw_response="",
                extracted_assignments={},
                tool_calls_made=[],
                assignment_accuracy=0.3,
                complexity_index=4.0,
                ground_truth_ci=2.0,
                ci_improvement=-1.0,
                latency_seconds=6.0,
                input_tokens=120,
                output_tokens=60,
            ),
            TrialResult(
                condition_name="mcp_only",
                test_case_id="test_001",
                test_case_type="in_distribution",
                raw_response="",
                extracted_assignments={},
                tool_calls_made=[{"tool": "analyze_panel"}],
                assignment_accuracy=0.8,
                complexity_index=2.0,
                ground_truth_ci=2.5,
                ci_improvement=0.2,
                latency_seconds=15.0,
                input_tokens=200,
                output_tokens=100,
            ),
        ]

    def test_to_dataframe(self, sample_trials):
        """Test conversion to pandas DataFrame."""
        results = ExperimentResults(
            experiment_name="test_exp",
            trials=sample_trials,
        )

        df = results.to_dataframe()

        assert len(df) == 3
        assert "condition" in df.columns
        assert "accuracy" in df.columns
        assert "tool_calls" in df.columns

    def test_summary_by_condition(self, sample_trials):
        """Test aggregation by condition."""
        results = ExperimentResults(
            experiment_name="test_exp",
            trials=sample_trials,
        )

        summary = results.summary_by_condition()

        # Should have baseline and mcp_only
        assert "baseline" in summary.index
        assert "mcp_only" in summary.index


class TestAblationRunner:
    """Tests for AblationRunner."""

    @pytest.fixture
    def mock_runner(self):
        """Create runner with mocked API client."""
        with patch("flow_panel_optimizer.evaluation.runner.Anthropic"):
            runner = AblationRunner(api_key="test_key", model="claude-test")
            return runner

    def test_initialization(self, mock_runner):
        """Test runner initialization."""
        assert mock_runner.model == "claude-test"
        assert mock_runner.max_concurrent == 5
        assert len(mock_runner.mcp_tools) == 4

    def test_build_prompt_baseline(self, mock_runner):
        """Test prompt building for baseline condition."""
        test_case = PanelDesignTestCase(
            id="test",
            case_type=TestCaseType.IN_DISTRIBUTION,
            biological_question="Design a T cell panel",
            required_markers=["CD3", "CD4"],
            marker_expression={"CD3": "high", "CD4": "high"},
            candidate_fluorophores=["BV421", "PE"],
            ground_truth_assignments={"CD3": "BV421", "CD4": "PE"},
            ground_truth_complexity_index=1.0,
        )

        baseline_condition = ExperimentalCondition(
            name="baseline",
            retrieval_mode=RetrievalMode.NONE,
            retrieval_weight=0.0,
            mcp_enabled=False,
            description="Test",
        )

        prompt = mock_runner._build_prompt(test_case, baseline_condition)

        assert "Design a T cell panel" in prompt
        assert "CD3" in prompt
        assert "CD4" in prompt
        # No retrieval context
        assert "OMIP" not in prompt

    def test_build_prompt_with_retrieval(self, mock_runner):
        """Test prompt building with retrieval."""
        test_case = PanelDesignTestCase(
            id="test",
            case_type=TestCaseType.IN_DISTRIBUTION,
            biological_question="Design a T cell panel",
            required_markers=["CD3", "CD4"],
            marker_expression={"CD3": "high", "CD4": "high"},
            candidate_fluorophores=["BV421", "PE"],
            ground_truth_assignments={"CD3": "BV421", "CD4": "PE"},
            ground_truth_complexity_index=1.0,
        )

        retrieval_condition = ExperimentalCondition(
            name="retrieval",
            retrieval_mode=RetrievalMode.STANDARD,
            retrieval_weight=1.0,
            mcp_enabled=False,
            description="Test",
        )

        prompt = mock_runner._build_prompt(test_case, retrieval_condition)

        assert "OMIP" in prompt
        assert "Reference panels" in prompt

    def test_parse_assignments_json(self, mock_runner):
        """Test parsing JSON assignments from response."""
        response = '''Here is my panel design:
        {"assignments": {"CD3": "BV421", "CD4": "PE", "CD8": "APC"}}
        '''

        assignments = mock_runner._parse_assignments(response)

        assert assignments["CD3"] == "BV421"
        assert assignments["CD4"] == "PE"
        assert assignments["CD8"] == "APC"

    def test_parse_assignments_fallback(self, mock_runner):
        """Test fallback parsing for non-JSON responses."""
        response = '''My recommendations:
        CD3: BV421
        CD4: PE
        CD8: APC
        '''

        assignments = mock_runner._parse_assignments(response)

        assert "CD3" in assignments
        assert "CD4" in assignments
        assert "CD8" in assignments

    def test_score_accuracy_perfect(self, mock_runner):
        """Test accuracy scoring with perfect match."""
        predicted = {"CD3": "BV421", "CD4": "PE"}
        ground_truth = {"CD3": "BV421", "CD4": "PE"}

        accuracy = mock_runner._score_accuracy(predicted, ground_truth)
        assert accuracy == 1.0

    def test_score_accuracy_partial(self, mock_runner):
        """Test accuracy scoring with partial match."""
        predicted = {"CD3": "BV421", "CD4": "FITC"}
        ground_truth = {"CD3": "BV421", "CD4": "PE"}

        accuracy = mock_runner._score_accuracy(predicted, ground_truth)
        assert accuracy == 0.5

    def test_score_accuracy_no_match(self, mock_runner):
        """Test accuracy scoring with no match."""
        predicted = {"CD3": "FITC", "CD4": "APC"}
        ground_truth = {"CD3": "BV421", "CD4": "PE"}

        accuracy = mock_runner._score_accuracy(predicted, ground_truth)
        assert accuracy == 0.0


class TestTestSuiteGeneration:
    """Tests for test suite generation."""

    def test_build_ablation_test_suite(self):
        """Test building complete test suite."""
        suite = build_ablation_test_suite(
            n_in_dist=3,
            n_near_dist=2,
            n_out_dist=2,
            n_adversarial=1,
        )

        assert suite.name == "mcp_ablation_v2"
        assert len(suite.test_cases) == 8

        # Check distribution
        in_dist = suite.filter_by_type(TestCaseType.IN_DISTRIBUTION)
        assert len(in_dist) == 3

        out_dist = suite.filter_by_type(TestCaseType.OUT_OF_DISTRIBUTION)
        assert len(out_dist) == 2

    def test_test_case_has_ground_truth(self):
        """Test that generated test cases have ground truth."""
        suite = build_ablation_test_suite(n_in_dist=2, n_near_dist=0, n_out_dist=0, n_adversarial=0)

        for tc in suite.test_cases:
            assert tc.ground_truth_assignments
            assert tc.ground_truth_complexity_index >= 0
            assert tc.required_markers
            assert tc.marker_expression


class TestConditions:
    """Tests for experimental conditions."""

    def test_conditions_complete(self):
        """Test that all conditions are properly defined."""
        assert len(CONDITIONS) == 8

        # Check each condition has required fields
        for cond in CONDITIONS:
            assert cond.name
            assert isinstance(cond.retrieval_mode, RetrievalMode)
            assert isinstance(cond.mcp_enabled, bool)
            assert cond.description

    def test_core_conditions_subset(self):
        """Test that core conditions are a subset of all conditions."""
        core_names = {c.name for c in CORE_CONDITIONS}
        all_names = {c.name for c in CONDITIONS}

        assert core_names.issubset(all_names)
        assert len(CORE_CONDITIONS) == 4
