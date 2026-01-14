"""Tests for SQLite experiment tracker."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from experiments.tracker import ExperimentTracker, TrackerConfig
from experiments.prediction_collector import Prediction
from experiments.batch_scorer import ScoringResult


@pytest.fixture
def tracker(tmp_path):
    """Create a tracker with temp database."""
    config = TrackerConfig(db_path=tmp_path / "test.db")
    return ExperimentTracker(config)


@pytest.fixture
def sample_prediction():
    """Create a sample prediction."""
    return Prediction(
        test_case_id="OMIP-074",
        model="claude-sonnet-4-20250514-cli",
        condition="claude-sonnet-cli_minimal_direct_none",
        bootstrap_run=1,
        raw_response='{"name": "All Events", "children": []}',
        tokens_used=150,
        timestamp=datetime.now(),
        prompt="Predict the gating hierarchy...",
        run_id="test-run-001",
    )


@pytest.fixture
def sample_scoring_result():
    """Create a sample scoring result."""
    return ScoringResult(
        test_case_id="OMIP-074",
        model="claude-sonnet-4-20250514-cli",
        condition="claude-sonnet-cli_minimal_direct_none",
        bootstrap_run=1,
        hierarchy_f1=0.325,
        structure_accuracy=0.045,
        critical_gate_recall=0.75,
        hallucination_rate=0.14,
        parse_success=True,
        raw_response='{"name": "All Events", "children": []}',
    )


class TestRunManagement:
    """Tests for run lifecycle."""

    def test_start_run(self, tracker):
        """Test starting a new run."""
        run_id = tracker.start_run(
            run_id="test-001",
            config={"n_bootstrap": 3, "models": ["claude-sonnet"]},
            notes="Test run",
        )

        assert run_id == "test-001"

        run = tracker.get_run(run_id)
        assert run is not None
        assert run["run_id"] == "test-001"
        assert run["notes"] == "Test run"
        assert run["completed_at"] is None

    def test_complete_run(self, tracker):
        """Test marking run as complete."""
        tracker.start_run("test-001")
        tracker.complete_run("test-001")

        run = tracker.get_run("test-001")
        assert run["completed_at"] is not None

    def test_get_nonexistent_run(self, tracker):
        """Test getting a run that doesn't exist."""
        assert tracker.get_run("nonexistent") is None


class TestPredictionLogging:
    """Tests for prediction logging."""

    def test_log_prediction(self, tracker, sample_prediction):
        """Test logging a prediction."""
        tracker.start_run("test-001")
        pred_id = tracker.log_prediction("test-001", sample_prediction)

        assert pred_id > 0

        results = tracker.query(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        )
        assert len(results) == 1
        assert results[0]["test_case_id"] == "OMIP-074"
        assert results[0]["model"] == "claude-sonnet-4-20250514-cli"
        assert results[0]["context_level"] == "minimal"
        assert results[0]["prompt_strategy"] == "direct"

    def test_log_prediction_with_error(self, tracker):
        """Test logging a prediction with an error."""
        tracker.start_run("test-001")

        error_pred = Prediction(
            test_case_id="OMIP-074",
            model="claude-sonnet-4-20250514-cli",
            condition="claude-sonnet-cli_minimal_direct_none",
            bootstrap_run=1,
            raw_response="",
            tokens_used=0,
            timestamp=datetime.now(),
            error="Rate limit exceeded (429)",
            run_id="test-001",
        )

        pred_id = tracker.log_prediction("test-001", error_pred)

        results = tracker.query(
            "SELECT error_type, error_message FROM predictions WHERE id = ?",
            (pred_id,),
        )
        assert results[0]["error_type"] == "rate_limit"
        assert "429" in results[0]["error_message"]

    def test_prediction_exists(self, tracker, sample_prediction):
        """Test checking if prediction exists."""
        tracker.start_run("test-001")

        # Doesn't exist yet
        assert tracker.prediction_exists(
            "test-001", "OMIP-074", "claude-sonnet-4-20250514-cli",
            "minimal", "direct", 1
        ) is None

        # Log it
        tracker.log_prediction("test-001", sample_prediction)

        # Now exists
        pred_id = tracker.prediction_exists(
            "test-001", "OMIP-074", "claude-sonnet-4-20250514-cli",
            "minimal", "direct", 1
        )
        assert pred_id is not None

    def test_get_completed_keys(self, tracker, sample_prediction):
        """Test getting completed prediction keys for resume."""
        tracker.start_run("test-001")
        tracker.log_prediction("test-001", sample_prediction)

        keys = tracker.get_completed_keys("test-001")

        assert len(keys) == 1
        key = list(keys)[0]
        assert key[0] == 1  # bootstrap_run
        assert key[1] == "OMIP-074"  # test_case_id

    def test_idempotent_logging(self, tracker, sample_prediction):
        """Test that logging the same prediction twice is idempotent."""
        tracker.start_run("test-001")

        tracker.log_prediction("test-001", sample_prediction)
        tracker.log_prediction("test-001", sample_prediction)

        count = tracker.query(
            "SELECT COUNT(*) as n FROM predictions WHERE run_id = 'test-001'"
        )
        assert count[0]["n"] == 1


class TestScoring:
    """Tests for score logging."""

    def test_log_scores(self, tracker, sample_prediction, sample_scoring_result):
        """Test logging scores for a prediction."""
        tracker.start_run("test-001")
        pred_id = tracker.log_prediction("test-001", sample_prediction)

        tracker.log_scores(pred_id, sample_scoring_result)

        results = tracker.query(
            "SELECT hierarchy_f1, parse_success FROM predictions WHERE id = ?",
            (pred_id,),
        )
        assert abs(results[0]["hierarchy_f1"] - 0.325) < 0.001
        assert results[0]["parse_success"] == 1

    def test_log_scores_by_key(self, tracker, sample_prediction, sample_scoring_result):
        """Test logging scores by matching key fields."""
        tracker.start_run("test-001")
        tracker.log_prediction("test-001", sample_prediction)

        tracker.log_scores_by_key("test-001", sample_scoring_result)

        results = tracker.query(
            "SELECT hierarchy_f1 FROM predictions WHERE run_id = 'test-001'"
        )
        assert abs(results[0]["hierarchy_f1"] - 0.325) < 0.001


class TestJudgeScores:
    """Tests for LLM judge score logging."""

    def test_log_judge_score(self, tracker, sample_prediction):
        """Test logging judge scores."""
        tracker.start_run("test-001")
        pred_id = tracker.log_prediction("test-001", sample_prediction)

        tracker.log_judge_score(
            prediction_id=pred_id,
            judge_model="gemini-2.5-pro",
            judge_style="default",
            score=0.75,
            rationale="Good hierarchy structure, minor issues with naming.",
        )

        results = tracker.query(
            "SELECT * FROM judge_scores WHERE prediction_id = ?",
            (pred_id,),
        )
        assert len(results) == 1
        assert results[0]["score"] == 0.75
        assert results[0]["judge_style"] == "default"

    def test_multiple_judge_styles(self, tracker, sample_prediction):
        """Test logging multiple judge styles for one prediction."""
        tracker.start_run("test-001")
        pred_id = tracker.log_prediction("test-001", sample_prediction)

        for style, score in [("default", 0.7), ("validation", 0.8), ("binary", 1.0)]:
            tracker.log_judge_score(pred_id, "gemini-2.5-pro", style, score)

        results = tracker.query(
            "SELECT judge_style, score FROM judge_scores WHERE prediction_id = ?",
            (pred_id,),
        )
        assert len(results) == 3


class TestQueries:
    """Tests for analysis queries."""

    def test_compare_models(self, tracker):
        """Test model comparison query."""
        tracker.start_run("test-001")

        # Log predictions for two models
        for model, f1 in [("claude-sonnet-4-20250514-cli", 0.4), ("claude-opus-4-20250514-cli", 0.35)]:
            pred = Prediction(
                test_case_id="OMIP-074",
                model=model,
                condition=f"{model.replace('-cli', '')}_minimal_direct_none",
                bootstrap_run=1,
                raw_response="{}",
                tokens_used=100,
                timestamp=datetime.now(),
                run_id="test-001",
            )
            pred_id = tracker.log_prediction("test-001", pred)

            # Update with scores directly via SQL for testing
            tracker.query(
                f"UPDATE predictions SET hierarchy_f1 = {f1}, parse_success = 1 WHERE id = {pred_id}"
            )

        results = tracker.compare_models(
            "%sonnet%", "%opus%", run_id="test-001"
        )

        assert len(results) == 1
        assert results[0]["delta"] == pytest.approx(0.05, abs=0.01)

    def test_get_stats_by_model(self, tracker, sample_prediction, sample_scoring_result):
        """Test aggregate stats by model."""
        tracker.start_run("test-001")
        pred_id = tracker.log_prediction("test-001", sample_prediction)
        tracker.log_scores(pred_id, sample_scoring_result)

        stats = tracker.get_stats_by_model("test-001")

        assert len(stats) == 1
        assert stats[0]["model"] == "claude-sonnet-4-20250514-cli"
        assert stats[0]["n_predictions"] == 1
        assert stats[0]["mean_f1"] == pytest.approx(0.325, abs=0.01)

    def test_query_to_dataframe(self, tracker, sample_prediction):
        """Test DataFrame output."""
        pytest.importorskip("pandas")

        tracker.start_run("test-001")
        tracker.log_prediction("test-001", sample_prediction)

        df = tracker.query_to_dataframe(
            "SELECT * FROM predictions WHERE run_id = ?",
            ("test-001",),
        )

        assert len(df) == 1
        assert df.iloc[0]["test_case_id"] == "OMIP-074"


class TestExport:
    """Tests for export functionality."""

    def test_export_to_json(self, tracker, sample_prediction, tmp_path):
        """Test exporting run to JSON."""
        tracker.start_run("test-001")
        pred_id = tracker.log_prediction("test-001", sample_prediction)
        tracker.log_judge_score(pred_id, "gemini-2.5-pro", "default", 0.8)

        output_path = tmp_path / "export.json"
        tracker.export_to_json("test-001", output_path)

        import json
        with open(output_path) as f:
            data = json.load(f)

        assert data["run_id"] == "test-001"
        assert len(data["predictions"]) == 1
        assert len(data["judge_scores"]) == 1
