"""
Tests for the experiment tracker.

Verifies that experiments are saved with proper outputs including:
- Config
- Metadata
- Inputs (trial inputs)
- Results
- Summary
- Conclusions
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from hypothesis_pipeline import (
    ExperimentTracker,
    ExperimentResults,
    TrialInput,
    TrialResult,
    HypothesisCondition,
    ReasoningType,
    ContextLevel,
    RAGMode,
)


@pytest.fixture
def temp_experiments_dir():
    """Create a temporary experiments directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample experiment configuration."""
    return {
        "name": "test_experiment",
        "description": "Test experiment for tracker",
        "hypothesis": "Testing saves all data correctly",
        "tags": ["test", "unit"],
        "models": ["claude-sonnet-4-20250514"],
        "reasoning_types": ["direct", "cot"],
        "context_levels": ["minimal", "standard"],
    }


@pytest.fixture
def sample_inputs():
    """Sample trial inputs."""
    return [
        TrialInput(
            id="input_1",
            raw_input={"data": "test data 1", "value": 42},
            prompt="What is the answer to input 1?",
            ground_truth={"answer": "42"},
            metadata={"complexity": "simple", "category": "math"},
        ),
        TrialInput(
            id="input_2",
            raw_input={"data": "test data 2", "nested": {"a": 1, "b": 2}},
            prompt="What is the answer to input 2?",
            ground_truth={"answer": "3"},
            metadata={"complexity": "medium", "category": "logic"},
        ),
        TrialInput(
            id="input_3",
            raw_input="simple string input",
            prompt="What is the answer to input 3?",
            ground_truth="simple answer",
            metadata={"complexity": "simple"},
        ),
    ]


@pytest.fixture
def sample_results():
    """Sample experiment results."""
    conditions = [
        HypothesisCondition(
            name="direct_minimal",
            reasoning_type=ReasoningType.DIRECT,
            context_level=ContextLevel.MINIMAL,
        ),
        HypothesisCondition(
            name="cot_standard",
            reasoning_type=ReasoningType.COT,
            context_level=ContextLevel.STANDARD,
        ),
    ]

    trials = [
        TrialResult(
            trial_id="input_1",
            condition_name="direct_minimal",
            start_time=datetime.now(),
            end_time=datetime.now(),
            latency_seconds=1.5,
            raw_response="The answer is 42",
            extracted_output={"answer": "42"},
            scores={"accuracy": 1.0, "f1": 0.95},
            input_tokens=100,
            output_tokens=50,
        ),
        TrialResult(
            trial_id="input_2",
            condition_name="direct_minimal",
            start_time=datetime.now(),
            end_time=datetime.now(),
            latency_seconds=2.0,
            raw_response="The answer is 3",
            extracted_output={"answer": "3"},
            scores={"accuracy": 1.0, "f1": 0.90},
            input_tokens=120,
            output_tokens=40,
        ),
        TrialResult(
            trial_id="input_1",
            condition_name="cot_standard",
            start_time=datetime.now(),
            end_time=datetime.now(),
            latency_seconds=3.0,
            raw_response="Let me think... The answer is 42",
            extracted_output={"answer": "42"},
            scores={"accuracy": 1.0, "f1": 0.98},
            input_tokens=150,
            output_tokens=80,
        ),
    ]

    return ExperimentResults(
        experiment_name="test_experiment",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        conditions=conditions,
        trials=trials,
    )


class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    def test_start_experiment_creates_directory(self, temp_experiments_dir, sample_config):
        """Test that start_experiment creates experiment directory."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)

        # Check directory exists
        exp_dir = temp_experiments_dir / metadata.experiment_id
        assert exp_dir.exists()

        # Check config file exists
        config_file = exp_dir / "config.yaml"
        assert config_file.exists()

        # Check metadata file exists
        meta_file = exp_dir / "metadata.yaml"
        assert meta_file.exists()

    def test_start_experiment_saves_config(self, temp_experiments_dir, sample_config):
        """Test that config is saved correctly."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)

        # Load saved config
        config_file = temp_experiments_dir / metadata.experiment_id / "config.yaml"
        with open(config_file) as f:
            saved_config = yaml.safe_load(f)

        # Verify config contents
        assert saved_config["name"] == sample_config["name"]
        assert saved_config["hypothesis"] == sample_config["hypothesis"]
        assert saved_config["tags"] == sample_config["tags"]
        assert saved_config["models"] == sample_config["models"]

    def test_start_experiment_saves_metadata(self, temp_experiments_dir, sample_config):
        """Test that metadata is saved with git info."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)

        # Load saved metadata
        meta_file = temp_experiments_dir / metadata.experiment_id / "metadata.yaml"
        with open(meta_file) as f:
            saved_meta = yaml.safe_load(f)

        # Verify metadata
        assert saved_meta["experiment_id"] == metadata.experiment_id
        assert saved_meta["name"] == sample_config["name"]
        assert saved_meta["hypothesis"] == sample_config["hypothesis"]
        assert "timestamp" in saved_meta
        assert "config_hash" in saved_meta
        assert "python_version" in saved_meta

    def test_save_inputs(self, temp_experiments_dir, sample_config, sample_inputs):
        """Test that inputs are saved correctly."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)

        # Save inputs
        inputs_file = tracker.save_inputs(metadata, sample_inputs)

        # Check file exists
        assert Path(inputs_file).exists()

        # Load and verify
        with open(inputs_file) as f:
            saved_data = json.load(f)

        assert saved_data["n_inputs"] == 3
        assert saved_data["input_ids"] == ["input_1", "input_2", "input_3"]
        assert len(saved_data["inputs"]) == 3

        # Check first input
        input_1 = saved_data["inputs"][0]
        assert input_1["id"] == "input_1"
        assert input_1["prompt"] == "What is the answer to input 1?"
        assert input_1["metadata"]["complexity"] == "simple"
        assert input_1["raw_input"]["value"] == 42
        assert input_1["ground_truth"]["answer"] == "42"

    def test_save_inputs_handles_complex_objects(self, temp_experiments_dir, sample_config):
        """Test that inputs with non-serializable objects are handled."""

        class CustomObject:
            def __init__(self, value):
                self.value = value

        inputs = [
            TrialInput(
                id="complex_1",
                raw_input=CustomObject(42),  # Non-serializable
                prompt="Test",
                ground_truth=CustomObject(100),  # Non-serializable
            ),
        ]

        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)

        # Should not raise
        inputs_file = tracker.save_inputs(metadata, inputs)

        # Load and verify fallback representation
        with open(inputs_file) as f:
            saved_data = json.load(f)

        input_data = saved_data["inputs"][0]
        assert "raw_input_repr" in input_data or "raw_input" in input_data
        assert "ground_truth_repr" in input_data or "ground_truth" in input_data

    def test_save_results_with_inputs(
        self, temp_experiments_dir, sample_config, sample_inputs, sample_results
    ):
        """Test that results and inputs are saved together."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)

        # Save results with inputs
        record = tracker.save_results(metadata, sample_results, trial_inputs=sample_inputs)

        # Verify record
        assert record.n_inputs == 3
        assert record.input_ids == ["input_1", "input_2", "input_3"]
        assert record.inputs_file != ""
        assert Path(record.inputs_file).exists()

        # Verify results file
        assert Path(record.results_file).exists()
        with open(record.results_file) as f:
            results_data = json.load(f)
        assert results_data["n_trials"] == 3

    def test_save_results_creates_summary(
        self, temp_experiments_dir, sample_config, sample_results
    ):
        """Test that summary statistics are computed correctly."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)

        record = tracker.save_results(metadata, sample_results)

        # Check summary
        summary = record.results_summary
        assert summary["n_trials"] == 3
        assert summary["n_conditions"] == 2
        assert summary["n_successful"] == 3
        assert summary["overall_success_rate"] == 1.0

        # Check metrics
        assert "accuracy_mean" in summary
        assert "f1_mean" in summary
        assert summary["accuracy_mean"] == 1.0

        # Check per-condition
        assert "by_condition" in summary
        assert "direct_minimal" in summary["by_condition"]
        assert "cot_standard" in summary["by_condition"]

    def test_get_experiment(self, temp_experiments_dir, sample_config, sample_results, sample_inputs):
        """Test that experiment can be retrieved after saving."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)
        tracker.save_results(metadata, sample_results, trial_inputs=sample_inputs)

        # Retrieve
        record = tracker.get_experiment(metadata.experiment_id)

        assert record is not None
        assert record.metadata.name == sample_config["name"]
        assert record.n_inputs == 3
        assert record.input_ids == ["input_1", "input_2", "input_3"]

    def test_get_inputs(self, temp_experiments_dir, sample_config, sample_inputs):
        """Test that inputs can be retrieved separately."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)
        tracker.save_inputs(metadata, sample_inputs)

        # Retrieve inputs
        inputs = tracker.get_inputs(metadata.experiment_id)

        assert inputs is not None
        assert len(inputs) == 3
        assert inputs[0]["id"] == "input_1"
        assert inputs[1]["id"] == "input_2"

    def test_add_conclusion(self, temp_experiments_dir, sample_config, sample_results):
        """Test that conclusions can be added."""
        tracker = ExperimentTracker(temp_experiments_dir)
        metadata = tracker.start_experiment(sample_config)
        tracker.save_results(metadata, sample_results)

        # Add conclusion
        conclusion = tracker.add_conclusion(
            metadata.experiment_id,
            summary="CoT outperformed direct by 3%",
            outcome="success",
            findings=["F1 improved with CoT", "Latency increased 2x"],
            next_steps=["Test with more complex inputs"],
            notes="Preliminary results look promising",
        )

        # Verify
        assert conclusion.outcome == "success"
        assert len(conclusion.findings) == 2

        # Check it's saved
        record = tracker.get_experiment(metadata.experiment_id)
        assert record.conclusion is not None
        assert record.conclusion.outcome == "success"

    def test_list_experiments(self, temp_experiments_dir):
        """Test listing experiments with filters."""
        tracker = ExperimentTracker(temp_experiments_dir)

        # Create multiple experiments
        config1 = {"name": "exp1", "tags": ["benchmark", "rag"]}
        config2 = {"name": "exp2", "tags": ["benchmark", "tools"]}
        config3 = {"name": "other", "tags": ["test"]}

        tracker.start_experiment(config1)
        tracker.start_experiment(config2)
        tracker.start_experiment(config3)

        # List all
        all_exps = tracker.list_experiments()
        assert len(all_exps) == 3

        # Filter by tag
        benchmark_exps = tracker.list_experiments(tags=["benchmark"])
        assert len(benchmark_exps) == 2

        # Filter by name
        exp_exps = tracker.list_experiments(name_contains="exp")
        assert len(exp_exps) == 2

    def test_index_updated(self, temp_experiments_dir, sample_config, sample_results, sample_inputs):
        """Test that index is updated correctly throughout lifecycle."""
        tracker = ExperimentTracker(temp_experiments_dir)

        # Start
        metadata = tracker.start_experiment(sample_config)
        assert tracker._index[metadata.experiment_id]["status"] == "running"

        # Save inputs
        tracker.save_inputs(metadata, sample_inputs)
        assert tracker._index[metadata.experiment_id]["n_inputs"] == 3

        # Save results
        tracker.save_results(metadata, sample_results)
        assert tracker._index[metadata.experiment_id]["status"] == "completed"

        # Add conclusion
        tracker.add_conclusion(metadata.experiment_id, "Test", outcome="success")
        assert tracker._index[metadata.experiment_id]["outcome"] == "success"
        assert tracker._index[metadata.experiment_id]["concluded"] == True


class TestTrialInputSerialization:
    """Tests for TrialInput.to_dict()."""

    def test_simple_input(self):
        """Test serialization of simple input."""
        trial = TrialInput(
            id="test",
            raw_input={"key": "value"},
            prompt="Test prompt",
            ground_truth=42,
            metadata={"tag": "test"},
        )

        data = trial.to_dict()

        assert data["id"] == "test"
        assert data["prompt"] == "Test prompt"
        assert data["raw_input"] == {"key": "value"}
        assert data["ground_truth"] == 42
        assert data["metadata"] == {"tag": "test"}

    def test_exclude_raw(self):
        """Test excluding raw_input."""
        trial = TrialInput(
            id="test",
            raw_input={"large": "data" * 1000},
            prompt="Test",
            ground_truth="answer",
        )

        data = trial.to_dict(include_raw=False)

        assert "raw_input" not in data
        assert data["id"] == "test"

    def test_non_serializable_fallback(self):
        """Test fallback for non-serializable objects."""

        class Custom:
            pass

        trial = TrialInput(
            id="test",
            raw_input=Custom(),
            prompt="Test",
            ground_truth=Custom(),
        )

        data = trial.to_dict()

        # Should have repr instead
        assert "raw_input_repr" in data or "raw_input" in data
        assert "raw_input_type" in data or "raw_input" in data


class TestExperimentDirectoryStructure:
    """Tests for verifying complete directory structure."""

    def test_complete_experiment_structure(
        self, temp_experiments_dir, sample_config, sample_inputs, sample_results
    ):
        """Test that all expected files are created."""
        tracker = ExperimentTracker(temp_experiments_dir)

        # Full lifecycle
        metadata = tracker.start_experiment(sample_config)
        tracker.save_results(metadata, sample_results, trial_inputs=sample_inputs)
        tracker.add_conclusion(metadata.experiment_id, "Test conclusion", outcome="success")

        # Check all files exist
        exp_dir = temp_experiments_dir / metadata.experiment_id

        expected_files = [
            "config.yaml",
            "metadata.yaml",
            "inputs.json",
            "results.json",
            "summary.yaml",
            "record.yaml",
            "conclusion.yaml",
        ]

        for filename in expected_files:
            file_path = exp_dir / filename
            assert file_path.exists(), f"Missing file: {filename}"

        # Verify index
        index_file = temp_experiments_dir / "index.yaml"
        assert index_file.exists()

        with open(index_file) as f:
            index = yaml.safe_load(f)

        assert metadata.experiment_id in index
        assert index[metadata.experiment_id]["status"] == "completed"
        assert index[metadata.experiment_id]["n_inputs"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
