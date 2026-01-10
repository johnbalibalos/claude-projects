"""
Tests for cross_validator module.

Tests the cross-validation logic without making actual API calls.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path

# Direct import to avoid numpy dependency
import importlib.util
libs_path = Path(__file__).parent.parent.parent
spec = importlib.util.spec_from_file_location(
    "hypothesis_pipeline.cross_validator",
    libs_path / "hypothesis_pipeline" / "cross_validator.py"
)
cross_validator_module = importlib.util.module_from_spec(spec)
sys.modules["hypothesis_pipeline.cross_validator"] = cross_validator_module
spec.loader.exec_module(cross_validator_module)

CrossValidator = cross_validator_module.CrossValidator
BatchCrossValidator = cross_validator_module.BatchCrossValidator
CrossValidationResult = cross_validator_module.CrossValidationResult
Discrepancy = cross_validator_module.Discrepancy


class TestCrossValidationResult:
    """Test CrossValidationResult dataclass."""

    def test_interpretation_ground_truth_clearly_correct(self):
        result = CrossValidationResult(confidence_score=15)
        assert result.interpretation == "ground_truth_clearly_correct"
        assert not result.should_trust_prediction

    def test_interpretation_ground_truth_preferred(self):
        result = CrossValidationResult(confidence_score=45)
        assert result.interpretation == "ground_truth_preferred"
        assert not result.should_trust_prediction

    def test_interpretation_both_acceptable(self):
        result = CrossValidationResult(confidence_score=60)
        assert result.interpretation == "both_acceptable"
        assert result.should_trust_prediction

    def test_interpretation_prediction_preferred(self):
        result = CrossValidationResult(confidence_score=80)
        assert result.interpretation == "prediction_preferred"
        assert result.should_trust_prediction

    def test_interpretation_prediction_clearly_correct(self):
        result = CrossValidationResult(confidence_score=95)
        assert result.interpretation == "prediction_clearly_correct"
        assert result.should_trust_prediction

    def test_summary(self):
        result = CrossValidationResult(
            confidence_score=75,
            n_critical=1,
            n_moderate=2,
            n_minor=3,
            reconciliation="The prediction uses valid alternative approach.",
        )
        summary = result.summary()
        assert "75/100" in summary
        assert "prediction_preferred" in summary
        assert "1 critical" in summary
        assert "2 moderate" in summary
        assert "3 minor" in summary


class TestCrossValidator:
    """Test CrossValidator class."""

    def create_mock_model(self, response: str):
        """Create a mock model that returns the given response."""
        model = Mock()
        model.generate = Mock(return_value=response)
        return model

    def test_validate_parses_json_response(self):
        """Test that validator correctly parses JSON response."""
        mock_response = """
Let me analyze the differences...

```json
{
  "discrepancies": [
    {
      "aspect": "gate_name",
      "prediction_value": "T cells",
      "ground_truth_value": "T lymphocytes",
      "severity": "minor",
      "leaning": 70,
      "reasoning": "Both names are valid, T cells is more common"
    },
    {
      "aspect": "missing_gate",
      "prediction_value": "none",
      "ground_truth_value": "CD45+ gate",
      "severity": "critical",
      "leaning": 20,
      "reasoning": "CD45+ gate is important for proper gating"
    }
  ],
  "prediction_defense": "The prediction uses a streamlined approach...",
  "ground_truth_defense": "The ground truth follows HIPC standards...",
  "reconciliation": "Overall the prediction is partially valid but missing key gates.",
  "confidence_score": 45
}
```
"""
        model = self.create_mock_model(mock_response)
        validator = CrossValidator(model, model_name="test-model")

        result = validator.validate(
            prediction={"gates": ["T cells", "B cells"]},
            ground_truth={"gates": ["T lymphocytes", "B lymphocytes", "CD45+"]},
        )

        assert result.confidence_score == 45
        assert result.interpretation == "ground_truth_preferred"
        assert len(result.discrepancies) == 2
        assert result.n_critical == 1
        assert result.n_minor == 1
        assert "streamlined" in result.prediction_defense
        assert "HIPC" in result.ground_truth_defense

    def test_validate_handles_malformed_json(self):
        """Test that validator handles malformed JSON gracefully."""
        mock_response = "This is not valid JSON at all..."
        model = self.create_mock_model(mock_response)
        validator = CrossValidator(model, model_name="test-model")

        result = validator.validate(
            prediction="some prediction",
            ground_truth="some ground truth",
        )

        # Should return default uncertain result
        assert result.confidence_score == 50
        assert "Failed to parse" in result.reconciliation

    def test_validate_clamps_confidence_score(self):
        """Test that confidence scores are clamped to 0-100."""
        mock_response = """```json
{
  "discrepancies": [],
  "confidence_score": 150
}
```"""
        model = self.create_mock_model(mock_response)
        validator = CrossValidator(model, model_name="test-model")

        result = validator.validate("pred", "gt")
        assert result.confidence_score == 100

        # Test lower bound
        mock_response2 = """```json
{
  "discrepancies": [],
  "confidence_score": -50
}
```"""
        model2 = self.create_mock_model(mock_response2)
        validator2 = CrossValidator(model2, model_name="test-model")
        result2 = validator2.validate("pred", "gt")
        assert result2.confidence_score == 0

    def test_validate_gating_result(self):
        """Test convenience method for gating results."""
        mock_response = """```json
{
  "discrepancies": [],
  "prediction_defense": "Good prediction",
  "ground_truth_defense": "Standard approach",
  "reconciliation": "Both valid",
  "confidence_score": 65
}
```"""
        model = self.create_mock_model(mock_response)
        validator = CrossValidator(model, model_name="test-model")

        result_dict = {
            "test_case_id": "OMIP-074",
            "predicted_gates": ["T cells", "B cells"],
            "evaluation": {
                "hierarchy_f1": 0.8,
            },
        }

        gt_hierarchy = {
            "root": {
                "name": "All Events",
                "children": [{"name": "T cells"}, {"name": "B cells"}],
            }
        }

        result = validator.validate_gating_result(
            result_dict=result_dict,
            ground_truth_hierarchy=gt_hierarchy,
            panel_markers=["CD3", "CD4", "CD19"],
        )

        assert result.confidence_score == 65
        assert result.should_trust_prediction


class TestBatchCrossValidator:
    """Test BatchCrossValidator class."""

    def test_compute_statistics(self):
        """Test aggregate statistics computation."""
        # Create mock validations
        validations = [
            ({}, CrossValidationResult(confidence_score=30, n_critical=1)),
            ({}, CrossValidationResult(confidence_score=60, n_moderate=2)),
            ({}, CrossValidationResult(confidence_score=80, n_minor=3)),
            ({}, CrossValidationResult(confidence_score=90, n_minor=1)),
        ]

        mock_validator = Mock()
        batch = BatchCrossValidator(mock_validator)
        stats = batch.compute_statistics(validations)

        assert stats["n_samples"] == 4
        assert stats["mean_confidence"] == 65.0
        assert stats["min_confidence"] == 30
        assert stats["max_confidence"] == 90
        assert stats["predictions_trusted"] == 3  # scores > 50
        assert stats["predictions_rejected"] == 1  # scores <= 50
        assert stats["total_critical_discrepancies"] == 1
        assert stats["total_moderate_discrepancies"] == 2
        assert stats["total_minor_discrepancies"] == 4

    def test_filter_by_confidence(self):
        """Test filtering validations by confidence range."""
        validations = [
            ({"id": 1}, CrossValidationResult(confidence_score=20)),
            ({"id": 2}, CrossValidationResult(confidence_score=50)),
            ({"id": 3}, CrossValidationResult(confidence_score=75)),
            ({"id": 4}, CrossValidationResult(confidence_score=90)),
        ]

        mock_validator = Mock()
        batch = BatchCrossValidator(mock_validator)

        # Filter to middle range
        filtered = batch.filter_by_confidence(validations, min_score=40, max_score=80)
        assert len(filtered) == 2
        assert filtered[0][0]["id"] == 2
        assert filtered[1][0]["id"] == 3

    def test_get_controversial(self):
        """Test getting controversial validations."""
        validations = [
            ({"id": 1}, CrossValidationResult(confidence_score=20)),
            ({"id": 2}, CrossValidationResult(confidence_score=50)),
            ({"id": 3}, CrossValidationResult(confidence_score=55)),
            ({"id": 4}, CrossValidationResult(confidence_score=90)),
        ]

        mock_validator = Mock()
        batch = BatchCrossValidator(mock_validator)

        controversial = batch.get_controversial(validations)
        assert len(controversial) == 2
        assert all(40 <= v.confidence_score <= 60 for _, v in controversial)


class TestDiscrepancy:
    """Test Discrepancy dataclass."""

    def test_discrepancy_creation(self):
        d = Discrepancy(
            aspect="gate_naming",
            prediction_value="T cells",
            ground_truth_value="T lymphocytes",
            severity="minor",
            leaning=70,
            reasoning="Both are valid names",
        )
        assert d.aspect == "gate_naming"
        assert d.severity == "minor"
        assert d.leaning == 70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
