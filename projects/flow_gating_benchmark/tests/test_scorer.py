"""Tests for the scorer module."""

import pytest
from datetime import date

from src.curation.schemas import (
    TestCase,
    Panel,
    PanelEntry,
    GatingHierarchy,
    GateNode,
    ExperimentContext,
    CurationMetadata,
    SourceType,
)
from src.evaluation.scorer import (
    GatingScorer,
    score_prediction,
    compute_aggregate_metrics,
    ScoringResult,
)


@pytest.fixture
def simple_test_case():
    """Create a simple test case for testing."""
    return TestCase(
        test_case_id="TEST-001",
        source_type=SourceType.OMIP_PAPER,
        context=ExperimentContext(
            sample_type="Human PBMC",
            species="human",
            application="Testing",
        ),
        panel=Panel(entries=[
            PanelEntry(marker="CD3", fluorophore="PE"),
            PanelEntry(marker="CD4", fluorophore="FITC"),
            PanelEntry(marker="Live/Dead", fluorophore="7-AAD"),
        ]),
        gating_hierarchy=GatingHierarchy(
            root=GateNode(
                name="All Events",
                children=[
                    GateNode(
                        name="Singlets",
                        markers=["FSC-A", "FSC-H"],
                        is_critical=True,
                        children=[
                            GateNode(
                                name="Live",
                                markers=["7-AAD"],
                                is_critical=True,
                                children=[
                                    GateNode(name="T cells", markers=["CD3"]),
                                ]
                            )
                        ]
                    )
                ]
            )
        ),
        metadata=CurationMetadata(curation_date=date.today(), curator="Test"),
    )


class TestGatingScorer:
    """Tests for GatingScorer class."""

    def test_score_valid_response(self, simple_test_case):
        """Test scoring a valid JSON response."""
        response = '''{
            "name": "All Events",
            "children": [
                {
                    "name": "Singlets",
                    "markers": ["FSC-A", "FSC-H"],
                    "children": [
                        {
                            "name": "Live",
                            "markers": ["7-AAD"],
                            "children": [
                                {"name": "T cells", "markers": ["CD3"], "children": []}
                            ]
                        }
                    ]
                }
            ]
        }'''

        scorer = GatingScorer()
        result = scorer.score(response, simple_test_case, "test-model", "test-condition")

        assert result.parse_success
        assert result.evaluation is not None
        assert result.hierarchy_f1 == 1.0  # Perfect match

    def test_score_partial_response(self, simple_test_case):
        """Test scoring a partial response."""
        response = '''{
            "name": "All Events",
            "children": [
                {"name": "Singlets", "children": []}
            ]
        }'''

        scorer = GatingScorer()
        result = scorer.score(response, simple_test_case, "test-model", "test-condition")

        assert result.parse_success
        assert result.evaluation is not None
        assert result.hierarchy_f1 < 1.0  # Missing gates
        assert result.hierarchy_f1 > 0.0  # Some match

    def test_score_invalid_response(self, simple_test_case):
        """Test scoring an invalid response."""
        response = "I cannot predict the gating hierarchy."

        scorer = GatingScorer()
        result = scorer.score(response, simple_test_case, "test-model", "test-condition")

        assert not result.parse_success
        assert result.evaluation is None

    def test_score_batch(self, simple_test_case):
        """Test batch scoring."""
        responses = [
            ('{"name": "All Events", "children": []}', simple_test_case, "model1", "cond1"),
            ('{"name": "All Events", "children": []}', simple_test_case, "model2", "cond1"),
        ]

        scorer = GatingScorer()
        results = scorer.score_batch(responses)

        assert len(results) == 2
        assert all(r.parse_success for r in results)


class TestScorePrediction:
    """Tests for score_prediction convenience function."""

    def test_score_prediction(self, simple_test_case):
        """Test the convenience function."""
        response = '{"name": "All Events", "children": []}'

        result = score_prediction(response, simple_test_case)

        assert isinstance(result, ScoringResult)
        assert result.parse_success


class TestComputeAggregateMetrics:
    """Tests for aggregate metrics computation."""

    def test_aggregate_empty(self):
        """Test with empty results."""
        metrics = compute_aggregate_metrics([])

        assert "error" in metrics

    def test_aggregate_valid_results(self, simple_test_case):
        """Test aggregating valid results."""
        scorer = GatingScorer()

        results = [
            scorer.score('{"name": "All Events", "children": []}', simple_test_case, "m1", "c1"),
            scorer.score('{"name": "All Events", "children": []}', simple_test_case, "m1", "c2"),
        ]

        metrics = compute_aggregate_metrics(results)

        assert metrics["total"] == 2
        assert metrics["valid"] == 2
        assert "hierarchy_f1_mean" in metrics

    def test_aggregate_with_failures(self, simple_test_case):
        """Test aggregating with parse failures."""
        scorer = GatingScorer()

        results = [
            scorer.score('{"name": "All Events", "children": []}', simple_test_case, "m1", "c1"),
            scorer.score('invalid response', simple_test_case, "m1", "c2"),
        ]

        metrics = compute_aggregate_metrics(results)

        assert metrics["total"] == 2
        assert metrics["valid"] == 1
        assert metrics["parse_success_rate"] == 0.5
