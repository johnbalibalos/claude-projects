"""Integration tests for the gating benchmark."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.experiments.runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)
from src.experiments.conditions import (
    ExperimentCondition,
    get_minimal_conditions,
    get_ablation_conditions,
    MODELS,
)
from src.experiments.prompts import (
    build_prompt,
    format_context_minimal,
    format_context_standard,
    format_context_rich,
    PROMPT_TEMPLATES,
)
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
from src.evaluation.scorer import GatingScorer, ScoringResult


class TestExperimentConditions:
    """Tests for experimental conditions."""

    def test_minimal_conditions(self):
        """Test minimal condition set."""
        conditions = get_minimal_conditions()

        assert len(conditions) == 2
        assert all(c.model == MODELS["claude-sonnet"] for c in conditions)

    def test_ablation_conditions(self):
        """Test ablation condition set."""
        conditions = get_ablation_conditions()

        assert len(conditions) == 6  # 3 context × 2 strategy
        assert all(c.model == MODELS["claude-sonnet"] for c in conditions)

        # Check all context levels present
        context_levels = {c.context_level for c in conditions}
        assert context_levels == {"minimal", "standard", "rich"}

        # Check all strategies present
        strategies = {c.prompt_strategy for c in conditions}
        assert strategies == {"direct", "cot"}

    def test_condition_ids_unique(self):
        """Test that condition IDs are unique."""
        conditions = get_ablation_conditions()
        ids = [c.condition_id for c in conditions]

        assert len(ids) == len(set(ids))


class TestPromptBuilding:
    """Tests for prompt construction."""

    @pytest.fixture
    def sample_test_case(self):
        """Create a sample test case."""
        return TestCase(
            test_case_id="TEST-001",
            source_type=SourceType.OMIP_PAPER,
            omip_id="OMIP-001",
            context=ExperimentContext(
                sample_type="Human PBMC",
                species="human",
                application="T cell phenotyping",
                tissue="Peripheral blood",
                additional_notes="Test panel",
            ),
            panel=Panel(
                entries=[
                    PanelEntry(marker="CD3", fluorophore="BV421", clone="UCHT1"),
                    PanelEntry(marker="CD4", fluorophore="BV510", clone="SK3"),
                    PanelEntry(marker="CD8", fluorophore="BV650", clone="SK1"),
                    PanelEntry(marker="Live/Dead", fluorophore="Zombie NIR"),
                ]
            ),
            gating_hierarchy=GatingHierarchy(
                root=GateNode(
                    name="All Events",
                    children=[
                        GateNode(name="Singlets", is_critical=True, children=[])
                    ],
                )
            ),
            metadata=CurationMetadata(
                curation_date="2026-01-07",
                curator="Test",
            ),
        )

    def test_build_prompt_direct(self, sample_test_case):
        """Test building direct prompt."""
        prompt = build_prompt(
            sample_test_case,
            template_name="direct",
            context_level="standard",
        )

        assert "expert flow cytometrist" in prompt
        assert "CD3" in prompt
        assert "CD4" in prompt
        assert "JSON" in prompt

    def test_build_prompt_cot(self, sample_test_case):
        """Test building chain-of-thought prompt."""
        prompt = build_prompt(
            sample_test_case,
            template_name="cot",
            context_level="standard",
        )

        assert "step-by-step" in prompt.lower()
        assert "Quality Control Gates" in prompt
        assert "Major Lineage" in prompt

    def test_context_minimal(self, sample_test_case):
        """Test minimal context formatting."""
        context = format_context_minimal(sample_test_case)

        assert "CD3" in context
        assert "CD4" in context
        # Should NOT have sample type in minimal
        assert "Human PBMC" not in context

    def test_context_standard(self, sample_test_case):
        """Test standard context formatting."""
        context = format_context_standard(sample_test_case)

        assert "Human PBMC" in context
        assert "human" in context
        assert "T cell phenotyping" in context
        assert "CD3: BV421" in context

    def test_context_rich(self, sample_test_case):
        """Test rich context formatting."""
        context = format_context_rich(sample_test_case)

        assert "Human PBMC" in context
        assert "Panel Size:" in context
        assert "Complexity:" in context
        assert "Test panel" in context


class TestGatingScorer:
    """Tests for the gating scorer."""

    @pytest.fixture
    def scorer(self):
        """Create a scorer instance."""
        return GatingScorer()

    @pytest.fixture
    def sample_test_case(self):
        """Create a sample test case with hierarchy."""
        return TestCase(
            test_case_id="TEST-001",
            source_type=SourceType.OMIP_PAPER,
            context=ExperimentContext(
                sample_type="Human PBMC",
                species="human",
                application="Test",
            ),
            panel=Panel(
                entries=[
                    PanelEntry(marker="CD3", fluorophore="BV421"),
                    PanelEntry(marker="CD4", fluorophore="BV510"),
                    PanelEntry(marker="Live/Dead", fluorophore="Zombie NIR"),
                ]
            ),
            gating_hierarchy=GatingHierarchy(
                root=GateNode(
                    name="All Events",
                    children=[
                        GateNode(
                            name="Singlets",
                            is_critical=True,
                            markers=["FSC-A", "FSC-H"],
                            children=[
                                GateNode(
                                    name="Live",
                                    is_critical=True,
                                    markers=["Zombie NIR"],
                                    children=[
                                        GateNode(
                                            name="T cells",
                                            markers=["CD3"],
                                            children=[
                                                GateNode(name="CD4+", markers=["CD4"], children=[]),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ),
            metadata=CurationMetadata(
                curation_date="2026-01-07",
                curator="Test",
            ),
        )

    def test_score_valid_json(self, scorer, sample_test_case):
        """Test scoring with valid JSON response."""
        response = json.dumps({
            "name": "All Events",
            "children": [
                {
                    "name": "Singlets",
                    "children": [
                        {
                            "name": "Live",
                            "children": [
                                {"name": "T cells", "children": []}
                            ]
                        }
                    ]
                }
            ]
        })

        result = scorer.score(
            response=response,
            test_case=sample_test_case,
            model="test-model",
            condition="test",
        )

        assert result.parse_success
        assert result.hierarchy_f1 > 0
        assert result.critical_gate_recall > 0

    def test_score_invalid_json(self, scorer, sample_test_case):
        """Test scoring with invalid JSON response."""
        response = "This is not valid JSON at all"

        result = scorer.score(
            response=response,
            test_case=sample_test_case,
            model="test-model",
            condition="test",
        )

        # Should try markdown parsing fallback
        assert isinstance(result, ScoringResult)

    def test_score_markdown_fallback(self, scorer, sample_test_case):
        """Test scoring with markdown list format."""
        response = """Here is the gating hierarchy:

- All Events
  - Singlets
    - Live
      - T cells
        - CD4+"""

        result = scorer.score(
            response=response,
            test_case=sample_test_case,
            model="test-model",
            condition="test",
        )

        # Should parse markdown format
        assert isinstance(result, ScoringResult)


class TestExperimentRunner:
    """Tests for the experiment runner."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config."""
        # Create test case file
        test_cases_dir = tmp_path / "test_cases"
        test_cases_dir.mkdir()

        test_case = {
            "test_case_id": "TEST-001",
            "source_type": "omip_paper",
            "context": {
                "sample_type": "Human PBMC",
                "species": "human",
                "application": "Test",
            },
            "panel": {
                "entries": [
                    {"marker": "CD3", "fluorophore": "BV421"},
                ]
            },
            "gating_hierarchy": {
                "root": {"name": "All Events", "children": []}
            },
            "metadata": {
                "curation_date": "2026-01-07",
                "curator": "Test",
            },
        }

        with open(test_cases_dir / "test_001.json", "w") as f:
            json.dump(test_case, f)

        return ExperimentConfig(
            name="test_experiment",
            test_cases_dir=str(test_cases_dir),
            output_dir=str(tmp_path / "output"),
            conditions=[
                ExperimentCondition(
                    name="test_condition",
                    model="claude-sonnet-4-20250514",
                    context_level="minimal",
                    prompt_strategy="direct",
                )
            ],
            dry_run=True,
        )

    def test_runner_initialization(self, mock_config):
        """Test runner initialization."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            runner = ExperimentRunner(mock_config)

            assert runner.config == mock_config
            assert runner.scorer is not None

    def test_dry_run(self, mock_config):
        """Test dry run mode."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            runner = ExperimentRunner(mock_config)
            result = runner.run()

            assert isinstance(result, ExperimentResult)
            assert result.config == mock_config


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_mock(self, tmp_path):
        """Test full pipeline with mock API."""
        # Create test data
        test_cases_dir = tmp_path / "test_cases"
        test_cases_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create a test case
        test_case = {
            "test_case_id": "OMIP-TEST",
            "source_type": "omip_paper",
            "omip_id": "OMIP-TEST",
            "context": {
                "sample_type": "Human PBMC",
                "species": "human",
                "application": "Test",
            },
            "panel": {
                "entries": [
                    {"marker": "CD3", "fluorophore": "BV421", "clone": "UCHT1"},
                    {"marker": "CD4", "fluorophore": "BV510", "clone": "SK3"},
                    {"marker": "Live/Dead", "fluorophore": "Zombie NIR"},
                ]
            },
            "gating_hierarchy": {
                "root": {
                    "name": "All Events",
                    "children": [
                        {
                            "name": "Singlets",
                            "is_critical": True,
                            "children": [
                                {
                                    "name": "Live",
                                    "is_critical": True,
                                    "children": []
                                }
                            ]
                        }
                    ]
                }
            },
            "metadata": {
                "curation_date": "2026-01-07",
                "curator": "Test",
            },
        }

        with open(test_cases_dir / "omip_test.json", "w") as f:
            json.dump(test_case, f)

        # Run with dry_run=True
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
            result = run_experiment(
                test_cases_dir=str(test_cases_dir),
                output_dir=str(output_dir),
                name="test",
                dry_run=True,
            )

        assert isinstance(result, ExperimentResult)
        assert output_dir.exists()
