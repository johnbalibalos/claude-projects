"""
Tests for the hypothesis testing framework.

These tests validate the core functionality of each hypothesis test module
without requiring actual LLM calls.
"""

from __future__ import annotations

import pytest
from datetime import date

from curation.schemas import (
    CurationMetadata,
    ExperimentContext,
    GateNode,
    GatingHierarchy,
    MarkerExpression,
    Panel,
    PanelEntry,
    SourceType,
    TestCase,
)

# Import hypothesis test modules
from hypothesis_tests.frequency_confound import (
    AlienCellTest,
    AlienCellMapping,
    FrequencyCorrelation,
    _pearson_correlation,
    _safe_log,
)
from hypothesis_tests.format_ablation import (
    FormatAblationResult,
    FormatAblationTest,
    PromptFormat,
)
from hypothesis_tests.cot_mechanistic import (
    CoTAnnotator,
    InferenceTag,
    ReasoningStep,
)
from hypothesis_tests.cognitive_refusal import (
    CognitiveRefusalTest,
    RefusalType,
    PROMPT_VARIANTS,
)
from hypothesis_tests.runner import (
    AblationConfig,
    HypothesisTestRunner,
    HypothesisType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_test_case() -> TestCase:
    """Create a sample test case for testing."""
    return TestCase(
        test_case_id="TEST-001",
        source_type=SourceType.SYNTHETIC,
        context=ExperimentContext(
            sample_type="Human PBMC",
            species="human",
            application="Immunophenotyping",
        ),
        panel=Panel(
            entries=[
                PanelEntry(marker="CD3", fluorophore="BUV395", clone="UCHT1"),
                PanelEntry(marker="CD4", fluorophore="BUV496", clone="SK3"),
                PanelEntry(marker="CD8", fluorophore="BUV661", clone="SK1"),
                PanelEntry(marker="CD19", fluorophore="BV605", clone="SJ25C1"),
                PanelEntry(marker="CD45", fluorophore="BV421", clone="HI30"),
                PanelEntry(marker="Live/Dead", fluorophore="Zombie NIR", clone=None),
            ]
        ),
        gating_hierarchy=GatingHierarchy(
            root=GateNode(
                name="All Events",
                children=[
                    GateNode(
                        name="Singlets",
                        markers=["FSC-A", "FSC-H"],
                        children=[
                            GateNode(
                                name="Live",
                                markers=["Zombie NIR"],
                                children=[
                                    GateNode(
                                        name="T cells",
                                        marker_logic=[
                                            MarkerExpression(marker="CD3", positive=True),
                                            MarkerExpression(marker="CD19", positive=False),
                                        ],
                                        children=[
                                            GateNode(
                                                name="CD4+ T cells",
                                                marker_logic=[
                                                    MarkerExpression(marker="CD4", positive=True),
                                                    MarkerExpression(marker="CD8", positive=False),
                                                ],
                                            ),
                                            GateNode(
                                                name="CD8+ T cells",
                                                marker_logic=[
                                                    MarkerExpression(marker="CD4", positive=False),
                                                    MarkerExpression(marker="CD8", positive=True),
                                                ],
                                            ),
                                        ],
                                    ),
                                    GateNode(
                                        name="B cells",
                                        marker_logic=[
                                            MarkerExpression(marker="CD3", positive=False),
                                            MarkerExpression(marker="CD19", positive=True),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ),
        metadata=CurationMetadata(
            curation_date=date.today(),
            curator="Test",
        ),
    )


# =============================================================================
# FREQUENCY CONFOUND TESTS
# =============================================================================


class TestFrequencyConfound:
    """Tests for frequency confound analysis."""

    def test_safe_log(self):
        """Test safe logarithm calculation."""
        assert _safe_log(0) == 0.0
        assert _safe_log(-1) == 0.0
        assert _safe_log(1) > 0
        assert _safe_log(100) > _safe_log(10)

    def test_pearson_correlation_perfect(self):
        """Test Pearson correlation with perfectly correlated data."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r, p = _pearson_correlation(x, y)
        assert abs(r - 1.0) < 0.001
        assert p < 0.05

    def test_pearson_correlation_negative(self):
        """Test Pearson correlation with negatively correlated data."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        r, p = _pearson_correlation(x, y)
        assert abs(r - (-1.0)) < 0.001

    def test_pearson_correlation_insufficient_data(self):
        """Test Pearson correlation with insufficient data."""
        x = [1.0, 2.0]
        y = [3.0, 4.0]
        r, p = _pearson_correlation(x, y)
        assert r == 0.0
        assert p == 1.0


class TestAlienCellTest:
    """Tests for alien cell injection."""

    def test_alien_cell_creation(self, sample_test_case):
        """Test creating an alien cell test case."""
        test = AlienCellTest()
        alien_case = test.create_alien_test_case(sample_test_case)

        # Check that mappings were created
        assert len(alien_case.mappings) > 0

        # Check that T cells was renamed
        t_cell_mapping = next(
            (m for m in alien_case.mappings if m.original_name == "T cells"),
            None,
        )
        assert t_cell_mapping is not None
        assert t_cell_mapping.alien_name != "T cells"

    def test_preserved_gates(self, sample_test_case):
        """Test that critical gates are preserved."""
        test = AlienCellTest(preserve_critical_gates=True)
        alien_case = test.create_alien_test_case(sample_test_case)

        # All Events should be preserved
        all_events_mapping = next(
            (m for m in alien_case.mappings if m.original_name == "All Events"),
            None,
        )
        assert all_events_mapping is None  # Should not be in mappings

        # Singlets should be preserved
        singlets_mapping = next(
            (m for m in alien_case.mappings if m.original_name == "Singlets"),
            None,
        )
        assert singlets_mapping is None

    def test_mapping_context_generated(self, sample_test_case):
        """Test that mapping context is generated."""
        test = AlienCellTest()
        alien_case = test.create_alien_test_case(
            sample_test_case,
            include_mapping_context=True,
        )

        assert alien_case.mapping_context != ""
        assert "Cell Population Definitions" in alien_case.mapping_context


# =============================================================================
# FORMAT ABLATION TESTS
# =============================================================================


class TestFormatAblation:
    """Tests for format ablation."""

    def test_generate_all_formats(self, sample_test_case):
        """Test generating all format variants."""
        test = FormatAblationTest()
        formats = test.generate_all_formats(sample_test_case)

        assert PromptFormat.PROSE in formats
        assert PromptFormat.TABLE in formats
        assert PromptFormat.PSEUDOCODE in formats

    def test_prose_format(self, sample_test_case):
        """Test prose format generation."""
        test = FormatAblationTest()
        prompt = test.format_as_prose(sample_test_case)

        assert prompt.format == PromptFormat.PROSE
        assert "Gating Strategy" in prompt.content
        assert "PBMC" in prompt.content or "Human" in prompt.content

    def test_table_format(self, sample_test_case):
        """Test table format generation."""
        test = FormatAblationTest()
        prompt = test.format_as_table(sample_test_case)

        assert prompt.format == PromptFormat.TABLE
        assert "| Marker |" in prompt.content
        assert "CD3" in prompt.content

    def test_pseudocode_format(self, sample_test_case):
        """Test pseudocode format generation."""
        test = FormatAblationTest()
        prompt = test.format_as_pseudocode(sample_test_case)

        assert prompt.format == PromptFormat.PSEUDOCODE
        assert "def gate_sample" in prompt.content
        assert "MARKERS" in prompt.content

    def test_analysis_robust_failure(self):
        """Test analysis detecting robust failure."""
        test = FormatAblationTest()

        # All formats fail
        results = {
            PromptFormat.PROSE: FormatAblationResult(
                format=PromptFormat.PROSE,
                f1_score=0.1,
                structure_accuracy=0.1,
                parse_success=True,
                task_failure=False,
            ),
            PromptFormat.TABLE: FormatAblationResult(
                format=PromptFormat.TABLE,
                f1_score=0.15,
                structure_accuracy=0.12,
                parse_success=True,
                task_failure=False,
            ),
            PromptFormat.PSEUDOCODE: FormatAblationResult(
                format=PromptFormat.PSEUDOCODE,
                f1_score=0.12,
                structure_accuracy=0.1,
                parse_success=True,
                task_failure=False,
            ),
        }

        analysis = test.analyze_results(results, "TEST-001")
        assert analysis.is_robust_failure is True

    def test_analysis_extraction_issue(self):
        """Test analysis detecting extraction issue."""
        test = FormatAblationTest()

        # Structured formats much better than prose
        results = {
            PromptFormat.PROSE: FormatAblationResult(
                format=PromptFormat.PROSE,
                f1_score=0.2,
                structure_accuracy=0.15,
                parse_success=True,
                task_failure=False,
            ),
            PromptFormat.TABLE: FormatAblationResult(
                format=PromptFormat.TABLE,
                f1_score=0.5,
                structure_accuracy=0.45,
                parse_success=True,
                task_failure=False,
            ),
            PromptFormat.PSEUDOCODE: FormatAblationResult(
                format=PromptFormat.PSEUDOCODE,
                f1_score=0.55,
                structure_accuracy=0.5,
                parse_success=True,
                task_failure=False,
            ),
        }

        analysis = test.analyze_results(results, "TEST-001")
        assert analysis.is_extraction_issue is True


# =============================================================================
# COT MECHANISTIC TESTS
# =============================================================================


class TestCoTMechanistic:
    """Tests for CoT mechanistic analysis."""

    def test_segment_numbered_steps(self):
        """Test segmentation of numbered steps."""
        annotator = CoTAnnotator()

        cot = """1. First, I'll identify the lineage markers.
2. CD3 is typically a T cell marker.
3. CD19 is a B cell marker.
4. I'll gate on CD3+ for T cells."""

        steps = annotator.segment_cot(cot)
        assert len(steps) == 4

    def test_segment_reasoning_indicators(self):
        """Test segmentation by reasoning indicators."""
        annotator = CoTAnnotator()

        cot = """First, I need to look at the markers. The panel includes CD3 and CD19.
Therefore, I can identify T and B cells. Based on this, I'll create the hierarchy."""

        steps = annotator.segment_cot(cot)
        assert len(steps) >= 2

    def test_classify_structural_step(self, sample_test_case):
        """Test classification of structural/meta statements."""
        annotator = CoTAnnotator()

        step = annotator.classify_step(
            "Let me think about the best gating strategy.",
            sample_test_case,
        )
        assert step.tag == InferenceTag.STRUCTURAL

    def test_classify_prior_hallucination(self, sample_test_case):
        """Test classification of prior hallucinations."""
        annotator = CoTAnnotator()

        step = annotator.classify_step(
            "CD3 is typically a T cell marker in immunology.",
            sample_test_case,
        )
        assert step.tag == InferenceTag.PRIOR_HALLUCINATION
        assert step.invoked_prior is not None

    def test_full_analysis(self, sample_test_case):
        """Test full CoT analysis."""
        annotator = CoTAnnotator()

        cot = """Let me analyze this panel.
1. Looking at the markers, I see CD3, CD4, CD8, CD19.
2. CD3 is typically used to identify T cells.
3. The panel includes a Live/Dead marker for viability.
4. I'll create the gating hierarchy."""

        analysis = annotator.analyze_cot(cot, sample_test_case)

        assert len(analysis.steps) >= 4
        assert analysis.n_structural > 0 or analysis.n_prior_hallucinations > 0


# =============================================================================
# COGNITIVE REFUSAL TESTS
# =============================================================================


class TestCognitiveRefusal:
    """Tests for cognitive refusal analysis."""

    def test_prompt_variants_exist(self):
        """Test that all prompt variants are defined."""
        assert "permissive" in PROMPT_VARIANTS
        assert "standard" in PROMPT_VARIANTS
        assert "assertive" in PROMPT_VARIANTS
        assert "gun_to_head" in PROMPT_VARIANTS

    def test_detect_clarifying_question(self):
        """Test detection of clarifying questions."""
        test = CognitiveRefusalTest()

        response = "Could you please provide more information about the sample type?"
        refusal = test.detect_refusal(response)
        assert refusal.refusal_type == RefusalType.CLARIFYING_QUESTION

    def test_detect_explicit_refusal(self):
        """Test detection of explicit refusal."""
        test = CognitiveRefusalTest()

        response = "I cannot provide a gating hierarchy without more context."
        refusal = test.detect_refusal(response)
        assert refusal.refusal_type == RefusalType.EXPLICIT_REFUSAL

    def test_detect_uncertainty_hedging(self):
        """Test detection of uncertainty hedging."""
        test = CognitiveRefusalTest()

        response = """I'm not sure about this, but here's my best guess:
{
    "name": "All Events",
    "children": []
}
This is just a tentative prediction."""

        refusal = test.detect_refusal(response)
        assert refusal.refusal_type == RefusalType.UNCERTAINTY_HEDGING

    def test_no_refusal_detected(self):
        """Test that valid responses are not flagged as refusals."""
        test = CognitiveRefusalTest()

        response = """{
    "name": "All Events",
    "children": [
        {"name": "Singlets", "children": []}
    ]
}"""

        refusal = test.detect_refusal(response)
        assert refusal.refusal_type == RefusalType.NONE

    def test_gun_to_head_prompt(self):
        """Test that gun-to-head prompt is more forceful."""
        test = CognitiveRefusalTest()

        system, _ = test.build_prompt_for_variant("Test prompt", "gun_to_head")

        assert "MUST" in system
        assert "MUST NOT ask" in system.lower() or "must not ask" in system.lower()


# =============================================================================
# RUNNER TESTS
# =============================================================================


class TestHypothesisTestRunner:
    """Tests for the hypothesis test runner."""

    def test_config_creation(self):
        """Test creating an ablation config."""
        config = AblationConfig(
            tests=[HypothesisType.FREQUENCY_CONFOUND, HypothesisType.ALIEN_CELL],
            model="claude-sonnet-4-20250514",
            output_dir="./test_output",
        )

        assert len(config.tests) == 2
        assert config.model == "claude-sonnet-4-20250514"

    def test_runner_initialization(self):
        """Test runner initialization."""
        config = AblationConfig(
            tests=[HypothesisType.FORMAT_ABLATION],
            dry_run=True,
        )

        runner = HypothesisTestRunner(config)

        assert runner.format_ablation is not None
        assert runner.alien_cell_test is None  # Not configured

    def test_runner_selective_tests(self):
        """Test that runner only initializes requested tests."""
        config = AblationConfig(
            tests=[HypothesisType.COT_MECHANISTIC],
        )

        runner = HypothesisTestRunner(config)

        assert runner.cot_annotator is not None
        assert runner.frequency_correlation is None
        assert runner.alien_cell_test is None
        assert runner.format_ablation is None
        assert runner.cognitive_refusal_test is None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_alien_cell_workflow(self, sample_test_case):
        """Test the complete alien cell workflow."""
        # Create alien test case
        alien_test = AlienCellTest()
        alien_case = alien_test.create_alien_test_case(sample_test_case)

        # Verify the modified test case has alien names
        modified_gates = alien_case.modified_test_case.gating_hierarchy.get_all_gates()

        # At least some gates should be renamed (excluding preserved ones)
        renamed_count = sum(
            1 for m in alien_case.mappings
            if m.alien_name != m.original_name
        )
        assert renamed_count > 0

    def test_format_ablation_with_analysis(self, sample_test_case):
        """Test format ablation with full analysis."""
        test = FormatAblationTest(formats=[
            PromptFormat.PROSE,
            PromptFormat.TABLE,
        ])

        formats = test.generate_all_formats(sample_test_case)

        # Create mock results
        results = {
            PromptFormat.PROSE: FormatAblationResult(
                format=PromptFormat.PROSE,
                f1_score=0.3,
                structure_accuracy=0.25,
                parse_success=True,
                task_failure=False,
            ),
            PromptFormat.TABLE: FormatAblationResult(
                format=PromptFormat.TABLE,
                f1_score=0.4,
                structure_accuracy=0.35,
                parse_success=True,
                task_failure=False,
            ),
        }

        analysis = test.analyze_results(results, sample_test_case.test_case_id)

        assert analysis.best_format == PromptFormat.TABLE
        assert analysis.worst_format == PromptFormat.PROSE
        assert "interpretation" in analysis.interpretation.lower() or len(analysis.interpretation) > 0
