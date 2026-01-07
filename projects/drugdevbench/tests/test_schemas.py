"""Tests for data schemas."""

import pytest
from datetime import datetime

from drugdevbench.data.schemas import (
    FigureType,
    Persona,
    PromptCondition,
    QuestionType,
    Question,
    Figure,
    Annotation,
    EvaluationResponse,
    BenchmarkResult,
    AblationResult,
    get_persona_for_figure_type,
    FIGURE_TYPE_TO_PERSONA,
)


class TestEnums:
    """Test enum definitions."""

    def test_figure_type_values(self):
        """All figure types should have valid string values."""
        assert FigureType.WESTERN_BLOT.value == "western_blot"
        assert FigureType.DOSE_RESPONSE.value == "dose_response"
        assert FigureType.PK_CURVE.value == "pk_curve"
        assert FigureType.FLOW_BIAXIAL.value == "flow_biaxial"
        assert FigureType.HEATMAP.value == "heatmap"

    def test_persona_values(self):
        """All personas should have valid string values."""
        assert Persona.IMMUNOLOGIST.value == "immunologist"
        assert Persona.PHARMACOLOGIST.value == "pharmacologist"
        assert Persona.BIOANALYTICAL_SCIENTIST.value == "bioanalytical_scientist"
        assert Persona.MOLECULAR_BIOLOGIST.value == "molecular_biologist"
        assert Persona.COMPUTATIONAL_BIOLOGIST.value == "computational_biologist"

    def test_prompt_condition_values(self):
        """All prompt conditions should have valid string values."""
        assert PromptCondition.VANILLA.value == "vanilla"
        assert PromptCondition.BASE_ONLY.value == "base_only"
        assert PromptCondition.FULL_STACK.value == "full_stack"

    def test_question_type_values(self):
        """All question types should have valid string values."""
        assert QuestionType.FACTUAL_EXTRACTION.value == "factual_extraction"
        assert QuestionType.VISUAL_ESTIMATION.value == "visual_estimation"
        assert QuestionType.QUALITY_ASSESSMENT.value == "quality_assessment"


class TestQuestion:
    """Test Question model."""

    def test_question_creation(self):
        """Question should be created with required fields."""
        q = Question(
            question_id="q_001",
            figure_id="fig_001",
            question_text="What is the IC50?",
            question_type=QuestionType.FACTUAL_EXTRACTION,
            gold_answer="2.5 nM",
        )
        assert q.question_id == "q_001"
        assert q.gold_answer == "2.5 nM"
        assert q.difficulty == "intermediate"  # default

    def test_question_with_optional_fields(self):
        """Question should accept optional fields."""
        q = Question(
            question_id="q_002",
            figure_id="fig_001",
            question_text="What is the Tmax?",
            question_type=QuestionType.VISUAL_ESTIMATION,
            gold_answer="2 hours",
            gold_answer_source="Figure legend",
            difficulty="basic",
            metadata={"source": "auto_generated"},
        )
        assert q.gold_answer_source == "Figure legend"
        assert q.difficulty == "basic"
        assert q.metadata["source"] == "auto_generated"


class TestFigure:
    """Test Figure model."""

    def test_figure_creation(self):
        """Figure should be created with required fields."""
        fig = Figure(
            figure_id="fig_001",
            figure_type=FigureType.WESTERN_BLOT,
            image_path="data/figures/western_blots/fig_001.png",
        )
        assert fig.figure_id == "fig_001"
        assert fig.figure_type == FigureType.WESTERN_BLOT
        assert fig.source == "unknown"  # default

    def test_figure_with_metadata(self):
        """Figure should accept paper metadata."""
        fig = Figure(
            figure_id="fig_002",
            figure_type=FigureType.PK_CURVE,
            image_path="data/figures/pk_curves/fig_002.png",
            legend_text="Figure 2. PK profile of compound X",
            paper_doi="10.1234/example",
            paper_title="Example Paper",
            source="biorxiv",
        )
        assert fig.legend_text is not None
        assert fig.paper_doi == "10.1234/example"
        assert fig.source == "biorxiv"


class TestAnnotation:
    """Test Annotation model."""

    def test_annotation_creation(self):
        """Annotation should combine figure and questions."""
        fig = Figure(
            figure_id="fig_001",
            figure_type=FigureType.DOSE_RESPONSE,
            image_path="data/figures/dose_response/fig_001.png",
        )
        q = Question(
            question_id="q_001",
            figure_id="fig_001",
            question_text="What is the EC50?",
            question_type=QuestionType.FACTUAL_EXTRACTION,
            gold_answer="10 nM",
        )
        annotation = Annotation(figure=fig, questions=[q])
        assert annotation.figure.figure_id == "fig_001"
        assert len(annotation.questions) == 1


class TestEvaluationResponse:
    """Test EvaluationResponse model."""

    def test_evaluation_response_creation(self):
        """EvaluationResponse should capture model output."""
        response = EvaluationResponse(
            figure_id="fig_001",
            question_id="q_001",
            model="claude-haiku",
            condition=PromptCondition.FULL_STACK,
            response_text="The IC50 is approximately 2.5 nM",
            gold_answer="2.5 nM",
        )
        assert response.model == "claude-haiku"
        assert response.cached is False  # default

    def test_evaluation_response_with_metrics(self):
        """EvaluationResponse should accept evaluation metrics."""
        response = EvaluationResponse(
            figure_id="fig_001",
            question_id="q_001",
            model="claude-sonnet",
            condition=PromptCondition.BASE_ONLY,
            response_text="The IC50 is 2.5 nM",
            gold_answer="2.5 nM",
            score=1.0,
            scoring_rationale="Exact match",
            prompt_tokens=1500,
            completion_tokens=50,
            cost_usd=0.005,
            latency_ms=1200,
        )
        assert response.score == 1.0
        assert response.cost_usd == 0.005


class TestFigureTypePersonaMapping:
    """Test figure type to persona mapping."""

    def test_all_figure_types_have_persona(self):
        """Every figure type should map to a persona."""
        for ft in FigureType:
            assert ft in FIGURE_TYPE_TO_PERSONA
            persona = FIGURE_TYPE_TO_PERSONA[ft]
            assert isinstance(persona, Persona)

    def test_western_blot_maps_to_molecular_biologist(self):
        """Western blots should map to molecular biologist."""
        assert get_persona_for_figure_type(FigureType.WESTERN_BLOT) == Persona.MOLECULAR_BIOLOGIST

    def test_flow_cytometry_maps_to_immunologist(self):
        """Flow cytometry figures should map to immunologist."""
        assert get_persona_for_figure_type(FigureType.FLOW_BIAXIAL) == Persona.IMMUNOLOGIST
        assert get_persona_for_figure_type(FigureType.FLOW_HISTOGRAM) == Persona.IMMUNOLOGIST

    def test_pk_curve_maps_to_pharmacologist(self):
        """PK curves should map to pharmacologist."""
        assert get_persona_for_figure_type(FigureType.PK_CURVE) == Persona.PHARMACOLOGIST
