"""Pydantic schemas for DrugDevBench data structures."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FigureType(str, Enum):
    """Types of figures in drug development research."""

    # Protein Analysis
    WESTERN_BLOT = "western_blot"
    COOMASSIE_GEL = "coomassie_gel"
    DOT_BLOT = "dot_blot"

    # Binding & Activity Assays
    ELISA = "elisa"
    DOSE_RESPONSE = "dose_response"
    IC50_EC50 = "ic50_ec50"

    # Pharmacokinetics
    PK_CURVE = "pk_curve"
    AUC_PLOT = "auc_plot"
    COMPARTMENT_MODEL = "compartment_model"

    # Flow Cytometry
    FLOW_BIAXIAL = "flow_biaxial"
    FLOW_HISTOGRAM = "flow_histogram"
    GATING_STRATEGY = "gating_strategy"

    # Genomics/Transcriptomics
    HEATMAP = "heatmap"
    VOLCANO_PLOT = "volcano_plot"
    PATHWAY_ENRICHMENT = "pathway_enrichment"

    # Cell-Based Assays
    VIABILITY_CURVE = "viability_curve"
    PROLIFERATION = "proliferation"
    CYTOTOXICITY = "cytotoxicity"


class Persona(str, Enum):
    """Domain expert personas for figure interpretation."""

    IMMUNOLOGIST = "immunologist"
    PHARMACOLOGIST = "pharmacologist"
    BIOANALYTICAL_SCIENTIST = "bioanalytical_scientist"
    MOLECULAR_BIOLOGIST = "molecular_biologist"
    COMPUTATIONAL_BIOLOGIST = "computational_biologist"
    CELL_BIOLOGIST = "cell_biologist"


class PromptCondition(str, Enum):
    """Ablation conditions for prompt construction."""

    VANILLA = "vanilla"  # No additional prompting
    BASE_ONLY = "base_only"  # Generic scientific reasoning only
    PERSONA_ONLY = "persona_only"  # Domain persona only
    BASE_PLUS_SKILL = "base_plus_skill"  # Base + figure-type skill
    FULL_STACK = "full_stack"  # Persona + base + skill
    WRONG_SKILL = "wrong_skill"  # Base + mismatched skill (negative control)


class QuestionType(str, Enum):
    """Types of questions about figures."""

    FACTUAL_EXTRACTION = "factual_extraction"  # "What is the reported EC50?"
    VISUAL_ESTIMATION = "visual_estimation"  # "Estimate the half-life from the curve"
    QUALITY_ASSESSMENT = "quality_assessment"  # "Is the loading control appropriate?"
    INTERPRETATION = "interpretation"  # "What does this suggest about drug clearance?"
    ERROR_DETECTION = "error_detection"  # "Are there any concerns with this blot?"


class Question(BaseModel):
    """A question about a figure."""

    question_id: str = Field(..., description="Unique identifier for the question")
    figure_id: str = Field(..., description="ID of the associated figure")
    question_text: str = Field(..., description="The question to ask about the figure")
    question_type: QuestionType = Field(..., description="Type of question")
    gold_answer: str = Field(..., description="Expected correct answer")
    gold_answer_source: str | None = Field(
        None, description="Source of the gold answer (e.g., figure legend)"
    )
    difficulty: str = Field(
        "intermediate", description="Difficulty level: basic, intermediate, expert"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class Figure(BaseModel):
    """A figure from a drug development paper."""

    figure_id: str = Field(..., description="Unique identifier for the figure")
    figure_type: FigureType = Field(..., description="Type of figure")
    image_path: str = Field(..., description="Path to the figure image file")
    legend_text: str | None = Field(None, description="Figure legend text")
    results_text: str | None = Field(None, description="Related results section text")
    paper_doi: str | None = Field(None, description="DOI of the source paper")
    paper_title: str | None = Field(None, description="Title of the source paper")
    source: str = Field("unknown", description="Source: biorxiv, pubmed, manual")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Annotation(BaseModel):
    """Complete annotation for a figure including questions."""

    figure: Figure = Field(..., description="The annotated figure")
    questions: list[Question] = Field(default_factory=list, description="Questions about the figure")
    annotator: str | None = Field(None, description="Who created the annotation")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class EvaluationResponse(BaseModel):
    """Response from an LLM evaluation."""

    figure_id: str = Field(..., description="ID of the evaluated figure")
    question_id: str = Field(..., description="ID of the question answered")
    model: str = Field(..., description="Model used for evaluation")
    condition: PromptCondition = Field(..., description="Prompt condition used")
    response_text: str = Field(..., description="Model's response")
    gold_answer: str = Field(..., description="Expected correct answer")
    score: float | None = Field(None, description="Score (0-1) if evaluated")
    scoring_rationale: str | None = Field(None, description="Explanation of score")
    prompt_tokens: int | None = Field(None, description="Input tokens used")
    completion_tokens: int | None = Field(None, description="Output tokens used")
    cost_usd: float | None = Field(None, description="Estimated cost in USD")
    latency_ms: int | None = Field(None, description="Response latency in milliseconds")
    cached: bool = Field(False, description="Whether response was from cache")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Aggregated results for a benchmark run."""

    run_id: str = Field(..., description="Unique identifier for this run")
    model: str = Field(..., description="Model evaluated")
    condition: PromptCondition = Field(..., description="Prompt condition")
    figure_type: FigureType | None = Field(None, description="Figure type filter if any")
    n_figures: int = Field(..., description="Number of figures evaluated")
    n_questions: int = Field(..., description="Number of questions answered")
    mean_score: float = Field(..., description="Mean score across all questions")
    std_score: float = Field(..., description="Standard deviation of scores")
    scores_by_question_type: dict[str, float] = Field(
        default_factory=dict, description="Mean scores by question type"
    )
    total_cost_usd: float = Field(..., description="Total cost of evaluation")
    total_time_s: float = Field(..., description="Total time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)


class AblationResult(BaseModel):
    """Results from an ablation study comparing conditions."""

    run_id: str = Field(..., description="Unique identifier for this ablation")
    model: str = Field(..., description="Model evaluated")
    figure_type: FigureType | None = Field(None, description="Figure type filter if any")
    results_by_condition: dict[str, BenchmarkResult] = Field(
        ..., description="Results for each condition"
    )
    baseline_condition: PromptCondition = Field(
        PromptCondition.VANILLA, description="Baseline for comparison"
    )
    improvements: dict[str, float] = Field(
        default_factory=dict, description="Improvement over baseline by condition"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


# Mappings between figure types and appropriate personas
FIGURE_TYPE_TO_PERSONA: dict[FigureType, Persona] = {
    # Protein Analysis -> Molecular Biologist
    FigureType.WESTERN_BLOT: Persona.MOLECULAR_BIOLOGIST,
    FigureType.COOMASSIE_GEL: Persona.MOLECULAR_BIOLOGIST,
    FigureType.DOT_BLOT: Persona.MOLECULAR_BIOLOGIST,
    # Binding & Activity Assays -> Bioanalytical Scientist
    FigureType.ELISA: Persona.BIOANALYTICAL_SCIENTIST,
    FigureType.DOSE_RESPONSE: Persona.BIOANALYTICAL_SCIENTIST,
    FigureType.IC50_EC50: Persona.BIOANALYTICAL_SCIENTIST,
    # Pharmacokinetics -> Pharmacologist
    FigureType.PK_CURVE: Persona.PHARMACOLOGIST,
    FigureType.AUC_PLOT: Persona.PHARMACOLOGIST,
    FigureType.COMPARTMENT_MODEL: Persona.PHARMACOLOGIST,
    # Flow Cytometry -> Immunologist
    FigureType.FLOW_BIAXIAL: Persona.IMMUNOLOGIST,
    FigureType.FLOW_HISTOGRAM: Persona.IMMUNOLOGIST,
    FigureType.GATING_STRATEGY: Persona.IMMUNOLOGIST,
    # Genomics/Transcriptomics -> Computational Biologist
    FigureType.HEATMAP: Persona.COMPUTATIONAL_BIOLOGIST,
    FigureType.VOLCANO_PLOT: Persona.COMPUTATIONAL_BIOLOGIST,
    FigureType.PATHWAY_ENRICHMENT: Persona.COMPUTATIONAL_BIOLOGIST,
    # Cell-Based Assays -> Cell Biologist
    FigureType.VIABILITY_CURVE: Persona.CELL_BIOLOGIST,
    FigureType.PROLIFERATION: Persona.CELL_BIOLOGIST,
    FigureType.CYTOTOXICITY: Persona.CELL_BIOLOGIST,
}


def get_persona_for_figure_type(figure_type: FigureType) -> Persona:
    """Get the appropriate persona for a figure type."""
    return FIGURE_TYPE_TO_PERSONA[figure_type]
