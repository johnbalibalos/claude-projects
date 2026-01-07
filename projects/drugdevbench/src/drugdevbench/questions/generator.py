"""Auto-generation of questions from figure legends."""

import re
import uuid
from dataclasses import dataclass

from drugdevbench.data.schemas import FigureType, Question, QuestionType


@dataclass
class QuestionTemplate:
    """Template for generating questions."""

    template: str
    question_type: QuestionType
    difficulty: str
    figure_types: list[FigureType]  # Applicable figure types


# Question templates by figure type
QUESTION_TEMPLATES: list[QuestionTemplate] = [
    # Dose-response questions
    QuestionTemplate(
        template="What is the reported {parameter}?",
        question_type=QuestionType.FACTUAL_EXTRACTION,
        difficulty="basic",
        figure_types=[FigureType.DOSE_RESPONSE, FigureType.IC50_EC50, FigureType.ELISA],
    ),
    QuestionTemplate(
        template="Estimate the {parameter} from the curve.",
        question_type=QuestionType.VISUAL_ESTIMATION,
        difficulty="intermediate",
        figure_types=[FigureType.DOSE_RESPONSE, FigureType.IC50_EC50],
    ),
    QuestionTemplate(
        template="How many replicates were used (n value)?",
        question_type=QuestionType.FACTUAL_EXTRACTION,
        difficulty="basic",
        figure_types=[
            FigureType.DOSE_RESPONSE,
            FigureType.WESTERN_BLOT,
            FigureType.PK_CURVE,
            FigureType.ELISA,
        ],
    ),
    # PK questions
    QuestionTemplate(
        template="What is the approximate Cmax?",
        question_type=QuestionType.VISUAL_ESTIMATION,
        difficulty="intermediate",
        figure_types=[FigureType.PK_CURVE],
    ),
    QuestionTemplate(
        template="Estimate the terminal half-life from the curve.",
        question_type=QuestionType.VISUAL_ESTIMATION,
        difficulty="intermediate",
        figure_types=[FigureType.PK_CURVE],
    ),
    QuestionTemplate(
        template="What route of administration is shown?",
        question_type=QuestionType.FACTUAL_EXTRACTION,
        difficulty="basic",
        figure_types=[FigureType.PK_CURVE],
    ),
    # Western blot questions
    QuestionTemplate(
        template="Is a loading control shown?",
        question_type=QuestionType.QUALITY_ASSESSMENT,
        difficulty="basic",
        figure_types=[FigureType.WESTERN_BLOT, FigureType.COOMASSIE_GEL],
    ),
    QuestionTemplate(
        template="What is the expected molecular weight of the target protein?",
        question_type=QuestionType.FACTUAL_EXTRACTION,
        difficulty="basic",
        figure_types=[FigureType.WESTERN_BLOT],
    ),
    QuestionTemplate(
        template="Are the band intensities consistent with the loading control?",
        question_type=QuestionType.QUALITY_ASSESSMENT,
        difficulty="intermediate",
        figure_types=[FigureType.WESTERN_BLOT],
    ),
    # Flow cytometry questions
    QuestionTemplate(
        template="What percentage of cells are in each quadrant?",
        question_type=QuestionType.VISUAL_ESTIMATION,
        difficulty="intermediate",
        figure_types=[FigureType.FLOW_BIAXIAL],
    ),
    QuestionTemplate(
        template="What markers are shown on each axis?",
        question_type=QuestionType.FACTUAL_EXTRACTION,
        difficulty="basic",
        figure_types=[FigureType.FLOW_BIAXIAL, FigureType.FLOW_HISTOGRAM],
    ),
    QuestionTemplate(
        template="Is proper gating strategy shown?",
        question_type=QuestionType.QUALITY_ASSESSMENT,
        difficulty="intermediate",
        figure_types=[FigureType.FLOW_BIAXIAL, FigureType.GATING_STRATEGY],
    ),
    # Heatmap questions
    QuestionTemplate(
        template="What is the color scale range?",
        question_type=QuestionType.FACTUAL_EXTRACTION,
        difficulty="basic",
        figure_types=[FigureType.HEATMAP],
    ),
    QuestionTemplate(
        template="Do the sample groups cluster as expected?",
        question_type=QuestionType.INTERPRETATION,
        difficulty="expert",
        figure_types=[FigureType.HEATMAP],
    ),
    # General quality questions
    QuestionTemplate(
        template="Are there any concerns with the data quality?",
        question_type=QuestionType.ERROR_DETECTION,
        difficulty="expert",
        figure_types=list(FigureType),  # All figure types
    ),
    QuestionTemplate(
        template="What statistical measure is shown for error bars?",
        question_type=QuestionType.FACTUAL_EXTRACTION,
        difficulty="basic",
        figure_types=[
            FigureType.DOSE_RESPONSE,
            FigureType.PK_CURVE,
            FigureType.ELISA,
            FigureType.VIABILITY_CURVE,
        ],
    ),
]


# Patterns for extracting claims from legends
CLAIM_PATTERNS = [
    # IC50/EC50 values
    (r"IC50\s*[=:]\s*([\d.]+\s*[μµnpm]?M)", "IC50"),
    (r"EC50\s*[=:]\s*([\d.]+\s*[μµnpm]?M)", "EC50"),
    # PK parameters
    (r"[Tt]max\s*[=:]\s*([\d.]+\s*(?:hr?|hours?|min|minutes?))", "Tmax"),
    (r"[Cc]max\s*[=:]\s*([\d.]+\s*[μµnpm]?[gG]/[mL]+)", "Cmax"),
    (r"half-life\s*(?:of\s*)?([\d.]+\s*(?:hr?|hours?|days?))", "half-life"),
    (r"t1/2\s*[=:]\s*([\d.]+\s*(?:hr?|hours?|days?))", "half-life"),
    (r"AUC\s*[=:]\s*([\d.]+\s*[^,.\n]+)", "AUC"),
    # Sample size
    (r"n\s*[=:]\s*(\d+)", "n"),
    (r"[Nn]\s*=\s*(\d+)", "n"),
    # Fold change
    (r"(\d+(?:\.\d+)?)-fold\s+(?:increase|decrease|change)", "fold change"),
    # p-values
    (r"[Pp]\s*[<>=]\s*([\d.]+)", "p-value"),
    # Molecular weights
    (r"(\d+)\s*kDa", "molecular weight"),
    # Concentrations
    (r"(\d+(?:\.\d+)?)\s*[μµnpm]M", "concentration"),
]


def extract_claims_from_legend(legend_text: str) -> list[dict[str, str]]:
    """Extract factual claims from a figure legend.

    Args:
        legend_text: The figure legend text

    Returns:
        List of dictionaries with 'claim_type', 'value', and 'context'
    """
    claims = []

    for pattern, claim_type in CLAIM_PATTERNS:
        matches = re.finditer(pattern, legend_text, re.IGNORECASE)
        for match in matches:
            value = match.group(1).strip()
            # Get surrounding context
            start = max(0, match.start() - 30)
            end = min(len(legend_text), match.end() + 30)
            context = legend_text[start:end]

            claims.append({
                "claim_type": claim_type,
                "value": value,
                "context": context,
                "full_match": match.group(0),
            })

    return claims


def generate_questions(
    figure_id: str,
    figure_type: FigureType,
    legend_text: str | None = None,
    include_templates: bool = True,
    include_claims: bool = True,
) -> list[Question]:
    """Generate questions for a figure.

    Args:
        figure_id: ID of the figure
        figure_type: Type of the figure
        legend_text: Optional figure legend for claim extraction
        include_templates: Include template-based questions
        include_claims: Include questions from extracted claims

    Returns:
        List of generated Question objects
    """
    questions = []

    # Generate template-based questions
    if include_templates:
        for template in QUESTION_TEMPLATES:
            if figure_type in template.figure_types:
                question_id = f"q_{uuid.uuid4().hex[:8]}"
                questions.append(
                    Question(
                        question_id=question_id,
                        figure_id=figure_id,
                        question_text=template.template,
                        question_type=template.question_type,
                        gold_answer="[TO BE FILLED]",  # Placeholder
                        difficulty=template.difficulty,
                        metadata={"source": "template"},
                    )
                )

    # Generate questions from legend claims
    if include_claims and legend_text:
        claims = extract_claims_from_legend(legend_text)
        for claim in claims:
            question_id = f"q_{uuid.uuid4().hex[:8]}"

            # Create question based on claim type
            if claim["claim_type"] in ("IC50", "EC50"):
                question_text = f"What is the reported {claim['claim_type']}?"
            elif claim["claim_type"] == "Tmax":
                question_text = "What is the time to maximum concentration (Tmax)?"
            elif claim["claim_type"] == "Cmax":
                question_text = "What is the maximum concentration (Cmax)?"
            elif claim["claim_type"] == "half-life":
                question_text = "What is the reported half-life?"
            elif claim["claim_type"] == "n":
                question_text = "How many subjects or replicates were used?"
            elif claim["claim_type"] == "molecular weight":
                question_text = "What is the molecular weight of the protein?"
            else:
                question_text = f"What is the reported {claim['claim_type']}?"

            questions.append(
                Question(
                    question_id=question_id,
                    figure_id=figure_id,
                    question_text=question_text,
                    question_type=QuestionType.FACTUAL_EXTRACTION,
                    gold_answer=claim["value"],
                    gold_answer_source=f"Extracted from legend: '{claim['full_match']}'",
                    difficulty="basic",
                    metadata={
                        "source": "claim_extraction",
                        "claim_type": claim["claim_type"],
                        "context": claim["context"],
                    },
                )
            )

    return questions


def generate_questions_for_annotation(
    figure_id: str,
    figure_type: FigureType,
    legend_text: str,
    max_questions: int = 5,
) -> list[Question]:
    """Generate a curated set of questions for annotation.

    Selects a diverse set of questions across difficulty levels and types.

    Args:
        figure_id: ID of the figure
        figure_type: Type of the figure
        legend_text: Figure legend text
        max_questions: Maximum number of questions to generate

    Returns:
        List of Question objects
    """
    all_questions = generate_questions(
        figure_id=figure_id,
        figure_type=figure_type,
        legend_text=legend_text,
        include_templates=True,
        include_claims=True,
    )

    # Prioritize questions with extracted gold answers
    questions_with_answers = [q for q in all_questions if q.gold_answer != "[TO BE FILLED]"]
    questions_without_answers = [q for q in all_questions if q.gold_answer == "[TO BE FILLED]"]

    # Select diverse questions
    selected = []

    # First, add questions with extracted answers
    selected.extend(questions_with_answers[:max_questions])

    # Then fill with template questions, preferring variety in difficulty
    remaining_slots = max_questions - len(selected)
    if remaining_slots > 0:
        # Sort by difficulty to get variety
        by_difficulty = {"basic": [], "intermediate": [], "expert": []}
        for q in questions_without_answers:
            by_difficulty[q.difficulty].append(q)

        for difficulty in ["basic", "intermediate", "expert"]:
            if remaining_slots <= 0:
                break
            for q in by_difficulty[difficulty]:
                if remaining_slots <= 0:
                    break
                selected.append(q)
                remaining_slots -= 1

    return selected[:max_questions]
