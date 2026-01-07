"""Question generation for DrugDevBench."""

from drugdevbench.questions.generator import (
    generate_questions,
    extract_claims_from_legend,
    QuestionTemplate,
    QUESTION_TEMPLATES,
)

__all__ = [
    "generate_questions",
    "extract_claims_from_legend",
    "QuestionTemplate",
    "QUESTION_TEMPLATES",
]
