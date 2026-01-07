"""DrugDevBench: Benchmark for evaluating LLM interpretation of drug development figures."""

from drugdevbench.data import (
    FigureType,
    Persona,
    PromptCondition,
    Question,
    QuestionType,
    Figure,
    Annotation,
    EvaluationResponse,
)
from drugdevbench.models import DrugDevBenchEvaluator, EvaluatorConfig
from drugdevbench.prompts import build_system_prompt

__version__ = "0.1.0"
__all__ = [
    # Data types
    "FigureType",
    "Persona",
    "PromptCondition",
    "Question",
    "QuestionType",
    "Figure",
    "Annotation",
    "EvaluationResponse",
    # Core classes
    "DrugDevBenchEvaluator",
    "EvaluatorConfig",
    # Functions
    "build_system_prompt",
]
