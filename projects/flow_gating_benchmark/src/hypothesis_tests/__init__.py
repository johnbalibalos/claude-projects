"""
Hypothesis testing framework for rigorous evaluation of LLM reasoning.

This module implements falsifiable hypothesis tests to distinguish between:
1. Token frequency effects vs. genuine reasoning
2. Task-specific vs. generalizable failures
3. Prior hallucination vs. context processing issues
4. Safety over-triggering vs. context blindness

Each test is designed to "disprove your own findings" by testing alternative
explanations for observed model behavior.
"""

from .frequency_confound import (
    AlienCellTest,
    FrequencyCorrelation,
    PubMedFrequencyLookup,
)
from .format_ablation import (
    FormatAblationTest,
    PromptFormat,
)
from .cot_mechanistic import (
    CoTAnnotator,
    InferenceTag,
    RedPenAnalysis,
)
from .cognitive_refusal import (
    CognitiveRefusalTest,
    RefusalType,
)
from .runner import (
    HypothesisTestRunner,
    HypothesisTestResult,
    AblationConfig,
)

__all__ = [
    # Frequency confound
    "AlienCellTest",
    "FrequencyCorrelation",
    "PubMedFrequencyLookup",
    # Format ablation
    "FormatAblationTest",
    "PromptFormat",
    # CoT mechanistic
    "CoTAnnotator",
    "InferenceTag",
    "RedPenAnalysis",
    # Cognitive refusal
    "CognitiveRefusalTest",
    "RefusalType",
    # Runner
    "HypothesisTestRunner",
    "HypothesisTestResult",
    "AblationConfig",
]
