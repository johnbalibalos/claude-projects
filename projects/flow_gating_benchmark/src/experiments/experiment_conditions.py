#!/usr/bin/env python3
"""
Experimental Conditions for A/B Testing.

Tests multiple factors:
1. Ground Truth Standard: HIPC (2016) vs OMIP paper-specific
2. Reasoning Approach: Chain-of-Thought vs Weight-of-Thought
3. Context: With RAG (HIPC reference) vs Without RAG

This creates a factorial design to understand:
- Do LLMs naturally align with expert standards (HIPC)?
- Does explicit reasoning improve gating predictions?
- Does providing HIPC reference material help?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReasoningApproach(str, Enum):
    """Reasoning strategy for gating predictions."""

    ZERO_SHOT = "zero_shot"
    # Direct prediction without explicit reasoning

    CHAIN_OF_THOUGHT = "chain_of_thought"
    # Step-by-step reasoning: "First identify... then gate..."
    # Sequential, deterministic approach

    WEIGHT_OF_THOUGHT = "weight_of_thought"
    # Consider multiple options with uncertainty
    # "T cells could be CD3+ only OR CD3+ CD19-. Given PBMC sample..."
    # Explores alternatives before deciding


class RAGCondition(str, Enum):
    """Whether to include reference material in context."""

    NO_RAG = "no_rag"
    # No reference material, rely on parametric knowledge

    HIPC_RAG = "hipc_rag"
    # Include HIPC 2016 cell definitions in context

    OMIP_RAG = "omip_rag"
    # Include relevant OMIP paper excerpts

    BOTH_RAG = "both_rag"
    # Include both HIPC and OMIP references


class GroundTruthStandard(str, Enum):
    """Which standard to evaluate predictions against."""

    HIPC = "hipc"
    # Expert-validated HIPC 2016 definitions

    OMIP = "omip"
    # Paper-specific gating from OMIP publications

    BOTH = "both"
    # Evaluate against both (A/B comparison)


@dataclass
class ExperimentalCondition:
    """A single experimental condition combining all factors."""

    name: str
    reasoning: ReasoningApproach
    rag: RAGCondition
    ground_truth: GroundTruthStandard
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reasoning": self.reasoning.value,
            "rag": self.rag.value,
            "ground_truth": self.ground_truth.value,
            "description": self.description,
        }


# Pre-defined experimental conditions
EXPERIMENTAL_CONDITIONS = {
    # Baseline conditions
    "baseline_zero_shot": ExperimentalCondition(
        name="baseline_zero_shot",
        reasoning=ReasoningApproach.ZERO_SHOT,
        rag=RAGCondition.NO_RAG,
        ground_truth=GroundTruthStandard.BOTH,
        description="Baseline: zero-shot with no RAG, evaluate against both standards",
    ),

    # Chain-of-Thought conditions
    "cot_no_rag": ExperimentalCondition(
        name="cot_no_rag",
        reasoning=ReasoningApproach.CHAIN_OF_THOUGHT,
        rag=RAGCondition.NO_RAG,
        ground_truth=GroundTruthStandard.BOTH,
        description="Chain-of-thought without reference material",
    ),
    "cot_hipc_rag": ExperimentalCondition(
        name="cot_hipc_rag",
        reasoning=ReasoningApproach.CHAIN_OF_THOUGHT,
        rag=RAGCondition.HIPC_RAG,
        ground_truth=GroundTruthStandard.BOTH,
        description="Chain-of-thought with HIPC definitions in context",
    ),

    # Weight-of-Thought conditions
    "wot_no_rag": ExperimentalCondition(
        name="wot_no_rag",
        reasoning=ReasoningApproach.WEIGHT_OF_THOUGHT,
        rag=RAGCondition.NO_RAG,
        ground_truth=GroundTruthStandard.BOTH,
        description="Weight-of-thought (consider alternatives) without RAG",
    ),
    "wot_hipc_rag": ExperimentalCondition(
        name="wot_hipc_rag",
        reasoning=ReasoningApproach.WEIGHT_OF_THOUGHT,
        rag=RAGCondition.HIPC_RAG,
        ground_truth=GroundTruthStandard.BOTH,
        description="Weight-of-thought with HIPC definitions - does explicit uncertainty + reference help?",
    ),

    # Full factorial for key comparisons
    "cot_both_rag": ExperimentalCondition(
        name="cot_both_rag",
        reasoning=ReasoningApproach.CHAIN_OF_THOUGHT,
        rag=RAGCondition.BOTH_RAG,
        ground_truth=GroundTruthStandard.BOTH,
        description="Chain-of-thought with both HIPC and OMIP references",
    ),
    "wot_both_rag": ExperimentalCondition(
        name="wot_both_rag",
        reasoning=ReasoningApproach.WEIGHT_OF_THOUGHT,
        rag=RAGCondition.BOTH_RAG,
        ground_truth=GroundTruthStandard.BOTH,
        description="Weight-of-thought with both references",
    ),
}


# Prompt templates for each reasoning approach
REASONING_PROMPTS = {
    ReasoningApproach.ZERO_SHOT: """
Given this flow cytometry panel, predict the gating hierarchy.

Panel:
{panel}

Sample: {sample_type}
Application: {application}

Provide the gating hierarchy as a tree structure.
""",

    ReasoningApproach.CHAIN_OF_THOUGHT: """
Given this flow cytometry panel, predict the gating hierarchy step by step.

Panel:
{panel}

Sample: {sample_type}
Application: {application}

Think through this systematically:
1. First, identify the quality control gates (singlets, live/dead)
2. Then, identify the major lineage markers in the panel
3. For each lineage, determine the appropriate parent population
4. Consider which negative markers are needed to exclude other lineages
5. Finally, identify any subset markers for further phenotyping

Show your reasoning, then provide the final gating hierarchy.
""",

    ReasoningApproach.WEIGHT_OF_THOUGHT: """
Given this flow cytometry panel, predict the gating hierarchy by considering multiple approaches.

Panel:
{panel}

Sample: {sample_type}
Application: {application}

For key gating decisions, consider alternatives:

1. **Lymphocyte gating**:
   - Option A: Use FSC/SSC scatter to gate lymphocytes first
   - Option B: Go directly to lineage markers from live cells
   - Weigh: Which is more robust for this panel?

2. **T cell definition**:
   - Option A: CD3+ only
   - Option B: CD3+ CD19- (exclude B cells)
   - Weigh: Given markers in panel, which is more precise?

3. **B cell definition**:
   - Option A: CD19+ only
   - Option B: CD3- CD19+
   - Option C: CD19+ CD20+
   - Weigh: Panel has {b_markers}, which approach?

For each decision, briefly explain your choice and confidence level.
Then provide the final gating hierarchy.
""",
}


# RAG context templates
RAG_CONTEXTS = {
    RAGCondition.HIPC_RAG: """
## Reference: HIPC 2016 Standardized Cell Definitions
Source: https://www.nature.com/articles/srep20686

### Key Population Definitions:

**T cells**: CD3+ CD19- (negative for B cell markers when in panel)
- CD4+ T cells: CD3+ CD4+ CD8-
- CD8+ T cells: CD3+ CD4- CD8+
- Naive: CD45RA+ CCR7+
- Central Memory: CD45RA- CCR7+
- Effector Memory: CD45RA- CCR7-
- TEMRA: CD45RA+ CCR7-

**B cells**: CD3- CD19+ (or CD20+, either acceptable)
- Naive B: CD19+ IgD+ CD27-
- Memory B: CD19+ CD27+
- Transitional: CD19+ CD24hi CD38hi
- Plasmablasts: CD19+ CD27++ CD38++

**NK cells**: CD3- CD56+ and/or CD16+
- CD56bright: CD3- CD56bright CD16dim/-
- CD56dim: CD3- CD56dim CD16+

**Monocytes**: CD14+
- Classical: CD14++ CD16-
- Intermediate: CD14++ CD16+
- Non-classical: CD14dim CD16++

Note: HIPC recommends going directly to lineage markers rather than
using FSC/SSC lymphocyte gates, as this reduces variability.
""",

    RAGCondition.OMIP_RAG: """
## Reference: OMIP Paper Gating Strategy
{omip_gating_excerpt}
""",

    RAGCondition.BOTH_RAG: """
## Reference 1: HIPC 2016 Standardized Definitions
{hipc_context}

## Reference 2: OMIP Paper-Specific Strategy
{omip_context}

Note: The OMIP paper may use slightly different gating than HIPC standards.
Consider both approaches when designing your hierarchy.
""",
}


@dataclass
class ExperimentPlan:
    """Complete experiment plan for A/B testing."""

    name: str
    description: str
    conditions: list[ExperimentalCondition]
    test_cases: list[str]  # OMIP IDs to test
    models: list[str]
    n_repetitions: int = 3  # For variance estimation

    # Hypotheses to test
    hypotheses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "conditions": [c.to_dict() for c in self.conditions],
            "test_cases": self.test_cases,
            "models": self.models,
            "n_repetitions": self.n_repetitions,
            "hypotheses": self.hypotheses,
            "total_runs": len(self.conditions) * len(self.test_cases) * len(self.models) * self.n_repetitions,
        }


# Default experiment plan
DEFAULT_EXPERIMENT_PLAN = ExperimentPlan(
    name="hipc_vs_omip_ab_test",
    description="""
    A/B test comparing LLM gating predictions against:
    - HIPC 2016 expert-validated definitions
    - OMIP paper-specific ground truth

    Tests effect of:
    - Reasoning approach (zero-shot vs CoT vs WoT)
    - RAG context (none vs HIPC vs OMIP vs both)
    """,
    conditions=[
        EXPERIMENTAL_CONDITIONS["baseline_zero_shot"],
        EXPERIMENTAL_CONDITIONS["cot_no_rag"],
        EXPERIMENTAL_CONDITIONS["cot_hipc_rag"],
        EXPERIMENTAL_CONDITIONS["wot_no_rag"],
        EXPERIMENTAL_CONDITIONS["wot_hipc_rag"],
    ],
    test_cases=[
        "OMIP-023",  # Simple 10-color
        "OMIP-069",  # Complex 40-color
        "OMIP-058",  # T/NK focused
        "OMIP-044",  # DC focused
    ],
    models=[
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "gpt-4o",
    ],
    n_repetitions=3,
    hypotheses=[
        "H1: LLMs with CoT+HIPC_RAG will score higher on HIPC standard than zero-shot",
        "H2: WoT will outperform CoT for complex panels (>25 colors)",
        "H3: Negative marker inclusion will be higher with HIPC_RAG",
        "H4: Zero-shot predictions will align more with HIPC than OMIP (training data effect)",
        "H5: RAG reduces hallucination rate for rare populations",
    ],
)


def get_prompt_for_condition(
    condition: ExperimentalCondition,
    panel: str,
    sample_type: str,
    application: str,
    omip_excerpt: str = "",
) -> str:
    """Generate complete prompt for an experimental condition."""

    # Base reasoning prompt
    prompt = REASONING_PROMPTS[condition.reasoning].format(
        panel=panel,
        sample_type=sample_type,
        application=application,
        b_markers="CD19, CD20" if "CD19" in panel or "CD20" in panel else "limited B markers",
    )

    # Add RAG context if needed
    if condition.rag == RAGCondition.HIPC_RAG:
        prompt = RAG_CONTEXTS[RAGCondition.HIPC_RAG] + "\n\n" + prompt
    elif condition.rag == RAGCondition.OMIP_RAG:
        prompt = RAG_CONTEXTS[RAGCondition.OMIP_RAG].format(
            omip_gating_excerpt=omip_excerpt
        ) + "\n\n" + prompt
    elif condition.rag == RAGCondition.BOTH_RAG:
        prompt = RAG_CONTEXTS[RAGCondition.BOTH_RAG].format(
            hipc_context=RAG_CONTEXTS[RAGCondition.HIPC_RAG],
            omip_context=omip_excerpt,
        ) + "\n\n" + prompt

    return prompt
