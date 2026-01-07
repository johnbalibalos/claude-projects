"""Prompt dispatcher for building system prompts from components."""

import random

from drugdevbench.data.schemas import FigureType, Persona, PromptCondition
from drugdevbench.prompts.base import BASE_SCIENTIFIC_PROMPT
from drugdevbench.prompts.personas import PERSONA_PROMPTS
from drugdevbench.prompts.skills import SKILL_PROMPTS


# Mapping from figure types to their appropriate skills
FIGURE_TYPE_TO_SKILL: dict[FigureType, str] = {
    # Protein Analysis
    FigureType.WESTERN_BLOT: "western_blot",
    FigureType.COOMASSIE_GEL: "western_blot",  # Similar interpretation
    FigureType.DOT_BLOT: "western_blot",
    # Binding & Activity Assays
    FigureType.ELISA: "elisa",
    FigureType.DOSE_RESPONSE: "dose_response",
    FigureType.IC50_EC50: "dose_response",
    # Pharmacokinetics
    FigureType.PK_CURVE: "pk_curve",
    FigureType.AUC_PLOT: "pk_curve",
    FigureType.COMPARTMENT_MODEL: "pk_curve",
    # Flow Cytometry
    FigureType.FLOW_BIAXIAL: "flow_biaxial",
    FigureType.FLOW_HISTOGRAM: "flow_histogram",
    FigureType.GATING_STRATEGY: "flow_biaxial",
    # Genomics/Transcriptomics
    FigureType.HEATMAP: "heatmap",
    FigureType.VOLCANO_PLOT: "heatmap",  # Similar statistical interpretation
    FigureType.PATHWAY_ENRICHMENT: "heatmap",
    # Cell-Based Assays
    FigureType.VIABILITY_CURVE: "dose_response",  # Similar curve fitting
    FigureType.PROLIFERATION: "dose_response",
    FigureType.CYTOTOXICITY: "dose_response",
}

# Mapping from figure types to appropriate personas
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


def get_skill_for_figure_type(figure_type: FigureType) -> str:
    """Get the skill name for a figure type.

    Args:
        figure_type: The type of figure

    Returns:
        Name of the appropriate skill
    """
    return FIGURE_TYPE_TO_SKILL[figure_type]


def get_persona_for_figure_type(figure_type: FigureType) -> Persona:
    """Get the appropriate persona for a figure type.

    Args:
        figure_type: The type of figure

    Returns:
        The appropriate persona
    """
    return FIGURE_TYPE_TO_PERSONA[figure_type]


def get_persona_prompt(persona: Persona) -> str:
    """Get the prompt text for a persona.

    Args:
        persona: The persona to get the prompt for

    Returns:
        The persona prompt text
    """
    return PERSONA_PROMPTS[persona]


def get_skill_prompt(skill_name: str) -> str:
    """Get the prompt text for a skill.

    Args:
        skill_name: Name of the skill

    Returns:
        The skill prompt text
    """
    return SKILL_PROMPTS[skill_name]


def get_wrong_skill(correct_skill: str) -> str:
    """Get a mismatched skill for ablation testing.

    Args:
        correct_skill: The skill that would be correct

    Returns:
        A different skill name (for wrong_skill ablation)
    """
    all_skills = list(SKILL_PROMPTS.keys())
    wrong_skills = [s for s in all_skills if s != correct_skill]
    return random.choice(wrong_skills)


def build_system_prompt(
    condition: PromptCondition,
    figure_type: FigureType,
    custom_persona: Persona | None = None,
    custom_skill: str | None = None,
) -> str:
    """Build a system prompt based on the ablation condition.

    Args:
        condition: The prompt condition (vanilla, base_only, etc.)
        figure_type: The type of figure being analyzed
        custom_persona: Override the default persona for the figure type
        custom_skill: Override the default skill for the figure type

    Returns:
        The complete system prompt string
    """
    if condition == PromptCondition.VANILLA:
        # No additional prompting - let model use its native capabilities
        return "You are an AI assistant. Please answer questions about the provided figure."

    # Determine components to use
    persona = custom_persona or get_persona_for_figure_type(figure_type)
    skill_name = custom_skill or get_skill_for_figure_type(figure_type)

    # Build prompt based on condition
    parts = []

    # Add persona if applicable
    if condition in (PromptCondition.PERSONA_ONLY, PromptCondition.FULL_STACK):
        parts.append(get_persona_prompt(persona))

    # Add base scientific prompt if applicable
    if condition in (
        PromptCondition.BASE_ONLY,
        PromptCondition.BASE_PLUS_SKILL,
        PromptCondition.FULL_STACK,
        PromptCondition.WRONG_SKILL,
    ):
        parts.append(BASE_SCIENTIFIC_PROMPT)

    # Add skill if applicable
    if condition == PromptCondition.BASE_PLUS_SKILL:
        parts.append(get_skill_prompt(skill_name))
    elif condition == PromptCondition.FULL_STACK:
        parts.append(get_skill_prompt(skill_name))
    elif condition == PromptCondition.WRONG_SKILL:
        # Use a mismatched skill
        wrong_skill = get_wrong_skill(skill_name)
        parts.append(get_skill_prompt(wrong_skill))

    return "\n\n---\n\n".join(parts)
