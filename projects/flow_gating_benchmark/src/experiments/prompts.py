"""
Prompt templates for gating hierarchy prediction.

Defines different prompting strategies:
- Direct: Simple instruction
- Chain-of-thought: Encourages step-by-step reasoning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from curation.schemas import TestCase


@dataclass
class PromptTemplate:
    """A prompt template for gating prediction."""

    name: str
    template: str
    strategy: Literal["direct", "cot"]


# JSON schema for output
OUTPUT_SCHEMA = """{
    "name": "Gate Name",
    "markers": ["marker1", "marker2"],
    "children": [
        {
            "name": "Child Gate",
            "markers": ["marker3"],
            "children": [...]
        }
    ]
}"""


DIRECT_TEMPLATE = """You are an expert flow cytometrist. Given the following flow cytometry panel information, predict the gating hierarchy that an expert would use for data analysis.

{context}

## Task

Predict the complete gating hierarchy, starting from "All Events" through quality control gates (time, singlets, live/dead) to final cell population identification.

Return your answer as a JSON object with this structure:
{schema}

Provide only the JSON hierarchy in your final answer."""


COT_TEMPLATE = """You are an expert flow cytometrist. Given the following flow cytometry panel information, predict the gating hierarchy that an expert would use for data analysis.

{context}

## Task

Predict the complete gating hierarchy, starting from "All Events" through quality control gates to final cell population identification.

Before providing your final answer, briefly consider:
- What quality control gates are needed for this specific panel?
- What populations can this panel's markers identify?
- How should gates be organized hierarchically?

Use your expertise to determine the best approach for this specific panel. Different panels may require different gating strategies.

After your reasoning, provide the final hierarchy as a JSON object with this structure:
{schema}

End with only the JSON hierarchy."""


# New: Explanation-capturing template for debugging model decisions
EXPLANATION_TEMPLATE = """You are an expert flow cytometrist. Given the following flow cytometry panel information, predict the gating hierarchy that an expert would use for data analysis.

{context}

## Task

Predict the complete gating hierarchy with explanations for each gating decision.

For each gate in your hierarchy, explain:
1. **Purpose**: Why is this gate needed?
2. **Markers**: Which markers define this population and how?
3. **Placement**: Why is this gate positioned under its parent?

Return your answer as a JSON object where each gate includes a "rationale" field:
{{
    "name": "Gate Name",
    "markers": ["marker1", "marker2"],
    "rationale": "Brief explanation of why this gate is included and its biological significance",
    "children": [...]
}}

Provide the complete hierarchy with rationales."""


PROMPT_TEMPLATES = {
    "direct": PromptTemplate(
        name="direct",
        template=DIRECT_TEMPLATE,
        strategy="direct",
    ),
    "cot": PromptTemplate(
        name="chain_of_thought",
        template=COT_TEMPLATE,
        strategy="cot",
    ),
    "explanation": PromptTemplate(
        name="explanation",
        template=EXPLANATION_TEMPLATE,
        strategy="direct",  # Uses direct strategy but captures rationales
    ),
}


def format_context_minimal(test_case: TestCase) -> str:
    """Format minimal context (markers only)."""
    markers = ", ".join(test_case.panel.markers)
    return f"## Panel\n\nMarkers: {markers}"


def format_context_standard(test_case: TestCase) -> str:
    """Format standard context (markers + sample type + application)."""
    lines = [
        "## Experiment Information",
        "",
        f"Sample Type: {test_case.context.sample_type}",
        f"Species: {test_case.context.species}",
        f"Application: {test_case.context.application}",
        "",
        "## Panel",
        "",
    ]

    for entry in test_case.panel.entries:
        line = f"- {entry.marker}: {entry.fluorophore}"
        if entry.clone:
            line += f" (clone: {entry.clone})"
        lines.append(line)

    return "\n".join(lines)


def format_context_rich(test_case: TestCase) -> str:
    """Format rich context (standard + notes + panel size)."""
    base = format_context_standard(test_case)

    additional = [
        "",
        "## Additional Information",
        "",
        f"Panel Size: {test_case.panel.n_colors} colors",
        f"Complexity: {test_case.complexity.value}",
    ]

    if test_case.context.tissue:
        additional.append(f"Tissue: {test_case.context.tissue}")

    if test_case.context.additional_notes:
        additional.append(f"Notes: {test_case.context.additional_notes}")

    if test_case.omip_id:
        additional.append(f"Reference: {test_case.omip_id}")

    return base + "\n".join(additional)


CONTEXT_FORMATTERS = {
    "minimal": format_context_minimal,
    "standard": format_context_standard,
    "rich": format_context_rich,
}


def build_prompt(
    test_case: TestCase,
    template_name: str = "direct",
    context_level: str = "standard",
) -> str:
    """
    Build a complete prompt for a test case.

    Args:
        test_case: TestCase with panel and context
        template_name: Which prompt template to use
        context_level: How much context to include

    Returns:
        Complete prompt string
    """
    template = PROMPT_TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown template: {template_name}")

    formatter = CONTEXT_FORMATTERS.get(context_level)
    if formatter is None:
        raise ValueError(f"Unknown context level: {context_level}")

    context = formatter(test_case)

    return template.template.format(
        context=context,
        schema=OUTPUT_SCHEMA,
    )
