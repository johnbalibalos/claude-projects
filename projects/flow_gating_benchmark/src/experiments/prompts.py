"""
Prompt templates for gating hierarchy prediction.

Defines different prompting strategies:
- Direct: Simple instruction
- Chain-of-thought: Encourages step-by-step reasoning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..curation.schemas import TestCase


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

Think through this step-by-step:

1. **Quality Control Gates**: What initial QC gates are needed? Consider:
   - Time gate (to exclude acquisition artifacts)
   - Singlet gate (to exclude doublets/aggregates)
   - Live/Dead discrimination

2. **Major Lineage Identification**: What major cell lineages can this panel identify? Consider:
   - Which markers define major populations (T cells, B cells, NK cells, myeloid)?
   - What is the logical order to separate these populations?

3. **Subset Identification**: For each major lineage, what subsets can be identified?
   - What markers distinguish subsets?
   - What is the gating order within each lineage?

4. **Hierarchy Structure**: Organize into a complete gating tree.

After your reasoning, provide the final hierarchy as a JSON object with this structure:
{schema}

Begin your analysis, then end with only the JSON hierarchy."""


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
