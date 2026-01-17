"""
Prompt templates for gating hierarchy prediction.

Defines different prompting strategies:
- Direct: Simple instruction
- Chain-of-thought: Encourages step-by-step reasoning
- Reference modes: none, hipc (static context injection)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from curation.schemas import TestCase

logger = logging.getLogger(__name__)


# =============================================================================
# LLM OUTPUT SCHEMA (Single Source of Truth)
# =============================================================================


class LLMGateNode(BaseModel):
    """
    Simplified gate schema for LLM output.

    This is the single source of truth for what the LLM should return.
    The full GateNode in schemas.py has additional fields (gate_type, is_critical,
    marker_logic, notes) that are for internal use, not LLM prediction.
    """

    name: str = Field(
        ...,
        description="Gate name (e.g., 'All Events', 'Singlets', 'CD3+ T cells')"
    )
    markers: list[str] = Field(
        default_factory=list,
        description="Markers/dimensions used for this gate (e.g., ['CD3', 'CD19'] or ['FSC-A', 'FSC-H'])"
    )
    children: list["LLMGateNode"] = Field(
        default_factory=list,
        description="Child gates in the hierarchy"
    )


def get_output_schema_json() -> str:
    """
    Generate JSON schema string from Pydantic model.

    This ensures the prompt schema always matches the validation schema.
    """
    schema = LLMGateNode.model_json_schema()
    # Return a simplified example that's easier for LLMs to follow
    example = {
        "name": "Gate Name",
        "markers": ["marker1", "marker2"],
        "children": [
            {
                "name": "Child Gate",
                "markers": ["marker3"],
                "children": []
            }
        ]
    }
    return json.dumps(example, indent=4)

# =============================================================================
# REFERENCE CONTEXT LOADING (Context is Data, not Code)
# =============================================================================

# Cache for loaded reference contexts
_context_cache: dict[str, str] = {}

# Path to context data directory (relative to project root)
_CONTEXT_DIR = Path(__file__).parent.parent.parent / "data" / "context"


def load_reference_context(name: str = "hipc_v1") -> str:
    """
    Load reference context from external data file.

    This treats context as data, enabling:
    - Version control for different context versions
    - A/B testing (hipc_v1 vs hipc_v2)
    - Easy updates without code changes

    Args:
        name: Context file name (without .md extension)

    Returns:
        Context string content

    Raises:
        FileNotFoundError: If context file doesn't exist
    """
    if name in _context_cache:
        return _context_cache[name]

    context_path = _CONTEXT_DIR / f"{name}.md"

    if not context_path.exists():
        logger.warning(f"Context file not found: {context_path}")
        raise FileNotFoundError(f"Reference context '{name}' not found at {context_path}")

    content = context_path.read_text()
    _context_cache[name] = content
    logger.debug(f"Loaded reference context '{name}' ({len(content)} chars)")
    return content


def get_available_contexts() -> list[str]:
    """List available reference context files."""
    if not _CONTEXT_DIR.exists():
        return []
    return [f.stem for f in _CONTEXT_DIR.glob("*.md")]


# Backward compatibility: HIPC_REFERENCE as a lazy-loaded property
# This maintains backward compatibility while loading from file
def _get_hipc_reference() -> str:
    """Get HIPC reference (backward compatibility wrapper)."""
    try:
        return load_reference_context("hipc_v1")
    except FileNotFoundError:
        logger.error("HIPC reference file not found. Returning empty string.")
        return ""


# For backward compatibility, provide HIPC_REFERENCE as a module-level constant
# that's loaded on first access
class _LazyHIPCReference:
    """Lazy loader for HIPC_REFERENCE to maintain backward compatibility."""
    _content: str | None = None

    def __str__(self) -> str:
        if self._content is None:
            self._content = _get_hipc_reference()
        return self._content

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, other: str) -> str:
        return str(self) + other

    def __radd__(self, other: str) -> str:
        return other + str(self)


HIPC_REFERENCE = _LazyHIPCReference()


@dataclass
class PromptTemplate:
    """A prompt template for gating prediction."""

    name: str
    template: str
    strategy: Literal["direct", "cot"]


# Dynamic schema generation - no more hardcoded strings!
# Use get_output_schema_json() to get the schema for prompts


DIRECT_TEMPLATE = """You are an expert cytometrist. Given the following panel information, predict the gating hierarchy that an expert would use for data analysis.

{context}

## Task

Predict the complete gating hierarchy, starting from "All Events" through appropriate quality control gates to final cell population identification. For flow cytometry, include doublet exclusion (singlets) and viability gates. For mass cytometry (CyTOF), adapt the QC strategy accordingly (no FSC/SSC scatter).

Return your answer as a JSON object with this structure:
{schema}

Provide only the JSON hierarchy in your final answer."""


COT_TEMPLATE = """You are an expert cytometrist. Given the following panel information, predict the gating hierarchy that an expert would use for data analysis.

{context}

## Task

Predict the complete gating hierarchy, starting from "All Events" through quality control gates to final cell population identification.

Before providing your final answer, briefly consider:
- What technology is this (flow cytometry or mass cytometry/CyTOF)? Check if fluorophores are metal isotopes (e.g., 145Nd, 176Yb) vs standard fluorophores (e.g., PE, APC, FITC).
- What quality control gates are needed? (Scatter-based singlets for flow cytometry; alternative QC for mass cytometry)
- What populations can this panel's markers identify?
- How should gates be organized hierarchically?

Use your expertise to determine the best approach for this specific panel. Different panels and technologies may require different gating strategies.

After your reasoning, provide the final hierarchy as a JSON object with this structure:
{schema}

End with only the JSON hierarchy."""


# New: Explanation-capturing template for debugging model decisions
EXPLANATION_TEMPLATE = """You are an expert cytometrist. Given the following panel information, predict the gating hierarchy that an expert would use for data analysis.

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
        fluor = entry.fluorophore or "unknown"
        line = f"- {entry.marker}: {fluor}"
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

    # NOTE: Intentionally NOT including omip_id here.
    # Including "Reference: OMIP-077" would let models retrieve from training
    # data rather than reason from markers - defeating the benchmark purpose.

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
    reference: str = "none",
) -> str:
    """
    Build a complete prompt for a test case.

    Args:
        test_case: TestCase with panel and context
        template_name: Which prompt template to use
        context_level: How much context to include
        reference: Reference mode - "none" or "hipc" (HIPC definitions)

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

    # Inject HIPC reference for static context augmentation
    if reference == "hipc":
        context = HIPC_REFERENCE + "\n\n" + context

    return template.template.format(
        context=context,
        schema=get_output_schema_json(),
    )
