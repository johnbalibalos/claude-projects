"""
LLM-based extraction of gating hierarchies from paper text.

Uses Claude to extract structured gating information from:
1. Methods section text
2. Results section text
3. Figure captions
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .extraction_schema import ExtractionMethod, MethodExtraction


PANEL_EXTRACTION_PROMPT = """Extract the antibody panel from this flow cytometry paper.

Paper title: {title}

Text from paper:
{text}

Extract ALL antibodies/markers mentioned with their fluorophores and clones.
Return a JSON object:
{{
  "entries": [
    {{"marker": "CD3", "fluorophore": "BUV395", "clone": "UCHT1"}},
    {{"marker": "CD4", "fluorophore": "BV421", "clone": "SK3"}},
    ...
  ],
  "notes": "any extraction notes"
}}

Rules:
- Include ALL markers mentioned in the panel
- Use null for missing fluorophore or clone
- Normalize marker names (e.g., "CD3e" -> "CD3")
- Include viability dyes (e.g., "Live/Dead", "Zombie")
- Include scatter parameters if mentioned (FSC, SSC)

Return ONLY the JSON, no other text."""


HIERARCHY_EXTRACTION_PROMPT = """Extract the gating strategy/hierarchy from this flow cytometry paper.

Paper title: {title}

Panel markers: {markers}

Text from paper (methods/results):
{text}

Figure captions:
{figure_captions}

Extract the complete gating hierarchy as a tree structure.
Return a JSON object:
{{
  "hierarchy": {{
    "name": "All Events",
    "marker_logic": null,
    "children": [
      {{
        "name": "Singlets",
        "marker_logic": "FSC-A vs FSC-H",
        "children": [
          {{
            "name": "Live cells",
            "marker_logic": "Viability-",
            "children": [
              {{
                "name": "Lymphocytes",
                "marker_logic": "FSC vs SSC",
                "children": [
                  {{
                    "name": "T cells",
                    "marker_logic": "CD3+",
                    "children": [
                      {{"name": "CD4+ T cells", "marker_logic": "CD4+ CD8-", "children": []}},
                      {{"name": "CD8+ T cells", "marker_logic": "CD4- CD8+", "children": []}}
                    ]
                  }}
                ]
              }}
            ]
          }}
        ]
      }}
    ]
  }},
  "notes": "extraction notes",
  "confidence": 0.8
}}

Rules:
- Start from "All Events" as root
- Include standard gates: Singlets, Live cells, Lymphocytes (if applicable)
- Use marker_logic to show how each gate is defined (e.g., "CD3+ CD19-")
- marker_logic should use +/- notation for marker expression
- Include ALL populations mentioned in the paper
- Maintain parent-child relationships as described
- Set confidence 0.0-1.0 based on how clearly the hierarchy was described

Return ONLY the JSON, no other text."""


CONTEXT_EXTRACTION_PROMPT = """Extract experimental context from this flow cytometry paper.

Paper title: {title}

Abstract:
{abstract}

Return a JSON object:
{{
  "sample_type": "Human PBMC",
  "species": "human",
  "tissue": "Peripheral blood",
  "application": "T cell immunophenotyping",
  "disease_state": null,
  "instrument": "BD FACSymphony"
}}

Return ONLY the JSON, no other text."""


@dataclass
class LLMExtractionResult:
    """Result from LLM extraction."""
    success: bool
    data: dict
    raw_response: str
    model: str
    confidence: float = 0.7
    error: str | None = None


class LLMExtractor:
    """
    Extract gating hierarchies using Claude LLM.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required: pip install anthropic")

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def _parse_json(self, response: str) -> dict | None:
        """Parse JSON from LLM response."""
        # Try to extract from code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try finding JSON object
        for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
            match = re.search(pattern, response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        return None

    def extract_panel(
        self,
        title: str,
        text: str,
    ) -> LLMExtractionResult:
        """Extract panel from paper text."""
        prompt = PANEL_EXTRACTION_PROMPT.format(
            title=title,
            text=text[:8000]  # Limit text length
        )

        try:
            response = self._call_llm(prompt)
            data = self._parse_json(response)

            if data and "entries" in data:
                return LLMExtractionResult(
                    success=True,
                    data=data,
                    raw_response=response,
                    model=self.model,
                    confidence=0.75
                )
            else:
                return LLMExtractionResult(
                    success=False,
                    data={},
                    raw_response=response,
                    model=self.model,
                    error="Could not parse panel entries"
                )

        except Exception as e:
            return LLMExtractionResult(
                success=False,
                data={},
                raw_response="",
                model=self.model,
                error=str(e)
            )

    def extract_hierarchy(
        self,
        title: str,
        text: str,
        markers: list[str],
        figure_captions: str = ""
    ) -> LLMExtractionResult:
        """Extract gating hierarchy from paper text."""
        prompt = HIERARCHY_EXTRACTION_PROMPT.format(
            title=title,
            markers=", ".join(markers) if markers else "Not specified",
            text=text[:10000],
            figure_captions=figure_captions[:2000] if figure_captions else "None"
        )

        try:
            response = self._call_llm(prompt, max_tokens=8192)
            data = self._parse_json(response)

            if data and "hierarchy" in data:
                confidence = data.get("confidence", 0.7)
                return LLMExtractionResult(
                    success=True,
                    data=data,
                    raw_response=response,
                    model=self.model,
                    confidence=confidence
                )
            else:
                return LLMExtractionResult(
                    success=False,
                    data={},
                    raw_response=response,
                    model=self.model,
                    error="Could not parse hierarchy"
                )

        except Exception as e:
            return LLMExtractionResult(
                success=False,
                data={},
                raw_response="",
                model=self.model,
                error=str(e)
            )

    def extract_context(
        self,
        title: str,
        abstract: str
    ) -> LLMExtractionResult:
        """Extract experimental context."""
        prompt = CONTEXT_EXTRACTION_PROMPT.format(
            title=title,
            abstract=abstract[:3000]
        )

        try:
            response = self._call_llm(prompt, max_tokens=1024)
            data = self._parse_json(response)

            if data:
                return LLMExtractionResult(
                    success=True,
                    data=data,
                    raw_response=response,
                    model=self.model,
                    confidence=0.8
                )
            else:
                return LLMExtractionResult(
                    success=False,
                    data={},
                    raw_response=response,
                    model=self.model,
                    error="Could not parse context"
                )

        except Exception as e:
            return LLMExtractionResult(
                success=False,
                data={},
                raw_response="",
                model=self.model,
                error=str(e)
            )


def create_method_extraction(
    result: LLMExtractionResult,
    source_file: str | None = None
) -> MethodExtraction:
    """Convert LLM result to MethodExtraction."""
    return MethodExtraction(
        method=ExtractionMethod.LLM,
        data=result.data,
        confidence=result.confidence if result.success else 0.0,
        model=result.model,
        source_file=source_file,
        notes=result.error if not result.success else None,
        raw_response=result.raw_response[:5000] if result.raw_response else None
    )
