"""
JSON extraction utilities and common protocols for LLM judge evaluation.

Extracted from llm_judge.py for reuse across judge modules.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

@dataclass
class JSONExtractionResult:
    """Result of extracting JSON from LLM response."""

    data: dict[str, Any]
    success: bool
    raw_json: str
    error: str | None = None


def extract_json_from_response(raw_response: str) -> JSONExtractionResult:
    """
    Extract JSON data from an LLM response.

    Tries multiple strategies:
    1. JSON in markdown code block (```json ... ```)
    2. Raw JSON object ({ ... })

    Args:
        raw_response: The raw LLM response text

    Returns:
        JSONExtractionResult with parsed data or error information
    """
    # Strategy 1: JSON in markdown code block
    json_match = re.search(r'```json\s*([\s\S]*?)```', raw_response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Strategy 2: Raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            json_str = json_match.group(0)
        else:
            return JSONExtractionResult(
                data={},
                success=False,
                raw_json="",
                error="No JSON found in response",
            )

    try:
        data = json.loads(json_str)
        return JSONExtractionResult(
            data=data,
            success=True,
            raw_json=json_str,
        )
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}. Raw JSON: {json_str[:200]}...")
        return JSONExtractionResult(
            data={},
            success=False,
            raw_json=json_str,
            error=f"JSON parse error: {e}",
        )

class JudgeModel(Protocol):
    """Protocol for judge model clients."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a judgment from the model."""
        ...

