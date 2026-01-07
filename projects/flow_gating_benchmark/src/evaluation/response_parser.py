"""
LLM response parser for gating hierarchy predictions.

Handles various output formats:
- JSON hierarchy
- Markdown nested lists
- Free text descriptions
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""

    success: bool
    hierarchy: dict | None = None
    format_detected: str | None = None
    error: str | None = None
    raw_response: str | None = None


def parse_llm_response(response: str) -> ParseResult:
    """
    Parse LLM response into a hierarchy dict.

    Tries multiple parsing strategies:
    1. JSON in code blocks
    2. Raw JSON
    3. Markdown nested lists
    4. Indented text hierarchy

    Args:
        response: Raw LLM response text

    Returns:
        ParseResult with parsed hierarchy or error
    """
    if not response or not response.strip():
        return ParseResult(
            success=False,
            error="Empty response",
            raw_response=response,
        )

    # Try JSON parsing first (most reliable)
    json_result = _try_parse_json(response)
    if json_result.success:
        return json_result

    # Try markdown list parsing
    markdown_result = _try_parse_markdown(response)
    if markdown_result.success:
        return markdown_result

    # Try indented text parsing
    indented_result = _try_parse_indented(response)
    if indented_result.success:
        return indented_result

    return ParseResult(
        success=False,
        error="Could not parse response as JSON, markdown, or indented text",
        raw_response=response,
    )


def _try_parse_json(response: str) -> ParseResult:
    """Try to parse JSON from response."""

    # Look for JSON in code blocks
    code_block_patterns = [
        r"```json\s*\n?([\s\S]*?)\n?```",  # ```json ... ```
        r"```\s*\n?([\s\S]*?)\n?```",  # ``` ... ```
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                hierarchy = json.loads(match.group(1).strip())
                return ParseResult(
                    success=True,
                    hierarchy=_normalize_hierarchy(hierarchy),
                    format_detected="json_code_block",
                    raw_response=response,
                )
            except json.JSONDecodeError:
                continue

    # Try to find raw JSON object
    json_patterns = [
        r"(\{[\s\S]*\})",  # Any JSON object
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                hierarchy = json.loads(match)
                # Validate it looks like a hierarchy
                if "name" in hierarchy or "root" in hierarchy:
                    return ParseResult(
                        success=True,
                        hierarchy=_normalize_hierarchy(hierarchy),
                        format_detected="json_raw",
                        raw_response=response,
                    )
            except json.JSONDecodeError:
                continue

    return ParseResult(success=False, raw_response=response)


def _try_parse_markdown(response: str) -> ParseResult:
    """Try to parse markdown nested list into hierarchy."""

    lines = response.split("\n")

    # Look for lines that look like markdown list items
    list_items = []
    for line in lines:
        # Match: "- Gate Name" or "* Gate Name" or "1. Gate Name"
        match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.+)$", line)
        if match:
            indent = len(match.group(1))
            name = match.group(3).strip()

            # Clean up name
            name = re.sub(r"\*\*([^*]+)\*\*", r"\1", name)  # Remove bold
            name = re.sub(r"\(.*?\)", "", name).strip()  # Remove parentheticals

            list_items.append({"indent": indent, "name": name})

    if not list_items:
        return ParseResult(success=False, raw_response=response)

    # Convert to hierarchy
    try:
        hierarchy = _list_items_to_hierarchy(list_items)
        return ParseResult(
            success=True,
            hierarchy=hierarchy,
            format_detected="markdown_list",
            raw_response=response,
        )
    except Exception as e:
        return ParseResult(
            success=False,
            error=f"Failed to convert markdown list: {e}",
            raw_response=response,
        )


def _try_parse_indented(response: str) -> ParseResult:
    """Try to parse indented text into hierarchy."""

    lines = response.split("\n")

    items = []
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Skip lines that don't look like gate names
        if any(skip in line.lower() for skip in ["here", "the", "this", "would", "could"]):
            continue

        # Calculate indent level
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Clean up the name
        name = stripped.strip()
        name = re.sub(r"^[-*•→>]\s*", "", name)  # Remove bullets
        name = re.sub(r":.*$", "", name)  # Remove descriptions after colon

        if name and len(name) > 1 and len(name) < 50:
            items.append({"indent": indent, "name": name})

    if len(items) < 3:  # Need at least a few items
        return ParseResult(success=False, raw_response=response)

    try:
        hierarchy = _list_items_to_hierarchy(items)
        return ParseResult(
            success=True,
            hierarchy=hierarchy,
            format_detected="indented_text",
            raw_response=response,
        )
    except Exception:
        return ParseResult(success=False, raw_response=response)


def _list_items_to_hierarchy(items: list[dict]) -> dict:
    """Convert list of items with indents to hierarchy dict."""

    if not items:
        return {"name": "root", "children": []}

    # Find minimum indent to use as base
    min_indent = min(item["indent"] for item in items)

    # Normalize indents
    for item in items:
        item["indent"] -= min_indent

    # Build hierarchy using stack
    root = {"name": items[0]["name"], "children": []}
    stack = [(root, items[0]["indent"])]

    for item in items[1:]:
        node = {"name": item["name"], "children": []}

        # Pop stack until we find parent
        while stack and stack[-1][1] >= item["indent"]:
            stack.pop()

        if stack:
            stack[-1][0]["children"].append(node)
        else:
            # Item is at root level - this shouldn't happen normally
            root["children"].append(node)

        stack.append((node, item["indent"]))

    return root


def _normalize_hierarchy(hierarchy: dict) -> dict:
    """Normalize hierarchy structure to standard format."""

    # Handle case where root is wrapped
    if "root" in hierarchy and isinstance(hierarchy["root"], dict):
        hierarchy = hierarchy["root"]

    # Ensure required fields
    if "name" not in hierarchy:
        if "gate_name" in hierarchy:
            hierarchy["name"] = hierarchy["gate_name"]
        else:
            hierarchy["name"] = "All Events"

    # Normalize children
    if "children" not in hierarchy:
        hierarchy["children"] = []

    # Recursively normalize children
    hierarchy["children"] = [
        _normalize_hierarchy(child) for child in hierarchy["children"]
    ]

    return hierarchy


def extract_markers_from_response(response: str) -> list[str]:
    """
    Extract marker mentions from response text.

    Useful for identifying what markers the LLM thinks are relevant
    even if parsing fails.
    """
    # Common marker patterns
    marker_patterns = [
        r"\bCD\d+\w*",  # CD markers (CD3, CD4, CD45RO, etc.)
        r"\bHLA-[A-Z]+",  # HLA markers
        r"\bFSC-[AHW]",  # Forward scatter
        r"\bSSC-[AHW]",  # Side scatter
    ]

    markers = set()
    for pattern in marker_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        markers.update(m.upper() for m in matches)

    return sorted(markers)


def validate_hierarchy_structure(hierarchy: dict) -> tuple[bool, list[str]]:
    """
    Validate that a hierarchy has valid structure.

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    if not isinstance(hierarchy, dict):
        return False, ["Hierarchy is not a dictionary"]

    if "name" not in hierarchy:
        issues.append("Root missing 'name' field")

    def validate_node(node: dict, path: str = "root") -> None:
        if not isinstance(node, dict):
            issues.append(f"{path}: Node is not a dictionary")
            return

        if "name" not in node:
            issues.append(f"{path}: Missing 'name' field")

        children = node.get("children", [])
        if not isinstance(children, list):
            issues.append(f"{path}: 'children' is not a list")
            return

        for i, child in enumerate(children):
            validate_node(child, f"{path}.children[{i}]")

    validate_node(hierarchy)

    return len(issues) == 0, issues
