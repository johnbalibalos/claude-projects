"""
LLM response parser for gating hierarchy predictions.

Handles various output formats:
- JSON hierarchy (code blocks or raw)
- Markdown nested lists
- Indented text hierarchies
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""

    success: bool
    hierarchy: dict | None = None
    format_detected: str | None = None
    error: str | None = None
    raw_response: str | None = None
    validation_warnings: list[str] | None = None


# Terms that indicate meta-commentary rather than gate names
META_TERMS = frozenset([
    "here", "the following", "this", "would", "could", "should",
    "based on", "given", "assuming", "typically", "usually",
    "research question", "populations of interest", "fluorochromes",
])

# Minimum requirements for a valid hierarchy
MIN_GATES = 3
MAX_NAME_LENGTH = 100
MIN_NAME_LENGTH = 2


def parse_llm_response(response: str) -> ParseResult:
    """
    Parse LLM response into a hierarchy dict.

    Tries multiple parsing strategies in order:
    1. JSON in code blocks (most reliable)
    2. Raw JSON object
    3. Markdown nested lists
    4. Indented text hierarchy

    Args:
        response: Raw LLM response text

    Returns:
        ParseResult with parsed hierarchy or error
    """
    if not response or not response.strip():
        return ParseResult(success=False, error="Empty response", raw_response=response)

    # Try each parser in order
    for parser, format_name in [
        (_parse_json_code_block, "json_code_block"),
        (_parse_json_raw, "json_raw"),
        (_parse_markdown_list, "markdown_list"),
        (_parse_indented_text, "indented_text"),
    ]:
        result = parser(response)
        if result is not None:
            # Validate the parsed hierarchy
            is_valid, warnings = validate_hierarchy(result)
            if is_valid:
                return ParseResult(
                    success=True,
                    hierarchy=result,
                    format_detected=format_name,
                    raw_response=response,
                    validation_warnings=warnings if warnings else None,
                )

    return ParseResult(
        success=False,
        error="Could not parse response as JSON, markdown, or indented text",
        raw_response=response,
    )


def _parse_json_code_block(response: str) -> dict | None:
    """Extract JSON from code blocks."""
    patterns = [
        r"```json\s*\n?([\s\S]*?)\n?```",
        r"```\s*\n?([\s\S]*?)\n?```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                hierarchy = json.loads(match.group(1).strip())
                return _normalize_hierarchy(hierarchy)
            except json.JSONDecodeError:
                continue
    return None


def _parse_json_raw(response: str) -> dict | None:
    """Extract raw JSON object from response."""
    # Find the outermost JSON object
    brace_start = response.find("{")
    if brace_start == -1:
        return None

    # Find matching closing brace
    depth = 0
    in_string = False
    escape = False

    for i, char in enumerate(response[brace_start:], start=brace_start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    json_str = response[brace_start : i + 1]
                    hierarchy = json.loads(json_str)
                    if _looks_like_hierarchy(hierarchy):
                        return _normalize_hierarchy(hierarchy)
                except json.JSONDecodeError:
                    pass
                break
    return None


def _looks_like_hierarchy(obj: dict) -> bool:
    """Check if a dict looks like a gating hierarchy."""
    if not isinstance(obj, dict):
        return False
    # Must have name or be a root wrapper
    if "name" in obj:
        return True
    if "root" in obj and isinstance(obj["root"], dict):
        return True
    return False


def _parse_markdown_list(response: str) -> dict | None:
    """Parse markdown nested list into hierarchy."""
    lines = response.split("\n")
    items = []

    for line in lines:
        match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.+)$", line)
        if match:
            indent = len(match.group(1))
            name = match.group(3).strip()

            # Clean up name
            name = re.sub(r"\*\*([^*]+)\*\*", r"\1", name)  # Remove bold
            name = re.sub(r"\(.*?\)", "", name).strip()  # Remove parentheticals

            if _is_valid_gate_name(name):
                items.append({"indent": indent, "name": name})

    if len(items) < MIN_GATES:
        return None

    try:
        return _items_to_hierarchy(items)
    except Exception:
        return None


def _parse_indented_text(response: str) -> dict | None:
    """Parse indented text into hierarchy."""
    lines = response.split("\n")
    items = []

    for line in lines:
        if not line.strip():
            continue

        # Skip meta-commentary lines
        line_lower = line.lower()
        if any(term in line_lower for term in META_TERMS):
            continue

        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Clean up the name
        name = stripped.strip()
        name = re.sub(r"^[-*•→>]\s*", "", name)  # Remove bullets
        name = re.sub(r":.*$", "", name)  # Remove descriptions after colon

        if _is_valid_gate_name(name):
            items.append({"indent": indent, "name": name})

    if len(items) < MIN_GATES:
        return None

    try:
        return _items_to_hierarchy(items)
    except Exception:
        return None


def _is_valid_gate_name(name: str) -> bool:
    """Check if a string looks like a valid gate name."""
    if not name or len(name) < MIN_NAME_LENGTH or len(name) > MAX_NAME_LENGTH:
        return False

    # Reject if contains meta-commentary phrases
    name_lower = name.lower()
    if any(term in name_lower for term in META_TERMS):
        return False

    return True


def _items_to_hierarchy(items: list[dict]) -> dict:
    """Convert list of items with indents to hierarchy dict."""
    if not items:
        return {"name": "All Events", "children": []}

    # Normalize indents
    min_indent = min(item["indent"] for item in items)
    for item in items:
        item["indent"] -= min_indent

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
            root["children"].append(node)

        stack.append((node, item["indent"]))

    return root


def _normalize_hierarchy(hierarchy: dict) -> dict:
    """Normalize hierarchy structure to standard format."""
    # Unwrap root wrapper
    if "root" in hierarchy and isinstance(hierarchy["root"], dict):
        hierarchy = hierarchy["root"]

    # Ensure name field
    if "name" not in hierarchy:
        if "gate_name" in hierarchy:
            hierarchy["name"] = hierarchy["gate_name"]
        else:
            hierarchy["name"] = "All Events"

    # Ensure children field
    if "children" not in hierarchy:
        hierarchy["children"] = []

    # Recursively normalize children
    hierarchy["children"] = [_normalize_hierarchy(child) for child in hierarchy["children"]]

    return hierarchy


def validate_hierarchy(hierarchy: dict) -> tuple[bool, list[str]]:
    """
    Validate that a hierarchy has valid structure.

    Returns:
        Tuple of (is_valid, list of warnings/issues)
    """
    warnings = []

    if not isinstance(hierarchy, dict):
        return False, ["Hierarchy is not a dictionary"]

    if "name" not in hierarchy:
        warnings.append("Root missing 'name' field")

    def validate_node(node: dict, path: str = "root") -> bool:
        nonlocal warnings

        if not isinstance(node, dict):
            warnings.append(f"{path}: Node is not a dictionary")
            return False

        if "name" not in node:
            warnings.append(f"{path}: Missing 'name' field")
            return False

        name = node.get("name", "")
        if not _is_valid_gate_name(name):
            warnings.append(f"{path}: Invalid gate name '{name}'")

        children = node.get("children", [])
        if not isinstance(children, list):
            warnings.append(f"{path}: 'children' is not a list")
            return False

        for i, child in enumerate(children):
            validate_node(child, f"{path}.children[{i}]")

        return True

    validate_node(hierarchy)

    # Count total gates
    def count_gates(node: dict) -> int:
        return 1 + sum(count_gates(c) for c in node.get("children", []))

    total_gates = count_gates(hierarchy)
    if total_gates < MIN_GATES:
        warnings.append(f"Hierarchy has only {total_gates} gates (minimum {MIN_GATES})")
        return False, warnings

    # Check for meta-commentary in gate names
    def check_meta_terms(node: dict) -> None:
        name = node.get("name", "").lower()
        for term in META_TERMS:
            if term in name:
                warnings.append(f"Gate name '{node.get('name')}' contains meta-commentary")
                break
        for child in node.get("children", []):
            check_meta_terms(child)

    check_meta_terms(hierarchy)

    return len(warnings) == 0 or all("meta-commentary" in w for w in warnings), warnings


def extract_markers_from_response(response: str) -> list[str]:
    """
    Extract marker mentions from response text.

    Useful for identifying what markers the LLM thinks are relevant
    even if parsing fails.
    """
    patterns = [
        r"\bCD\d+\w*",  # CD markers
        r"\bHLA-[A-Z]+",  # HLA markers
        r"\bFSC-[AHW]",  # Forward scatter
        r"\bSSC-[AHW]",  # Side scatter
    ]

    markers = set()
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        markers.update(m.upper() for m in matches)

    return sorted(markers)
