"""Tests for LLM response parsing."""

import json
import pytest
from src.evaluation.response_parser import (
    parse_llm_response,
    extract_markers_from_response,
    validate_hierarchy_structure,
)


class TestParseJsonResponse:
    """Tests for JSON response parsing."""

    def test_parse_json_code_block(self):
        """Test parsing JSON in code block."""
        response = '''Here is the gating hierarchy:

```json
{
    "name": "All Events",
    "children": [
        {"name": "Singlets", "markers": ["FSC-A", "FSC-H"], "children": []}
    ]
}
```
'''
        result = parse_llm_response(response)

        assert result.success
        assert result.format_detected == "json_code_block"
        assert result.hierarchy["name"] == "All Events"
        assert len(result.hierarchy["children"]) == 1

    def test_parse_raw_json(self):
        """Test parsing raw JSON without code block."""
        response = '''{"name": "All Events", "children": []}'''

        result = parse_llm_response(response)

        assert result.success
        assert result.hierarchy["name"] == "All Events"

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding explanation."""
        response = '''Based on the panel, I recommend this hierarchy:

{
    "name": "All Events",
    "markers": [],
    "children": [
        {
            "name": "Live",
            "markers": ["Live/Dead"],
            "children": []
        }
    ]
}

This hierarchy starts with...'''

        result = parse_llm_response(response)

        assert result.success
        assert result.hierarchy["name"] == "All Events"


class TestParseMarkdownResponse:
    """Tests for markdown list parsing."""

    def test_parse_markdown_list(self):
        """Test parsing markdown nested list."""
        response = '''Here is the gating hierarchy:

- All Events
  - Singlets
    - Live
      - CD45+
        - T cells
        - B cells
'''
        result = parse_llm_response(response)

        assert result.success
        assert result.format_detected == "markdown_list"
        assert result.hierarchy["name"] == "All Events"

    def test_parse_numbered_list(self):
        """Test parsing numbered markdown list."""
        response = '''
1. All Events
   1. Singlets
   2. Live
'''
        result = parse_llm_response(response)

        # May or may not parse successfully depending on structure
        # At minimum should not crash
        assert result is not None


class TestParseIndentedText:
    """Tests for indented text parsing."""

    def test_parse_indented_hierarchy(self):
        """Test parsing indented text hierarchy."""
        response = '''
All Events
    Singlets
        Live
            CD45+
                T cells
                B cells
'''
        result = parse_llm_response(response)

        assert result.success
        assert result.hierarchy["name"] == "All Events"


class TestParseFailures:
    """Tests for parse failure handling."""

    def test_empty_response(self):
        """Test with empty response."""
        result = parse_llm_response("")

        assert not result.success
        assert result.error is not None

    def test_invalid_json(self):
        """Test with invalid JSON."""
        response = '''```json
{invalid json here}
```'''
        result = parse_llm_response(response)

        # Should try other parsers or fail gracefully
        assert result is not None

    def test_no_hierarchy(self):
        """Test with response that has no hierarchy structure."""
        response = "I don't know how to answer this question."

        result = parse_llm_response(response)

        assert not result.success


class TestExtractMarkers:
    """Tests for marker extraction from text."""

    def test_extract_cd_markers(self):
        """Test extracting CD markers."""
        response = "The panel includes CD3, CD4, CD8, and CD45."
        markers = extract_markers_from_response(response)

        assert "CD3" in markers
        assert "CD4" in markers
        assert "CD8" in markers
        assert "CD45" in markers

    def test_extract_scatter(self):
        """Test extracting scatter parameters."""
        response = "Gate on FSC-A vs FSC-H for singlets, then SSC-A vs FSC-A."
        markers = extract_markers_from_response(response)

        assert "FSC-A" in markers
        assert "FSC-H" in markers
        assert "SSC-A" in markers

    def test_case_insensitive(self):
        """Test case insensitive extraction."""
        response = "cd3 and Cd4 markers"
        markers = extract_markers_from_response(response)

        assert "CD3" in markers
        assert "CD4" in markers


class TestValidateHierarchyStructure:
    """Tests for hierarchy structure validation."""

    def test_valid_hierarchy(self, sample_hierarchy):
        """Test with valid hierarchy."""
        is_valid, issues = validate_hierarchy_structure(sample_hierarchy)

        assert is_valid
        assert len(issues) == 0

    def test_missing_name(self):
        """Test with missing name field."""
        invalid = {"children": []}
        is_valid, issues = validate_hierarchy_structure(invalid)

        assert not is_valid
        assert any("name" in issue.lower() for issue in issues)

    def test_invalid_children_type(self):
        """Test with invalid children type."""
        invalid = {"name": "root", "children": "not a list"}
        is_valid, issues = validate_hierarchy_structure(invalid)

        assert not is_valid

    def test_not_a_dict(self):
        """Test with non-dict input."""
        is_valid, issues = validate_hierarchy_structure("not a dict")

        assert not is_valid
