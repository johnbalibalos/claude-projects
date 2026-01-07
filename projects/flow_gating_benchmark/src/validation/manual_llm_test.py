"""
Manual LLM test for gating strategy prediction.

This module provides a simple test to validate that the core task
(predicting gating hierarchies from panel information) is meaningful
and feasible for LLMs.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Example panel for testing (from OMIP-069 subset)
EXAMPLE_PANEL = {
    "omip_id": "OMIP-069",
    "title": "40-Color Full Spectrum Flow Cytometry Panel",
    "sample_type": "Human PBMC",
    "species": "human",
    "application": "Deep immunophenotyping of major immune lineages",
    "panel": [
        {"marker": "CD3", "fluorophore": "BUV395", "clone": "UCHT1"},
        {"marker": "CD4", "fluorophore": "BUV496", "clone": "SK3"},
        {"marker": "CD8", "fluorophore": "BUV661", "clone": "SK1"},
        {"marker": "CD45", "fluorophore": "BV421", "clone": "HI30"},
        {"marker": "CD45RA", "fluorophore": "BV510", "clone": "HI100"},
        {"marker": "CD19", "fluorophore": "BV605", "clone": "SJ25C1"},
        {"marker": "CD14", "fluorophore": "BV650", "clone": "M5E2"},
        {"marker": "CD16", "fluorophore": "BV711", "clone": "3G8"},
        {"marker": "CD56", "fluorophore": "BV750", "clone": "5.1H11"},
        {"marker": "Live/Dead", "fluorophore": "Zombie NIR", "clone": "N/A"},
    ],
}

# Expected hierarchy (simplified ground truth)
EXPECTED_HIERARCHY = {
    "name": "All Events",
    "children": [
        {
            "name": "Time",
            "markers": ["Time"],
            "children": [
                {
                    "name": "Singlets",
                    "markers": ["FSC-A", "FSC-H"],
                    "children": [
                        {
                            "name": "Live",
                            "markers": ["Zombie NIR"],
                            "children": [
                                {
                                    "name": "CD45+",
                                    "markers": ["CD45"],
                                    "children": [
                                        {"name": "T cells", "markers": ["CD3"]},
                                        {"name": "B cells", "markers": ["CD19"]},
                                        {"name": "NK cells", "markers": ["CD56", "CD16"]},
                                        {"name": "Monocytes", "markers": ["CD14"]},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ],
}

PROMPT_TEMPLATE = """You are an expert flow cytometrist. Given the following flow cytometry panel, predict the gating hierarchy that an expert would use for data analysis.

## Panel Information

OMIP: {omip_id}
Title: {title}
Sample Type: {sample_type}
Species: {species}
Application: {application}

### Markers and Fluorophores

{marker_list}

## Task

Predict the complete gating hierarchy, starting from "All Events" through quality control gates (time, singlets, live/dead) to final cell population identification.

Return your answer as a JSON object with this structure:
{{
    "name": "Gate Name",
    "markers": ["marker1", "marker2"],  // markers used for this gate
    "children": [...]  // nested child gates
}}

Think through this step-by-step:
1. What quality control gates are needed first?
2. What major cell lineages can be identified with this panel?
3. What is the logical gating order to identify each population?

Provide only the JSON hierarchy in your final answer, no additional text.
"""


def format_marker_list(panel: list[dict]) -> str:
    """Format panel markers as a readable list."""
    lines = []
    for entry in panel:
        line = f"- {entry['marker']}: {entry['fluorophore']}"
        if entry.get("clone") and entry["clone"] != "N/A":
            line += f" (clone: {entry['clone']})"
        lines.append(line)
    return "\n".join(lines)


def run_manual_test(
    model: str = "claude-sonnet-4-20250514",
    panel: dict | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a manual test of gating hierarchy prediction.

    Args:
        model: Model to use (claude-sonnet-4-20250514, gpt-4o, etc.)
        panel: Panel to test (defaults to EXAMPLE_PANEL)
        verbose: Whether to print progress

    Returns:
        Dictionary with test results
    """
    if panel is None:
        panel = EXAMPLE_PANEL

    # Format the prompt
    marker_list = format_marker_list(panel["panel"])
    prompt = PROMPT_TEMPLATE.format(
        omip_id=panel.get("omip_id", "Unknown"),
        title=panel.get("title", "Unknown"),
        sample_type=panel.get("sample_type", "Unknown"),
        species=panel.get("species", "Unknown"),
        application=panel.get("application", "Unknown"),
        marker_list=marker_list,
    )

    if verbose:
        print("=" * 60)
        print("MANUAL LLM TEST FOR GATING HIERARCHY PREDICTION")
        print("=" * 60)
        print(f"\nModel: {model}")
        print(f"Panel: {panel.get('omip_id', 'Custom')}")
        print(f"Markers: {len(panel['panel'])}")
        print("\n" + "-" * 60)
        print("PROMPT:")
        print("-" * 60)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # Call the model
    result = {
        "model": model,
        "panel_id": panel.get("omip_id", "custom"),
        "n_markers": len(panel["panel"]),
        "prompt_length": len(prompt),
    }

    try:
        if "claude" in model.lower():
            response = call_claude(model, prompt)
        elif "gpt" in model.lower():
            response = call_openai(model, prompt)
        else:
            response = None
            result["error"] = f"Unknown model type: {model}"

        if response:
            result["response"] = response
            result["response_length"] = len(response)

            # Try to parse JSON from response
            try:
                # Extract JSON from response (may be wrapped in markdown)
                json_str = extract_json(response)
                predicted_hierarchy = json.loads(json_str)
                result["parsed_hierarchy"] = predicted_hierarchy
                result["parse_success"] = True
            except (json.JSONDecodeError, ValueError) as e:
                result["parse_success"] = False
                result["parse_error"] = str(e)

    except Exception as e:
        result["error"] = str(e)

    if verbose:
        print("\n" + "-" * 60)
        print("RESPONSE:")
        print("-" * 60)
        if "response" in result:
            print(result["response"][:1000])
            if len(result["response"]) > 1000:
                print("... (truncated)")
        elif "error" in result:
            print(f"ERROR: {result['error']}")

        print("\n" + "-" * 60)
        print("EXPECTED HIERARCHY (simplified):")
        print("-" * 60)
        print(json.dumps(EXPECTED_HIERARCHY, indent=2))

        if result.get("parsed_hierarchy"):
            print("\n" + "-" * 60)
            print("PARSED RESPONSE:")
            print("-" * 60)
            print(json.dumps(result["parsed_hierarchy"], indent=2))

    return result


def call_claude(model: str, prompt: str) -> str | None:
    """Call Claude API."""
    if Anthropic is None:
        raise ImportError("anthropic package not installed")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def call_openai(model: str, prompt: str) -> str | None:
    """Call OpenAI API."""
    if OpenAI is None:
        raise ImportError("openai package not installed")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )

    return response.choices[0].message.content


def extract_json(text: str) -> str:
    """Extract JSON from text that may include markdown formatting."""
    # Try to find JSON in code blocks
    import re

    code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block:
        return code_block.group(1)

    # Try to find raw JSON
    json_match = re.search(r"(\{[\s\S]*\})", text)
    if json_match:
        return json_match.group(1)

    return text


def evaluate_prediction(predicted: dict, expected: dict) -> dict[str, Any]:
    """
    Compare predicted hierarchy to expected.

    This is a simplified evaluation - the full benchmark will have
    more sophisticated metrics.
    """

    def get_all_gates(hierarchy: dict, prefix: str = "") -> set[str]:
        """Extract all gate names from hierarchy."""
        gates = set()
        if "name" in hierarchy:
            gates.add(hierarchy["name"])
        if "children" in hierarchy:
            for child in hierarchy["children"]:
                gates.update(get_all_gates(child))
        return gates

    predicted_gates = get_all_gates(predicted)
    expected_gates = get_all_gates(expected)

    intersection = predicted_gates & expected_gates
    precision = len(intersection) / len(predicted_gates) if predicted_gates else 0
    recall = len(intersection) / len(expected_gates) if expected_gates else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "predicted_gates": list(predicted_gates),
        "expected_gates": list(expected_gates),
        "matching_gates": list(intersection),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": list(expected_gates - predicted_gates),
        "extra": list(predicted_gates - expected_gates),
    }


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-20250514"

    print("\nRunning manual LLM test...")
    print("Make sure ANTHROPIC_API_KEY or OPENAI_API_KEY is set\n")

    result = run_manual_test(model=model)

    if result.get("parsed_hierarchy"):
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        eval_result = evaluate_prediction(result["parsed_hierarchy"], EXPECTED_HIERARCHY)
        print(json.dumps(eval_result, indent=2))

    # Save result
    output_file = "data/manual_test_result.json"
    try:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_file}")
    except Exception as e:
        print(f"\nCould not save result: {e}")
