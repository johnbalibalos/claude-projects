#!/usr/bin/env python3
"""
Run A/B Test comparing gating strategies.

Compares LLM predictions against:
- HIPC 2016 expert-validated definitions
- OMIP paper-specific ground truth

Tests effect of:
- Reasoning approach (zero-shot vs CoT vs WoT)
- RAG context (none vs HIPC reference)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.ab_testing import ABTester, HIPC_CELL_DEFINITIONS, save_hipc_definitions
from experiments.experiment_conditions import (
    EXPERIMENTAL_CONDITIONS,
    ExperimentalCondition,
    ReasoningApproach,
    RAGCondition,
    get_prompt_for_condition,
)

# Check for API keys
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def load_test_cases(data_dir: Path) -> list[dict]:
    """Load all test cases from ground truth directory."""
    test_cases = []
    gt_dir = data_dir / "verified"

    for json_file in sorted(gt_dir.glob("*.json")):
        with open(json_file) as f:
            test_case = json.load(f)
            test_cases.append(test_case)

    return test_cases


def format_panel_for_prompt(panel: dict) -> str:
    """Format panel entries for prompt."""
    lines = []
    for entry in panel.get("entries", []):
        marker = entry.get("marker", "")
        fluor = entry.get("fluorophore", "")
        lines.append(f"- {marker}: {fluor}")
    return "\n".join(lines)


def call_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Claude API with prompt."""
    if not HAS_ANTHROPIC:
        return "[Anthropic SDK not installed]"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ANTHROPIC_API_KEY not set]"

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API with prompt."""
    if not HAS_OPENAI:
        return "[OpenAI SDK not installed]"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "[OPENAI_API_KEY not set]"

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def parse_hierarchy_from_response(response: str) -> dict:
    """Parse gating hierarchy from LLM response."""
    import re

    # Try to extract JSON if present
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    hierarchy = {
        "name": "All Events",
        "children": [],
        "marker_logic": [],
    }

    # Find the code block with the tree structure
    tree_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if tree_match:
        tree_text = tree_match.group(1)
    else:
        tree_text = response

    # Parse lines looking for gate definitions
    lines = tree_text.split("\n")
    stack = [(0, hierarchy)]  # (indent_level, node)

    # Tree characters to strip (including Unicode box drawing)
    tree_chars = set("│├└─┌┐┘┬┴┼╔╗╚╝╠╣╦╩╬| \t")

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Strip tree characters from the left manually
        idx = 0
        for i, char in enumerate(line):
            if char in tree_chars:
                idx = i + 1
            else:
                break
        stripped = line[idx:].strip()
        if not stripped:
            continue

        # Calculate indent by position of first non-tree char
        level = idx // 4  # Tree structures typically use 4-char indents

        # Extract population name
        # Two formats to handle:
        # 1. "Live cells (7-AAD negative)" -> gate_name="Live cells" (marker info in parens)
        # 2. "CD4+ CD8- (CD4+ T cells)" -> gate_name="CD4+ T cells" (name in parens)
        paren_match = re.search(r'\(([^)]+)\)', stripped)
        if paren_match:
            paren_content = paren_match.group(1).strip()
            pre_paren = stripped[:stripped.find('(')].strip()

            # Check if paren content looks like marker info (contains +/- or common markers)
            marker_pattern = r'(CD\d+\w*[+-]|CCR\d+[+-]|7-AAD|positive|negative|[+-]\s*$)'
            if re.search(marker_pattern, paren_content, re.IGNORECASE):
                # Paren content is marker info, use pre-paren as gate name
                gate_name = pre_paren if pre_paren else paren_content
                marker_part = paren_content
            else:
                # Paren content is the gate name
                gate_name = paren_content
                marker_part = pre_paren
        else:
            gate_name = stripped
            marker_part = stripped

        # Skip non-gate lines (explanatory text, headers, etc.)
        skip_patterns = [
            r'^root',
            r'^all\s*events?',
            r'^key\s+gating',
            r'^note',
            r'^\d+\.',
            r'^\*\*',
            r'^the\s+',
            r'^this\s+',
            r'^for\s+',
        ]
        # Check both the full line and extracted gate name
        if any(re.match(pat, stripped.lower()) for pat in skip_patterns):
            continue
        if any(re.match(pat, gate_name.lower()) for pat in skip_patterns):
            continue

        if len(gate_name) > 80 or len(gate_name) < 2:
            continue

        # Extract marker logic from the marker part
        marker_logic = []
        markers = re.findall(r'(CD\d+\w*|CCR\d+|7-AAD|Zombie\s*\w+)([+-])?', marker_part, re.IGNORECASE)
        for marker, sign in markers:
            if sign:
                marker_logic.append({
                    "marker": marker,
                    "positive": sign == "+",
                    "level": None,
                })

        node = {
            "name": gate_name,
            "children": [],
            "marker_logic": marker_logic,
        }

        # Find correct parent based on indent level
        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            parent = stack[-1][1]
            parent["children"].append(node)

        stack.append((level, node))

    return hierarchy


def run_single_test(
    test_case: dict,
    condition: ExperimentalCondition,
    model_name: str,
    call_fn,
) -> dict:
    """Run a single test case with given condition."""

    # Format panel
    panel_str = format_panel_for_prompt(test_case.get("panel", {}))
    context = test_case.get("context", {})

    # Generate prompt
    prompt = get_prompt_for_condition(
        condition=condition,
        panel=panel_str,
        sample_type=context.get("sample_type", "Unknown"),
        application=context.get("application", "Immunophenotyping"),
        omip_excerpt=test_case.get("validation", {}).get("paper_source", ""),
    )

    # Call LLM
    response = call_fn(prompt, model_name)

    # Parse response
    predicted_hierarchy = parse_hierarchy_from_response(response)

    return {
        "test_case_id": test_case["test_case_id"],
        "condition": condition.name,
        "model": model_name,
        "prompt": prompt,
        "response": response,
        "predicted_hierarchy": predicted_hierarchy,
        "ground_truth": test_case.get("gating_hierarchy", {}),
    }


def run_ab_test_suite(
    test_cases: list[dict],
    conditions: list[ExperimentalCondition],
    models: list[tuple[str, callable]],
    output_dir: Path,
) -> dict:
    """Run full A/B test suite."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_test_cases": len(test_cases),
        "conditions": [c.name for c in conditions],
        "models": [m[0] for m in models],
        "runs": [],
    }

    tester = ABTester()

    # Run tests
    for test_case in test_cases:
        print(f"\nTesting: {test_case['test_case_id']}")

        for condition in conditions:
            print(f"  Condition: {condition.name}")

            for model_name, call_fn in models:
                print(f"    Model: {model_name}...", end=" ")

                try:
                    run_result = run_single_test(
                        test_case=test_case,
                        condition=condition,
                        model_name=model_name,
                        call_fn=call_fn,
                    )

                    # Evaluate against both standards
                    ab_result = tester.evaluate_prediction(
                        predicted_hierarchy=run_result["predicted_hierarchy"],
                        omip_ground_truth=run_result["ground_truth"],
                        test_case_id=run_result["test_case_id"],
                        model=model_name,
                    )

                    run_result["ab_result"] = ab_result.to_dict()
                    results["runs"].append(run_result)

                    print(f"HIPC: {ab_result.hipc_score:.2f}, OMIP: {ab_result.omip_score:.2f}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    results["runs"].append({
                        "test_case_id": test_case["test_case_id"],
                        "condition": condition.name,
                        "model": model_name,
                        "error": str(e),
                    })

    # Save results
    output_file = output_dir / f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Generate summary
    summary = generate_summary(results)
    summary_file = output_dir / f"ab_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")

    return results


def generate_summary(results: dict) -> dict:
    """Generate summary statistics from results."""

    summary = {
        "timestamp": results["timestamp"],
        "n_runs": len(results["runs"]),
        "by_condition": {},
        "by_model": {},
        "overall": {
            "mean_hipc_score": 0,
            "mean_omip_score": 0,
            "hipc_wins": 0,
            "omip_wins": 0,
            "ties": 0,
        },
    }

    valid_runs = [r for r in results["runs"] if "ab_result" in r]

    if not valid_runs:
        return summary

    # Aggregate by condition
    for condition in set(r["condition"] for r in valid_runs):
        cond_runs = [r for r in valid_runs if r["condition"] == condition]
        summary["by_condition"][condition] = {
            "n_runs": len(cond_runs),
            "mean_hipc_score": sum(r["ab_result"]["hipc_score"] for r in cond_runs) / len(cond_runs),
            "mean_omip_score": sum(r["ab_result"]["omip_score"] for r in cond_runs) / len(cond_runs),
        }

    # Aggregate by model
    for model in set(r["model"] for r in valid_runs):
        model_runs = [r for r in valid_runs if r["model"] == model]
        summary["by_model"][model] = {
            "n_runs": len(model_runs),
            "mean_hipc_score": sum(r["ab_result"]["hipc_score"] for r in model_runs) / len(model_runs),
            "mean_omip_score": sum(r["ab_result"]["omip_score"] for r in model_runs) / len(model_runs),
        }

    # Overall
    summary["overall"]["mean_hipc_score"] = sum(r["ab_result"]["hipc_score"] for r in valid_runs) / len(valid_runs)
    summary["overall"]["mean_omip_score"] = sum(r["ab_result"]["omip_score"] for r in valid_runs) / len(valid_runs)

    for r in valid_runs:
        adv = r["ab_result"]["hipc_advantage"]
        if adv > 0:
            summary["overall"]["hipc_wins"] += 1
        elif adv < 0:
            summary["overall"]["omip_wins"] += 1
        else:
            summary["overall"]["ties"] += 1

    return summary


def run_dry_run(test_cases: list[dict], conditions: list[ExperimentalCondition]) -> None:
    """Run a dry run to show prompts without API calls."""

    print("=" * 60)
    print("DRY RUN - Showing prompts without API calls")
    print("=" * 60)

    for test_case in test_cases[:2]:  # Just first 2
        print(f"\n{'='*60}")
        print(f"Test Case: {test_case['test_case_id']}")
        print(f"{'='*60}")

        panel_str = format_panel_for_prompt(test_case.get("panel", {}))
        context = test_case.get("context", {})

        for condition in conditions[:2]:  # Just first 2 conditions
            print(f"\n--- Condition: {condition.name} ---")
            prompt = get_prompt_for_condition(
                condition=condition,
                panel=panel_str,
                sample_type=context.get("sample_type", "Unknown"),
                application=context.get("application", "Immunophenotyping"),
            )
            print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run A/B test for gating strategies")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts without API calls")
    parser.add_argument("--models", nargs="+", default=["claude-sonnet-4-20250514"], help="Models to test")
    parser.add_argument("--conditions", nargs="+", help="Conditions to test (default: all)")
    parser.add_argument("--test-cases", nargs="+", help="Specific test cases (default: all)")
    args = parser.parse_args()

    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "results" / "ab_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export HIPC definitions
    ref_dir = data_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    save_hipc_definitions(ref_dir / "hipc_2016_definitions.json")

    # Load test cases
    test_cases = load_test_cases(data_dir)
    print(f"Loaded {len(test_cases)} test cases")

    if args.test_cases:
        test_cases = [tc for tc in test_cases if tc["test_case_id"] in args.test_cases]
        print(f"Filtered to {len(test_cases)} test cases")

    # Select conditions
    if args.conditions:
        conditions = [EXPERIMENTAL_CONDITIONS[c] for c in args.conditions if c in EXPERIMENTAL_CONDITIONS]
    else:
        conditions = [
            EXPERIMENTAL_CONDITIONS["baseline_zero_shot"],
            EXPERIMENTAL_CONDITIONS["cot_no_rag"],
            EXPERIMENTAL_CONDITIONS["cot_hipc_rag"],
        ]

    print(f"Testing {len(conditions)} conditions: {[c.name for c in conditions]}")

    # Dry run mode
    if args.dry_run:
        run_dry_run(test_cases, conditions)
        return

    # Set up model callers
    models = []
    for model in args.models:
        if "claude" in model.lower():
            models.append((model, call_claude))
        elif "gpt" in model.lower():
            models.append((model, call_openai))
        else:
            print(f"Unknown model type: {model}")

    if not models:
        print("No valid models specified. Use --dry-run to see prompts.")
        return

    # Run tests
    run_ab_test_suite(
        test_cases=test_cases[:5],  # Limit for initial run
        conditions=conditions,
        models=models,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
