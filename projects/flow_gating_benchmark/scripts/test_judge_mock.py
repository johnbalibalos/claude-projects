#!/usr/bin/env python3
"""
Test the LLM judge on mock benchmark responses.

This script evaluates a sample of mock results using Gemini 2.5 Pro as judge,
without requiring the hypothesis_pipeline dependencies.
"""

import json
import re
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.llm_client import GeminiClient  # noqa: E402


def load_ground_truth(test_case_id: str) -> dict | None:
    """Load ground truth for a test case."""
    filename = test_case_id.lower().replace("-", "_") + ".json"
    gt_path = PROJECT_ROOT / "data" / "verified" / filename

    if not gt_path.exists():
        print(f"  Ground truth not found: {gt_path}")
        return None

    with open(gt_path) as f:
        return json.load(f)


def flatten_hierarchy(hierarchy: dict, path: str = "") -> str:
    """Convert hierarchy dict to flat arrow notation."""
    if "root" in hierarchy:
        hierarchy = hierarchy["root"]

    name = hierarchy.get("name", "Unknown")
    current = f"{path} > {name}" if path else name

    children = hierarchy.get("children", [])
    if not children:
        return current

    # Get all leaf paths
    paths = []
    for child in children:
        paths.append(flatten_hierarchy(child, current))

    return "\n".join(paths)


def build_judge_prompt(
    test_case_id: str,
    predicted_response: str,
    ground_truth: dict,
    metrics: dict,
) -> str:
    """Build a simple flat-format prompt for the LLM judge."""

    gt_hierarchy = ground_truth.get("gating_hierarchy", {})
    gt_flat = flatten_hierarchy(gt_hierarchy)

    # Get first 5 paths only
    gt_lines = gt_flat.split("\n")[:5]
    gt_summary = "\n".join(gt_lines)
    if len(gt_flat.split("\n")) > 5:
        gt_summary += f"\n... (+{len(gt_flat.split(chr(10))) - 5} more paths)"

    context = ground_truth.get("context", {})

    prompt = f"""Score this flow cytometry gating hierarchy prediction (0-10 scale).

TEST CASE: {test_case_id}
SAMPLE: {context.get('sample_type', 'unknown')} ({context.get('species', 'unknown')})

EXPECTED HIERARCHY (ground truth paths):
{gt_summary}

PREDICTED RESPONSE:
{predicted_response}

AUTO METRICS:
- F1: {metrics.get('hierarchy_f1', 0):.2f}
- Structure: {metrics.get('structure_accuracy', 0):.2f}
- Critical gates: {metrics.get('critical_gate_recall', 0):.2f}

Rate on these dimensions (0-10 each):

Reply in this EXACT format (one line each):
COMPLETENESS: [0-10]
ACCURACY: [0-10]
SCIENTIFIC: [0-10]
OVERALL: [0-10]
ISSUES: [comma-separated list or "none"]
SUMMARY: [one sentence explanation]
"""
    return prompt


def parse_judge_response(content: str) -> dict | None:
    """Parse flat-format judge response."""
    result = {}

    patterns = {
        "completeness": r"COMPLETENESS:\s*(\d+)",
        "accuracy": r"ACCURACY:\s*(\d+)",
        "scientific": r"SCIENTIFIC:\s*(\d+)",
        "overall": r"OVERALL:\s*(\d+)",
        "issues": r"ISSUES:\s*(.+)",
        "summary": r"SUMMARY:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key in ["completeness", "accuracy", "scientific", "overall"]:
                try:
                    result[key] = int(value)
                except ValueError:
                    pass
            else:
                result[key] = value

    # Need at least overall score
    if "overall" in result:
        return result
    return None


def run_judge_test(n_samples: int = 3):
    """Run judge on a sample of mock results."""

    print("=" * 60)
    print("LLM JUDGE TEST ON MOCK RESPONSES")
    print("=" * 60)

    # Load results
    results_dir = PROJECT_ROOT / "results" / "full_benchmark"
    results_files = list(results_dir.glob("results_*.json"))

    if not results_files:
        print("No benchmark results found!")
        return

    results_file = sorted(results_files)[-1]
    print(f"\nLoading results from: {results_file.name}")

    with open(results_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Total results: {len(results)}")

    # Get unique test cases
    unique_cases = {}
    for r in results:
        tc_id = r["test_case_id"]
        if tc_id not in unique_cases:
            unique_cases[tc_id] = r

    print(f"Unique test cases: {len(unique_cases)}")
    samples = list(unique_cases.values())[:n_samples]

    print("\nInitializing Gemini 2.5 Pro judge...")
    try:
        judge = GeminiClient(model="gemini-2.5-pro")
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return

    print(f"\nEvaluating {len(samples)} samples...")
    print("-" * 60)

    judge_results = []

    for i, result in enumerate(samples, 1):
        test_case_id = result["test_case_id"]
        print(f"\n[{i}/{len(samples)}] Evaluating: {test_case_id}")
        print(f"  Model: {result['model']}")
        print(f"  Condition: {result['condition']}")

        gt = load_ground_truth(test_case_id)
        if not gt:
            continue

        metrics = {
            "hierarchy_f1": result.get("hierarchy_f1", 0),
            "structure_accuracy": result.get("structure_accuracy", 0),
            "critical_gate_recall": result.get("critical_gate_recall", 0),
            "parse_success": result.get("parse_success", False),
        }

        prompt = build_judge_prompt(
            test_case_id=test_case_id,
            predicted_response=result.get("raw_response", ""),
            ground_truth=gt,
            metrics=metrics,
        )

        print("  Calling Gemini judge...")
        try:
            # Use 2000 tokens to account for Gemini 2.5 Pro thinking
            response = judge.call(prompt, max_tokens=2000, temperature=0.0)
            print(f"  Tokens used: {response.tokens_used}")

            judgment = parse_judge_response(response.content)

            if judgment:
                print(f"  Overall score: {judgment.get('overall', 'N/A')}/10")
                print(f"  Issues: {judgment.get('issues', 'N/A')}")
                print(f"  Summary: {judgment.get('summary', 'N/A')[:80]}...")

                judge_results.append({
                    "test_case_id": test_case_id,
                    "model": result["model"],
                    "auto_metrics": metrics,
                    "judge_scores": judgment,
                })
            else:
                print("  Failed to parse response")
                print(f"  Raw: {response.content[:200]}...")

        except Exception as e:
            print(f"  Error calling judge: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("JUDGE RESULTS SUMMARY")
    print("=" * 60)

    if judge_results:
        avg_scores = {}
        for jr in judge_results:
            scores = jr.get("judge_scores", {})
            for key in ["completeness", "accuracy", "scientific", "overall"]:
                if key in scores and isinstance(scores[key], (int, float)):
                    avg_scores.setdefault(key, []).append(scores[key])

        print("\nAverage Judge Scores:")
        for key, values in avg_scores.items():
            avg = sum(values) / len(values) if values else 0
            print(f"  {key}: {avg:.1f}/10")

        # Collect issues
        all_issues = []
        for jr in judge_results:
            issues = jr.get("judge_scores", {}).get("issues", "")
            if issues and issues.lower() != "none":
                all_issues.append(f"  - {jr['test_case_id']}: {issues}")

        if all_issues:
            print("\nIssues identified:")
            for issue in all_issues:
                print(issue)

        # Save results
        output_path = PROJECT_ROOT / "results" / "judge_test_results.json"
        with open(output_path, "w") as f:
            json.dump({
                "n_samples": len(samples),
                "judge_model": "gemini-2.5-pro",
                "results": judge_results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        print("\nNo successful judge evaluations.")

    return judge_results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run_judge_test(n_samples=n)
