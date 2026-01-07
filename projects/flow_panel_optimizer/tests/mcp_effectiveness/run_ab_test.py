#!/usr/bin/env python3
"""
Run A/B test comparing Claude panel design with and without MCP guidance.

Usage:
    ANTHROPIC_API_KEY=sk-... python run_ab_test.py
"""
import os
import sys
import json
import time
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import anthropic
from test_mcp_effectiveness import (
    load_test_case,
    format_marker_list,
    parse_panel_from_response,
    evaluate_panel,
    OMIPReferencePanels,
    compare_to_omip,
    print_omip_comparison,
)

# Load prompts
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template."""
    return (PROMPTS_DIR / f"{name}_prompt.txt").read_text()


def call_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Claude API and return response."""
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def run_condition(test_case: dict, condition: str) -> dict:
    """Run a single test condition (control or treatment)."""
    prompt_template = load_prompt(condition)
    marker_list = format_marker_list(test_case["markers"])

    prompt = prompt_template.format(
        instrument=test_case["instrument"],
        marker_list=marker_list
    )

    print(f"\n{'='*60}")
    print(f"Running {condition.upper()} condition for: {test_case['name']}")
    print(f"Markers: {len(test_case['markers'])}")
    print(f"{'='*60}")

    start_time = time.time()
    response = call_claude(prompt)
    elapsed = time.time() - start_time

    print(f"\nResponse received in {elapsed:.1f}s")
    print(f"\n--- Claude's Response ---\n")
    print(response[:1500] + "..." if len(response) > 1500 else response)

    # Parse and evaluate
    panel = parse_panel_from_response(response)
    print(f"\n--- Parsed Panel ({len(panel)} assignments) ---")
    for item in panel:
        print(f"  {item['marker']}: {item['fluorophore']}")

    metrics = evaluate_panel(panel)

    return {
        "condition": condition,
        "test_case": test_case["name"],
        "response": response,
        "panel": panel,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }


def run_ab_test(test_case_name: str = "basic_tcell"):
    """Run full A/B test for a single test case."""
    test_case = load_test_case(test_case_name)

    # Run both conditions
    control = run_condition(test_case, "control")
    treatment = run_condition(test_case, "treatment")

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Control':>15} {'Treatment':>15} {'Winner':>12}")
    print("-" * 70)

    def compare_metric(name, c_val, t_val, lower_is_better=True):
        if c_val is None and t_val is None:
            winner = "Tie"
        elif c_val is None:
            winner = "Treatment"
        elif t_val is None:
            winner = "Control"
        elif lower_is_better:
            winner = "Control" if c_val < t_val else ("Treatment" if t_val < c_val else "Tie")
        else:
            winner = "Control" if c_val > t_val else ("Treatment" if t_val > c_val else "Tie")

        c_str = f"{c_val:.4f}" if isinstance(c_val, float) else str(c_val)
        t_str = f"{t_val:.4f}" if isinstance(t_val, float) else str(t_val)
        print(f"{name:<25} {c_str:>15} {t_str:>15} {winner:>12}")
        return winner

    c_metrics = control["metrics"]
    t_metrics = treatment["metrics"]

    winners = []
    winners.append(compare_metric("Complexity Index", c_metrics.get("complexity_index", 0), t_metrics.get("complexity_index", 0)))
    winners.append(compare_metric("Max Similarity", c_metrics.get("max_similarity", 0), t_metrics.get("max_similarity", 0)))
    winners.append(compare_metric("High-Risk Pairs", c_metrics.get("high_risk_pairs", 0), t_metrics.get("high_risk_pairs", 0)))
    winners.append(compare_metric("Critical Pairs", c_metrics.get("critical_pairs", 0), t_metrics.get("critical_pairs", 0)))
    winners.append(compare_metric("Valid Fluorophores", c_metrics.get("valid_fluorophores", len(control["panel"])), t_metrics.get("valid_fluorophores", len(treatment["panel"])), lower_is_better=False))

    print("-" * 70)
    control_wins = winners.count("Control")
    treatment_wins = winners.count("Treatment")
    print(f"\nOverall: Control wins {control_wins}, Treatment wins {treatment_wins}")

    # OMIP Comparison
    print("\n" + "=" * 70)
    print("OMIP REFERENCE COMPARISON")
    print("=" * 70)

    # Find matching OMIP panel if available
    omip = OMIPReferencePanels.OMIP_030
    print(f"\nComparing to {omip['name']}: {omip['url']}")

    if control["panel"]:
        print("\n--- Control vs OMIP-030 ---")
        control_comparison = compare_to_omip(control["panel"], omip)
        print_omip_comparison(control_comparison)

    if treatment["panel"]:
        print("\n--- Treatment vs OMIP-030 ---")
        treatment_comparison = compare_to_omip(treatment["panel"], omip)
        print_omip_comparison(treatment_comparison)

    return {"control": control, "treatment": treatment}


def main():
    """Run A/B tests on all test cases."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    test_cases = ["basic_tcell"]  # Start with one

    all_results = {}
    for tc in test_cases:
        try:
            results = run_ab_test(tc)
            all_results[tc] = results
        except Exception as e:
            print(f"ERROR running {tc}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for tc, results in all_results.items():
        c = results["control"]["metrics"]
        t = results["treatment"]["metrics"]
        print(f"\n{tc}:")
        print(f"  Control:   CI={c['complexity_index']:.2f}, MaxSim={c['max_similarity']:.4f}, HighRisk={c['high_risk_pairs']}")
        print(f"  Treatment: CI={t['complexity_index']:.2f}, MaxSim={t['max_similarity']:.4f}, HighRisk={t['high_risk_pairs']}")


if __name__ == "__main__":
    main()
