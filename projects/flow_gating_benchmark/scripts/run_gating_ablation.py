#!/usr/bin/env python3
"""
Run gating hierarchy ablation study.

Tests all combinations of:
- Reasoning: direct, cot, wot
- Context: minimal, standard, rich
- RAG: none, oracle (HIPC definitions)

With 3 test cases (1 per difficulty level: simple, medium, complex).

Usage:
    python scripts/run_gating_ablation.py           # Interactive mode
    python scripts/run_gating_ablation.py --force   # Skip confirmation
    python scripts/run_gating_ablation.py --dry-run # Show cost only
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add project paths
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / "src"))
sys.path.insert(0, str(project_dir.parent.parent / "libs"))

from curation.schemas import TestCase
from curation.omip_extractor import load_test_case
from evaluation.scorer import GatingScorer

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


# HIPC 2016 reference definitions for oracle RAG
HIPC_DEFINITIONS = """
## HIPC 2016 Standardized Cell Definitions
Source: https://www.nature.com/articles/srep20686

### Key Population Definitions:

**T cells**: CD3+ CD19- (negative for B cell markers when in panel)
- CD4+ T cells: CD3+ CD4+ CD8-
- CD8+ T cells: CD3+ CD4- CD8+
- Naive: CD45RA+ CCR7+
- Central Memory: CD45RA- CCR7+
- Effector Memory: CD45RA- CCR7-
- TEMRA: CD45RA+ CCR7-

**B cells**: CD3- CD19+ (or CD20+, either acceptable)
- Naive B: CD19+ IgD+ CD27-
- Memory B: CD19+ CD27+

**NK cells**: CD3- CD56+ and/or CD16+
- CD56bright: CD3- CD56bright CD16dim/-
- CD56dim: CD3- CD56dim CD16+

**Monocytes**: CD14+
- Classical: CD14++ CD16-
- Non-classical: CD14dim CD16++

Note: HIPC recommends going directly to lineage markers rather than
using FSC/SSC lymphocyte gates, as this reduces variability.
"""


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_test_cases(config: dict, data_dir: Path) -> list[TestCase]:
    """Load specified test cases from config."""
    test_cases = []
    for filename in config.get("test_cases", []):
        path = data_dir / "synthetic" / filename
        if path.exists():
            tc = load_test_case(path)
            test_cases.append(tc)
            print(f"  Loaded: {tc.test_case_id} ({tc.complexity.value}, {tc.n_colors} colors)")
        else:
            print(f"  Warning: {filename} not found")
    return test_cases


def build_prompt(
    test_case: TestCase,
    reasoning: str,
    context_level: str,
    rag_mode: str,
    config: dict,
) -> str:
    """Build prompt for a specific condition."""

    # Build context based on level
    if context_level == "minimal":
        markers = ", ".join(test_case.panel.markers)
        context = f"Markers: {markers}"
    elif context_level == "standard":
        context = test_case.to_prompt_context("standard")
    else:  # rich
        context = test_case.to_prompt_context("rich")

    # Base instruction
    base = f"""Given this flow cytometry panel, predict the gating hierarchy.

{context}

Provide the gating hierarchy as a JSON tree structure with this format:
{{
  "name": "All Events",
  "children": [
    {{
      "name": "Gate Name",
      "markers": ["marker1", "marker2"],
      "marker_logic": [{{"marker": "CD3", "positive": true}}],
      "children": [...]
    }}
  ]
}}
"""

    # Add reasoning instruction
    strategy_configs = config.get("strategy_configs", {})

    if reasoning == "cot":
        cot_config = strategy_configs.get("cot", {})
        reasoning_prompt = cot_config.get("reasoning_prompt", "Think through this step by step.")
        base = f"{base}\n{reasoning_prompt}\n\nShow your reasoning, then provide the final gating hierarchy JSON."
    elif reasoning == "wot":
        base = f"""{base}

For key gating decisions, consider multiple approaches:

1. **Lymphocyte gating**:
   - Option A: Use FSC/SSC scatter to gate lymphocytes first
   - Option B: Go directly to lineage markers from live cells
   - Which is more robust for this panel?

2. **T cell definition**:
   - Option A: CD3+ only
   - Option B: CD3+ CD19- (exclude B cells)
   - Given markers in panel, which is more precise?

For each decision, briefly explain your choice.
Then provide the final gating hierarchy JSON.
"""

    # Add RAG context if oracle mode
    if rag_mode == "oracle":
        base = f"{HIPC_DEFINITIONS}\n\n{base}"

    return base


def estimate_cost(config: dict, n_test_cases: int) -> dict:
    """Estimate experiment cost."""
    n_models = len(config.get("models", []))
    n_reasoning = len(config.get("reasoning_types", []))
    n_context = len(config.get("context_levels", []))
    n_rag = len(config.get("rag_modes", []))
    n_tools = len(config.get("tool_configs", []))
    n_bootstrap = config.get("n_bootstrap_runs", 1)

    n_conditions = n_models * n_reasoning * n_context * n_rag * n_tools
    n_calls = n_conditions * n_test_cases * n_bootstrap

    # Estimate tokens (gating prompts are larger than synthetic)
    est_input_tokens = n_calls * 1500  # ~1500 tokens per prompt
    est_output_tokens = n_calls * 800  # ~800 tokens per response

    # Sonnet pricing: $3/M input, $15/M output
    input_cost = (est_input_tokens / 1_000_000) * 3
    output_cost = (est_output_tokens / 1_000_000) * 15

    return {
        "n_conditions": n_conditions,
        "n_test_cases": n_test_cases,
        "n_bootstrap": n_bootstrap,
        "n_calls": n_calls,
        "est_input_tokens": est_input_tokens,
        "est_output_tokens": est_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def print_cost_estimate(estimate: dict) -> None:
    """Print formatted cost estimate."""
    print("=" * 55)
    print("COST ESTIMATE")
    print("=" * 55)
    print(f"Conditions:        {estimate['n_conditions']}")
    print(f"Test cases:        {estimate['n_test_cases']}")
    print(f"Bootstrap runs:    {estimate['n_bootstrap']}")
    print(f"Total API calls:   {estimate['n_calls']}")
    print()
    print(f"Est. input tokens:  {estimate['est_input_tokens']:,}")
    print(f"Est. output tokens: {estimate['est_output_tokens']:,}")
    print()
    print(f"Input cost:    ${estimate['input_cost']:.2f}")
    print(f"Output cost:   ${estimate['output_cost']:.2f}")
    print("-" * 55)
    print(f"TOTAL COST:    ${estimate['total_cost']:.2f}")
    print("=" * 55)


def run_experiment(
    config: dict,
    test_cases: list[TestCase],
    api_key: str,
) -> dict:
    """Run the ablation experiment."""

    if not Anthropic:
        raise RuntimeError("anthropic package not installed")

    client = Anthropic(api_key=api_key)
    scorer = GatingScorer()

    results = []
    errors = []

    # Get bootstrap runs
    n_bootstrap = config.get("n_bootstrap_runs", 1)

    # Generate all conditions
    conditions = []
    for reasoning in config.get("reasoning_types", ["direct"]):
        for context in config.get("context_levels", ["standard"]):
            for rag in config.get("rag_modes", ["none"]):
                conditions.append({
                    "reasoning": reasoning,
                    "context": context,
                    "rag": rag,
                    "name": f"sonnet_{reasoning}_{context}_{rag}",
                })

    total = len(conditions) * len(test_cases) * n_bootstrap
    completed = 0

    for bootstrap_run in range(n_bootstrap):
        for test_case in test_cases:
            for cond in conditions:
                completed += 1
                pct = (completed / total) * 100

                run_label = f" (run {bootstrap_run + 1}/{n_bootstrap})" if n_bootstrap > 1 else ""
                print(f"[{pct:5.1f}%] {cond['name']} / {test_case.test_case_id}{run_label}...", end=" ", flush=True)

                try:
                    # Build prompt
                    prompt = build_prompt(
                        test_case=test_case,
                        reasoning=cond["reasoning"],
                        context_level=cond["context"],
                        rag_mode=cond["rag"],
                        config=config,
                    )

                    # Call API
                    start_time = datetime.now()
                    response = client.messages.create(
                        model=config.get("models", ["claude-sonnet-4-20250514"])[0],
                        max_tokens=config.get("max_tokens", 4096),
                        temperature=config.get("temperature", 0.0),
                        messages=[{"role": "user", "content": prompt}],
                    )
                    latency = (datetime.now() - start_time).total_seconds()

                    response_text = response.content[0].text

                    # Score the response
                    score_result = scorer.score(
                        response=response_text,
                        test_case=test_case,
                        model="claude-sonnet-4-20250514",
                        condition=cond["name"],
                    )

                    # Get hallucination rate from evaluation object
                    halluc_rate = 0.0
                    if score_result.evaluation:
                        halluc_rate = score_result.evaluation.hallucination_rate

                    results.append({
                        "test_case_id": test_case.test_case_id,
                        "complexity": test_case.complexity.value,
                        "n_colors": test_case.n_colors,
                        "condition": cond["name"],
                        "reasoning": cond["reasoning"],
                        "context": cond["context"],
                        "rag": cond["rag"],
                        "bootstrap_run": bootstrap_run,
                        "hierarchy_f1": score_result.hierarchy_f1,
                        "structure_accuracy": score_result.structure_accuracy,
                        "critical_gate_recall": score_result.critical_gate_recall,
                        "hallucination_rate": halluc_rate,
                        "parse_success": score_result.parse_success,
                        "latency": latency,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    })

                    if score_result.parse_success:
                        print(f"F1={score_result.hierarchy_f1:.2f}, {latency:.1f}s")
                    else:
                        print(f"PARSE_ERROR, {latency:.1f}s")

                except Exception as e:
                    print(f"ERROR: {e}")
                    errors.append({
                        "test_case_id": test_case.test_case_id,
                        "condition": cond["name"],
                        "bootstrap_run": bootstrap_run,
                        "error": str(e),
                    })

    return {
        "config": config,
        "n_bootstrap_runs": n_bootstrap,
        "results": results,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }


def save_results(results: dict, output_dir: Path) -> None:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"gating_ablation_results_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if results["results"]:
        # Group by condition
        by_condition = {}
        for r in results["results"]:
            cond = r["condition"]
            if cond not in by_condition:
                by_condition[cond] = []
            by_condition[cond].append(r)

        for cond, cond_results in by_condition.items():
            successful = [r for r in cond_results if r["parse_success"]]
            if successful:
                avg_f1 = sum(r["hierarchy_f1"] for r in successful) / len(successful)
                avg_latency = sum(r["latency"] for r in successful) / len(successful)
                print(f"\n{cond}:")
                print(f"  F1: {avg_f1:.3f}")
                print(f"  Parse rate: {len(successful)}/{len(cond_results)}")
                print(f"  Avg latency: {avg_latency:.1f}s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run gating hierarchy ablation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cost confirmation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cost estimate only",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/gating_ablation.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the gating ablation experiment."""
    args = parse_args()

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = project_dir / "configs" / "gating_ablation.yaml"

    config = load_config(config_path)

    print(f"Loaded config: {config['name']}")
    print(f"Data source: {config.get('data_source', 'unknown')}")

    # Load test cases
    data_dir = project_dir / "data" / "verified"
    print(f"\nLoading test cases from {data_dir}:")
    test_cases = load_test_cases(config, data_dir)

    if not test_cases:
        print("ERROR: No test cases found!")
        return 1

    # Generate conditions
    n_conditions = (
        len(config.get("reasoning_types", [])) *
        len(config.get("context_levels", [])) *
        len(config.get("rag_modes", []))
    )

    print(f"\nConditions to test: {n_conditions}")
    for r in config.get("reasoning_types", []):
        for c in config.get("context_levels", []):
            for g in config.get("rag_modes", []):
                print(f"  - sonnet_{r}_{c}_{g}")

    # Cost estimate
    estimate = estimate_cost(config, len(test_cases))
    print()
    print_cost_estimate(estimate)

    # Dry run - just show cost
    if args.dry_run:
        return 0

    # Cost confirmation
    if not args.force:
        print()
        response = input("Proceed with experiment? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return 1

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return 1

    # Run!
    print("\n" + "=" * 60)
    print(f"RUNNING: {config['name']}")
    print("=" * 60 + "\n")

    results = run_experiment(config, test_cases, api_key)

    # Save results
    output_dir = project_dir / config.get("output_dir", "results")
    save_results(results, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
