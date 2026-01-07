#!/usr/bin/env python3
"""
Model Comparison Test: Does better reasoning reduce MCP benefit?

Research Question:
- Sonnet + MCP beats Sonnet alone (confirmed)
- Does Opus alone beat Sonnet + MCP?
- Does Opus + MCP beat Opus alone?

This tests whether tool use is a substitute for or complement to reasoning.

Usage:
    ANTHROPIC_API_KEY=sk-... python run_model_comparison.py
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import anthropic
from omip_panels import get_omip_panel, get_markers_only
from flow_panel_optimizer.mcp.server import MCP_TOOLS, execute_tool, analyze_panel


@dataclass
class TestCondition:
    name: str
    model: str
    use_tools: bool


# Test matrix
CONDITIONS = [
    TestCondition("Sonnet (no tools)", "claude-sonnet-4-20250514", False),
    TestCondition("Sonnet + MCP", "claude-sonnet-4-20250514", True),
    TestCondition("Opus (no tools)", "claude-opus-4-20250514", False),
    TestCondition("Opus + MCP", "claude-opus-4-20250514", True),
]


CONTROL_PROMPT = """You are an expert flow cytometry panel designer. Design a panel for these markers on a {instrument}.

## Markers:
{marker_list}

## Instructions:
Assign a fluorophore to each marker. Minimize spectral overlap between fluorophores.
Consider brightness matching (bright dyes for low expression, dim dyes for high expression).

## Output as markdown table:
| Marker | Fluorophore | Rationale |
|--------|-------------|-----------|

Include ALL {n_markers} markers."""


TREATMENT_PROMPT = """You are an expert flow cytometry panel designer with spectral analysis tools.

## Markers:
{marker_list}

## Instrument: {instrument}

## Available Tools:
1. **suggest_fluorophores(existing_panel, expression_level)** - Get ranked options
2. **check_compatibility(candidate, existing_panel)** - Verify choice
3. **analyze_panel(fluorophores)** - Get complexity index

## Process:
1. For each marker, use suggest_fluorophores to get options
2. Use check_compatibility to verify
3. After all assigned, use analyze_panel to check quality
4. If complexity > 2.0 or critical pairs exist, revise

## Output as markdown table:
| Marker | Fluorophore | Compatibility |
|--------|-------------|---------------|

Include ALL {n_markers} markers. End with analyze_panel results."""


def format_markers(markers: list[dict]) -> str:
    lines = []
    for m in markers:
        line = f"- {m['name']}: {m['expression']} expression"
        if m.get('notes'):
            line += f" ({m['notes']})"
        lines.append(line)
    return "\n".join(lines)


def call_model(prompt: str, model: str, use_tools: bool) -> tuple[str, float, int]:
    """Call a model with or without tools."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    tool_calls = 0

    start = time.time()

    if not use_tools:
        response = client.messages.create(
            model=model,
            max_tokens=3000,
            messages=messages
        )
        elapsed = time.time() - start
        return response.content[0].text, elapsed, 0

    # With tools - loop until done
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            tools=MCP_TOOLS,
            messages=messages,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            text_blocks = [b for b in response.content if b.type == "text"]
            final_text = "\n".join(b.text for b in text_blocks)
            break

        # Execute tools
        tool_results = []
        for tool_use in tool_use_blocks:
            result = execute_tool(tool_use.name, tool_use.input)
            tool_calls += 1
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        if tool_calls > 50:
            break

    elapsed = time.time() - start

    text_blocks = [b for b in response.content if b.type == "text"]
    final_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""

    return final_text, elapsed, tool_calls


def parse_panel(response: str) -> list[dict]:
    """Parse markdown table to assignments."""
    assignments = []
    for line in response.split("\n"):
        if not line.strip().startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2:
            marker, fluor = parts[0], parts[1]
            if marker.lower() in ["marker", "---"] or "---" in marker:
                continue
            marker = marker.replace("**", "").strip()
            fluor = fluor.replace("**", "").strip()
            if marker and fluor:
                assignments.append({"marker": marker, "fluorophore": fluor})
    return assignments


def run_condition(condition: TestCondition, markers: list[dict], instrument: str) -> dict:
    """Run a single test condition."""
    print(f"\n{'='*60}")
    print(f"Running: {condition.name}")
    print(f"Model: {condition.model}, Tools: {condition.use_tools}")
    print(f"{'='*60}")

    if condition.use_tools:
        prompt = TREATMENT_PROMPT.format(
            marker_list=format_markers(markers),
            instrument=instrument,
            n_markers=len(markers),
        )
    else:
        prompt = CONTROL_PROMPT.format(
            marker_list=format_markers(markers),
            instrument=instrument,
            n_markers=len(markers),
        )

    response, elapsed, tool_calls = call_model(prompt, condition.model, condition.use_tools)

    print(f"Response time: {elapsed:.1f}s")
    if tool_calls:
        print(f"Tool calls: {tool_calls}")

    assignments = parse_panel(response)
    print(f"Parsed {len(assignments)} assignments")

    if assignments:
        fluorophores = [a["fluorophore"] for a in assignments]
        metrics = analyze_panel(fluorophores)
    else:
        metrics = {"complexity_index": None, "max_similarity": None,
                   "critical_pairs": [], "problematic_pairs": []}

    return {
        "condition": condition.name,
        "model": condition.model,
        "use_tools": condition.use_tools,
        "assignments": assignments,
        "metrics": metrics,
        "elapsed": elapsed,
        "tool_calls": tool_calls,
    }


def print_comparison(results: list[dict]):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON RESULTS")
    print("=" * 90)

    print(f"\n{'Condition':<25} {'CI':>10} {'MaxSim':>10} {'Critical':>10} {'Time':>10} {'Tools':>8}")
    print("-" * 90)

    for r in results:
        m = r["metrics"]
        ci = m.get("complexity_index")
        ci_str = f"{ci:.2f}" if ci is not None else "N/A"
        max_sim = m.get("max_similarity")
        sim_str = f"{max_sim:.3f}" if max_sim is not None else "N/A"
        crit = len(m.get("critical_pairs", []))

        print(f"{r['condition']:<25} {ci_str:>10} {sim_str:>10} {crit:>10} {r['elapsed']:>9.1f}s {r['tool_calls']:>8}")

    print("-" * 90)

    # Find winners
    valid = [r for r in results if r["metrics"].get("complexity_index") is not None]
    if valid:
        best_ci = min(valid, key=lambda x: x["metrics"]["complexity_index"])
        print(f"\n★ Lowest Complexity Index: {best_ci['condition']} (CI={best_ci['metrics']['complexity_index']:.2f})")

        best_sim = min(valid, key=lambda x: x["metrics"]["max_similarity"])
        print(f"★ Lowest Max Similarity: {best_sim['condition']} (MaxSim={best_sim['metrics']['max_similarity']:.3f})")

    # Key comparisons
    print("\n" + "=" * 90)
    print("KEY COMPARISONS")
    print("=" * 90)

    def get_ci(name):
        for r in results:
            if r["condition"] == name:
                return r["metrics"].get("complexity_index")
        return None

    sonnet_no = get_ci("Sonnet (no tools)")
    sonnet_mcp = get_ci("Sonnet + MCP")
    opus_no = get_ci("Opus (no tools)")
    opus_mcp = get_ci("Opus + MCP")

    if sonnet_no and sonnet_mcp:
        improvement = (sonnet_no - sonnet_mcp) / sonnet_no * 100
        print(f"\n1. MCP benefit for Sonnet: {improvement:.0f}% lower CI")
        print(f"   Sonnet alone: {sonnet_no:.2f} → Sonnet+MCP: {sonnet_mcp:.2f}")

    if opus_no and opus_mcp:
        improvement = (opus_no - opus_mcp) / opus_no * 100 if opus_no > 0 else 0
        print(f"\n2. MCP benefit for Opus: {improvement:.0f}% lower CI")
        print(f"   Opus alone: {opus_no:.2f} → Opus+MCP: {opus_mcp:.2f}")

    if opus_no and sonnet_mcp:
        if opus_no < sonnet_mcp:
            print(f"\n3. Opus alone BEATS Sonnet+MCP!")
            print(f"   → Better reasoning may substitute for tools")
        else:
            print(f"\n3. Sonnet+MCP beats Opus alone")
            print(f"   → Tools provide value even vs stronger model")

    if opus_no and sonnet_no:
        improvement = (sonnet_no - opus_no) / sonnet_no * 100
        print(f"\n4. Opus vs Sonnet (no tools): {improvement:.0f}% {'better' if improvement > 0 else 'worse'}")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Use OMIP-030 (10 markers, manageable size)
    panel = get_omip_panel(30)
    markers = get_markers_only(panel)

    print(f"Testing with {panel.name}")
    print(f"Markers: {len(markers)}")
    print(f"Conditions: {len(CONDITIONS)}")

    results = []
    for condition in CONDITIONS:
        try:
            result = run_condition(condition, markers, panel.instrument)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {condition.name} failed: {e}")
            import traceback
            traceback.print_exc()

    print_comparison(results)

    # Cost estimate
    print("\n" + "=" * 90)
    print("COST ESTIMATE")
    print("=" * 90)
    sonnet_cost = sum(r["elapsed"] * 0.003/60 for r in results if "sonnet" in r["model"].lower())
    opus_cost = sum(r["elapsed"] * 0.015/60 for r in results if "opus" in r["model"].lower())
    print(f"Estimated Sonnet cost: ${sonnet_cost:.3f}")
    print(f"Estimated Opus cost: ${opus_cost:.3f}")


if __name__ == "__main__":
    main()
