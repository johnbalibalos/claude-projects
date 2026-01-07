#!/usr/bin/env python3
"""
OMIP Comparison Test: Can Claude design panels as good as published OMIPs?

This test:
1. Gives Claude ONLY the marker list from a published OMIP panel
2. Runs two conditions:
   - Control: Claude designs panel with general knowledge
   - Treatment: Claude designs panel with MCP spectral tools
3. Compares both to the published OMIP assignments
4. Evaluates if either can IMPROVE on the published panel

Usage:
    ANTHROPIC_API_KEY=sk-... python run_omip_comparison.py [--omip 30] [--runs 3]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import anthropic
from omip_panels import (
    get_omip_panel,
    get_markers_only,
    get_published_assignments,
    ALL_OMIP_PANELS,
    OMIPPanel,
)
from flow_panel_optimizer.mcp.server import (
    MCP_TOOLS,
    execute_tool,
    analyze_panel,
)


def format_markers_for_prompt(markers: list[dict]) -> str:
    """Format marker list for Claude prompt."""
    lines = []
    for m in markers:
        line = f"- **{m['name']}**: {m['expression']} expression"
        if m.get('notes'):
            line += f" ({m['notes']})"
        if m.get('intracellular'):
            line += " [intracellular]"
        lines.append(line)
    return "\n".join(lines)


CONTROL_PROMPT = """You are an expert flow cytometry panel designer. Design a flow cytometry panel for the following markers.

## Instrument Configuration
{instrument}

## Markers to include (design fluorophore assignments for EACH marker):

{marker_list}

## Instructions:

For each marker, assign an appropriate fluorophore. Consider:
1. **Spectral overlap**: Avoid fluorophores with similar emission spectra
2. **Brightness matching**: Match fluorophore brightness to antigen expression level
   - High expression markers can use dimmer fluorophores (FITC, PerCP)
   - Low expression markers need brighter fluorophores (PE, APC, BV421)
3. **Laser lines**: Ensure fluorophores match available lasers

## Output your panel as a markdown table:

| Marker | Fluorophore | Rationale |
|--------|-------------|-----------|
| CD3    | BV421       | High expression, bright violet |
| ...    | ...         | ... |

Include ALL {n_markers} markers in your table. After the table, rate overall panel quality.
"""

TREATMENT_PROMPT = """You are an expert flow cytometry panel designer with access to spectral analysis tools.

## Instrument Configuration
{instrument}

## Markers to include (design fluorophore assignments for EACH marker):

{marker_list}

## Available Tools:

You have access to these MCP tools for panel design. USE THEM.

1. **suggest_fluorophores(existing_panel, expression_level)**: Get ranked fluorophore suggestions
   - existing_panel: list of fluorophores already selected (start with [])
   - expression_level: "high", "medium", or "low"

2. **check_compatibility(candidate, existing_panel)**: Verify a fluorophore choice
   - Returns similarity score and "SAFE"/"CAUTION"/"AVOID" recommendation

3. **analyze_panel(fluorophores)**: Analyze complete panel for conflicts
   - Returns complexity index and problematic pairs

## Design Process:

Build your panel step-by-step:
1. Start with high-expression markers (more flexibility)
2. For EACH marker, call suggest_fluorophores to get options
3. Call check_compatibility to verify your choice
4. After all markers assigned, call analyze_panel to check quality
5. If quality is poor, use find_alternatives to improve

## Output Format:

Show your tool usage, then provide final panel as a markdown table:

| Marker | Fluorophore | Expression | Compatibility |
|--------|-------------|------------|---------------|
| CD3    | BV421       | high       | SAFE (max sim: 0.45) |
| ...    | ...         | ...        | ... |

Include ALL {n_markers} markers. End with final analyze_panel results.
"""


def call_claude_control(markers: list[dict], instrument: str) -> tuple[str, float]:
    """Call Claude without tools (control condition)."""
    client = anthropic.Anthropic()

    prompt = CONTROL_PROMPT.format(
        instrument=instrument,
        marker_list=format_markers_for_prompt(markers),
        n_markers=len(markers),
    )

    start = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed = time.time() - start

    return response.content[0].text, elapsed


def call_claude_with_tools(markers: list[dict], instrument: str) -> tuple[str, float, list]:
    """Call Claude with MCP tools (treatment condition)."""
    client = anthropic.Anthropic()

    prompt = TREATMENT_PROMPT.format(
        instrument=instrument,
        marker_list=format_markers_for_prompt(markers),
        n_markers=len(markers),
    )

    messages = [{"role": "user", "content": prompt}]
    tool_calls = []

    start = time.time()

    # Loop to handle tool calls
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            tools=MCP_TOOLS,
            messages=messages,
        )

        # Check if we have tool use
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No more tool calls, get final text
            text_blocks = [b for b in response.content if b.type == "text"]
            final_text = "\n".join(b.text for b in text_blocks)
            break

        # Execute each tool call
        tool_results = []
        for tool_use in tool_use_blocks:
            result = execute_tool(tool_use.name, tool_use.input)
            tool_calls.append({
                "tool": tool_use.name,
                "input": tool_use.input,
                "output": result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result),
            })

        # Add assistant response and tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # Safety limit
        if len(tool_calls) > 60:
            print("Warning: Exceeded tool call limit")
            break

    elapsed = time.time() - start

    # Get final text from last response
    text_blocks = [b for b in response.content if b.type == "text"]
    final_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""

    return final_text, elapsed, tool_calls


def parse_panel_table(response: str) -> list[dict]:
    """Parse a markdown table into marker-fluorophore assignments."""
    assignments = []

    lines = response.split("\n")
    in_table = False

    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            in_table = False
            continue

        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]  # Remove empty

        if len(parts) >= 2:
            marker = parts[0]
            fluor = parts[1]

            # Skip header rows
            if marker.lower() in ["marker", "---", "-----"]:
                in_table = True
                continue
            if "---" in marker or "---" in fluor:
                continue

            # Clean up
            marker = marker.replace("**", "").strip()
            fluor = fluor.replace("**", "").strip()

            if marker and fluor and not marker.startswith("-"):
                assignments.append({
                    "marker": marker,
                    "fluorophore": fluor,
                })

    return assignments


def evaluate_panel(assignments: list[dict]) -> dict:
    """Evaluate a panel using the MCP analyze_panel tool."""
    fluorophores = [a["fluorophore"] for a in assignments]
    return analyze_panel(fluorophores)


def compare_to_published(
    generated: list[dict],
    published: dict[str, str],
    panel_name: str
) -> dict:
    """Compare generated panel to published OMIP assignments."""
    gen_by_marker = {a["marker"]: a["fluorophore"] for a in generated}

    matches = 0
    differences = []
    missing = []

    for marker, pub_fluor in published.items():
        if marker not in gen_by_marker:
            missing.append(marker)
            continue

        gen_fluor = gen_by_marker[marker]
        if gen_fluor == pub_fluor:
            matches += 1
        else:
            differences.append({
                "marker": marker,
                "generated": gen_fluor,
                "published": pub_fluor,
            })

    total = len(published)
    match_rate = matches / total if total > 0 else 0

    return {
        "panel": panel_name,
        "total_markers": total,
        "matches": matches,
        "match_rate": round(match_rate * 100, 1),
        "differences": differences,
        "missing_markers": missing,
    }


def run_single_comparison(panel: OMIPPanel, condition: str) -> dict:
    """Run a single condition (control or treatment) on an OMIP panel."""
    markers = get_markers_only(panel)
    published = get_published_assignments(panel)

    print(f"\n{'='*70}")
    print(f"Running {condition.upper()} condition for OMIP-{panel.omip_number:03d}")
    print(f"Markers: {len(markers)}")
    print(f"{'='*70}")

    if condition == "control":
        response, elapsed = call_claude_control(markers, panel.instrument)
        tool_calls = []
    else:
        response, elapsed, tool_calls = call_claude_with_tools(markers, panel.instrument)

    print(f"\nResponse received in {elapsed:.1f}s")
    if tool_calls:
        print(f"Tool calls made: {len(tool_calls)}")

    # Parse the panel
    assignments = parse_panel_table(response)
    print(f"Parsed {len(assignments)} assignments")

    # Evaluate
    metrics = evaluate_panel(assignments)
    comparison = compare_to_published(assignments, published, f"OMIP-{panel.omip_number:03d}")

    # Also evaluate the published panel
    published_list = [{"marker": m, "fluorophore": f} for m, f in published.items()]
    published_metrics = evaluate_panel(published_list)

    return {
        "condition": condition,
        "omip": panel.omip_number,
        "response": response[:2000] + "..." if len(response) > 2000 else response,
        "assignments": assignments,
        "metrics": metrics,
        "published_metrics": published_metrics,
        "comparison": comparison,
        "elapsed": elapsed,
        "tool_calls": len(tool_calls) if tool_calls else 0,
    }


def print_comparison_report(control: dict, treatment: dict):
    """Print formatted comparison report."""
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    # Metrics comparison
    c_m = control["metrics"]
    t_m = treatment["metrics"]
    p_m = control["published_metrics"]  # Same for both conditions

    print(f"\n{'Metric':<25} {'Control':>12} {'Treatment':>12} {'Published':>12} {'Best':>12}")
    print("-" * 80)

    def compare_row(name, c_val, t_val, p_val, lower_is_better=True):
        values = [("Control", c_val), ("Treatment", t_val), ("Published", p_val)]
        values = [(n, v) for n, v in values if v is not None]

        if lower_is_better:
            best = min(values, key=lambda x: x[1])[0]
        else:
            best = max(values, key=lambda x: x[1])[0]

        c_str = f"{c_val:.4f}" if isinstance(c_val, float) else str(c_val)
        t_str = f"{t_val:.4f}" if isinstance(t_val, float) else str(t_val)
        p_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)

        print(f"{name:<25} {c_str:>12} {t_str:>12} {p_str:>12} {best:>12}")
        return best

    winners = []
    winners.append(compare_row("Complexity Index",
                               c_m.get("complexity_index", 0),
                               t_m.get("complexity_index", 0),
                               p_m.get("complexity_index", 0)))
    winners.append(compare_row("Max Similarity",
                               c_m.get("max_similarity", 0),
                               t_m.get("max_similarity", 0),
                               p_m.get("max_similarity", 0)))
    winners.append(compare_row("Critical Pairs",
                               len(c_m.get("critical_pairs", [])),
                               len(t_m.get("critical_pairs", [])),
                               len(p_m.get("critical_pairs", []))))
    winners.append(compare_row("High-Risk Pairs",
                               len(c_m.get("problematic_pairs", [])),
                               len(t_m.get("problematic_pairs", [])),
                               len(p_m.get("problematic_pairs", []))))

    print("-" * 80)

    # Count wins
    control_wins = winners.count("Control")
    treatment_wins = winners.count("Treatment")
    published_wins = winners.count("Published")

    print(f"\nWins: Control={control_wins}, Treatment={treatment_wins}, Published={published_wins}")

    # Match comparison
    print(f"\n{'Match vs Published OMIP':<30}")
    print("-" * 50)
    c_comp = control["comparison"]
    t_comp = treatment["comparison"]
    print(f"Control:   {c_comp['matches']}/{c_comp['total_markers']} ({c_comp['match_rate']}%)")
    print(f"Treatment: {t_comp['matches']}/{t_comp['total_markers']} ({t_comp['match_rate']}%)")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if treatment_wins > control_wins:
        print("✓ Treatment (MCP tools) designed BETTER panel than Control")
    elif control_wins > treatment_wins:
        print("✗ Control (no tools) designed better panel than Treatment")
    else:
        print("= Control and Treatment performed equally")

    if treatment_wins > published_wins or control_wins > published_wins:
        better = "Treatment" if treatment_wins > control_wins else "Control"
        print(f"★ {better} IMPROVED on published OMIP panel!")
    else:
        print("⊡ Published OMIP panel remains optimal")

    return {
        "control_wins": control_wins,
        "treatment_wins": treatment_wins,
        "published_wins": published_wins,
    }


def main():
    parser = argparse.ArgumentParser(description="Run OMIP comparison tests")
    parser.add_argument("--omip", type=int, default=30, help="OMIP number to test")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per condition")
    parser.add_argument("--all", action="store_true", help="Test all available OMIPs")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    panels_to_test = list(ALL_OMIP_PANELS.keys()) if args.all else [args.omip]

    all_results = []

    for omip_num in panels_to_test:
        panel = get_omip_panel(omip_num)
        print(f"\n{'#' * 80}")
        print(f"# Testing: {panel.name}")
        print(f"# {panel.description}")
        print(f"# DOI: {panel.doi}")
        print(f"{'#' * 80}")

        for run in range(args.runs):
            if args.runs > 1:
                print(f"\n--- Run {run + 1}/{args.runs} ---")

            # Run both conditions
            control = run_single_comparison(panel, "control")
            treatment = run_single_comparison(panel, "treatment")

            # Print comparison
            result = print_comparison_report(control, treatment)
            result["omip"] = omip_num
            result["run"] = run
            all_results.append(result)

    # Final summary
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("AGGREGATE RESULTS")
        print("=" * 80)

        total_control = sum(r["control_wins"] for r in all_results)
        total_treatment = sum(r["treatment_wins"] for r in all_results)
        total_published = sum(r["published_wins"] for r in all_results)

        print(f"Total metric wins across all tests:")
        print(f"  Control:   {total_control}")
        print(f"  Treatment: {total_treatment}")
        print(f"  Published: {total_published}")

        if total_treatment > total_control:
            print(f"\n★ MCP tools improved panel design by {total_treatment - total_control} metric wins")
        elif total_control > total_treatment:
            print(f"\n✗ No tools performed better by {total_control - total_treatment} metric wins")


if __name__ == "__main__":
    main()
