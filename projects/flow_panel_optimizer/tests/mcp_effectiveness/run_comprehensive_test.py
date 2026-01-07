#!/usr/bin/env python3
"""
Comprehensive Model Comparison: Multi-OMIP, Multi-Run Analysis

Tests whether MCP tools improve panel design across:
- Multiple OMIP panels (different sizes/complexity)
- Multiple runs (variance estimation)
- Multiple models (Sonnet vs Opus)

Usage:
    ANTHROPIC_API_KEY=sk-... python run_comprehensive_test.py
"""

import os
import sys
import json
import time
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import anthropic
from omip_panels import get_omip_panel, get_markers_only, ALL_OMIP_PANELS
from flow_panel_optimizer.mcp.server import MCP_TOOLS, execute_tool, analyze_panel


@dataclass
class TestCondition:
    name: str
    model: str
    use_tools: bool


@dataclass
class TestResult:
    condition: str
    model: str
    use_tools: bool
    omip: int
    run: int
    complexity_index: Optional[float]
    max_similarity: Optional[float]
    critical_pairs: int
    high_risk_pairs: int
    n_assignments: int
    elapsed: float
    tool_calls: int


# Test conditions
CONDITIONS = [
    TestCondition("Sonnet", "claude-sonnet-4-20250514", False),
    TestCondition("Sonnet+MCP", "claude-sonnet-4-20250514", True),
    TestCondition("Opus", "claude-opus-4-20250514", False),
    TestCondition("Opus+MCP", "claude-opus-4-20250514", True),
]

# OMIPs to test (varying complexity)
OMIP_PANELS = [30, 47, 63]  # 10, 16, 20 markers


CONTROL_PROMPT = """You are an expert flow cytometry panel designer. Design a panel for these markers.

## Instrument: {instrument}

## Markers (assign a fluorophore to EACH):
{marker_list}

## Guidelines:
- Minimize spectral overlap between fluorophores
- Match brightness to expression level (bright dyes for low expression)
- Use different lasers when possible to reduce overlap
- Common fluorophores: BV421, BV510, BV605, BV711, BV785, FITC, PE, PE-Cy7, APC, APC-Cy7, PerCP-Cy5.5

## Output as markdown table with ALL {n_markers} markers:
| Marker | Fluorophore | Rationale |
|--------|-------------|-----------|
"""


TREATMENT_PROMPT = """You are an expert flow cytometry panel designer with spectral analysis tools.

## Instrument: {instrument}

## Markers (assign a fluorophore to EACH):
{marker_list}

## Available Tools - USE THEM:
1. **suggest_fluorophores(existing_panel, expression_level)** - Get ranked fluorophore options
2. **check_compatibility(candidate, existing_panel)** - Check if fluorophore fits (returns similarity score)
3. **analyze_panel(fluorophores)** - Get complexity index and problematic pairs

## Required Process:
1. Start with empty panel []
2. For each marker: call suggest_fluorophores, pick best, call check_compatibility
3. After ALL markers assigned, call analyze_panel
4. If complexity_index > 2.0 or critical_pairs > 0, revise choices

## Output as markdown table with ALL {n_markers} markers:
| Marker | Fluorophore | Max Similarity |
|--------|-------------|----------------|

End with the analyze_panel output showing final complexity index.
"""


def format_markers(markers: list[dict]) -> str:
    lines = []
    for m in markers:
        expr = m.get('expression', 'medium')
        notes = f" - {m.get('notes', '')}" if m.get('notes') else ""
        lines.append(f"- {m['name']}: {expr} expression{notes}")
    return "\n".join(lines)


def call_model(prompt: str, model: str, use_tools: bool) -> tuple[str, float, int]:
    """Call Claude with or without tools."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    tool_calls = 0

    start = time.time()

    if not use_tools:
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=messages
        )
        elapsed = time.time() - start
        return response.content[0].text, elapsed, 0

    # With tools
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
            print("  [Tool limit reached]")
            break

    elapsed = time.time() - start
    text_blocks = [b for b in response.content if b.type == "text"]
    final_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""

    return final_text, elapsed, tool_calls


def parse_panel(response: str) -> list[dict]:
    """Parse markdown table."""
    assignments = []
    in_table = False

    for line in response.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue

        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]

        if len(parts) >= 2:
            marker, fluor = parts[0], parts[1]

            # Skip headers
            if marker.lower() in ["marker", "---", "-----"] or "---" in marker or "---" in fluor:
                in_table = True
                continue

            marker = marker.replace("**", "").replace("*", "").strip()
            fluor = fluor.replace("**", "").replace("*", "").strip()

            # Skip if looks like header
            if fluor.lower() in ["fluorophore", "fluorochrome", "dye"]:
                continue

            if marker and fluor and len(marker) < 30 and len(fluor) < 30:
                assignments.append({"marker": marker, "fluorophore": fluor})

    return assignments


def run_single_test(condition: TestCondition, omip_num: int, run_num: int) -> TestResult:
    """Run a single test."""
    panel = get_omip_panel(omip_num)
    markers = get_markers_only(panel)

    if condition.use_tools:
        prompt = TREATMENT_PROMPT.format(
            instrument=panel.instrument,
            marker_list=format_markers(markers),
            n_markers=len(markers),
        )
    else:
        prompt = CONTROL_PROMPT.format(
            instrument=panel.instrument,
            marker_list=format_markers(markers),
            n_markers=len(markers),
        )

    response, elapsed, tool_calls = call_model(prompt, condition.model, condition.use_tools)
    assignments = parse_panel(response)

    # Evaluate
    if assignments:
        fluorophores = [a["fluorophore"] for a in assignments]
        metrics = analyze_panel(fluorophores)
        ci = metrics.get("complexity_index")
        max_sim = metrics.get("max_similarity")
        critical = len(metrics.get("critical_pairs", []))
        high_risk = len(metrics.get("problematic_pairs", []))
    else:
        ci, max_sim, critical, high_risk = None, None, 0, 0

    return TestResult(
        condition=condition.name,
        model=condition.model,
        use_tools=condition.use_tools,
        omip=omip_num,
        run=run_num,
        complexity_index=ci,
        max_similarity=max_sim,
        critical_pairs=critical,
        high_risk_pairs=high_risk,
        n_assignments=len(assignments),
        elapsed=elapsed,
        tool_calls=tool_calls,
    )


def run_comprehensive_test(n_runs: int = 3) -> list[TestResult]:
    """Run all tests."""
    results = []
    total_tests = len(CONDITIONS) * len(OMIP_PANELS) * n_runs
    completed = 0

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MCP EVALUATION")
    print(f"{'='*80}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"OMIP panels: {OMIP_PANELS}")
    print(f"Runs per condition: {n_runs}")
    print(f"Total tests: {total_tests}")
    print(f"{'='*80}\n")

    for omip_num in OMIP_PANELS:
        panel = get_omip_panel(omip_num)
        print(f"\n{'#'*60}")
        print(f"# OMIP-{omip_num:03d}: {len(get_markers_only(panel))} markers")
        print(f"{'#'*60}")

        for run in range(1, n_runs + 1):
            print(f"\n--- Run {run}/{n_runs} ---")

            for condition in CONDITIONS:
                completed += 1
                print(f"  [{completed}/{total_tests}] {condition.name}...", end=" ", flush=True)

                try:
                    result = run_single_test(condition, omip_num, run)
                    results.append(result)

                    ci_str = f"CI={result.complexity_index:.2f}" if result.complexity_index else "CI=N/A"
                    print(f"{ci_str}, {result.n_assignments} assignments, {result.elapsed:.1f}s")
                except Exception as e:
                    print(f"ERROR: {e}")

    return results


def print_summary(results: list[TestResult]):
    """Print comprehensive summary."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)

    # Group by condition
    conditions = {}
    for r in results:
        if r.condition not in conditions:
            conditions[r.condition] = []
        conditions[r.condition].append(r)

    # Summary table
    print(f"\n{'Condition':<15} {'Avg CI':>10} {'Std CI':>10} {'Avg MaxSim':>12} {'Avg Crit':>10} {'Avg Time':>10} {'Valid':>8}")
    print("-" * 100)

    condition_stats = {}
    for cond_name, cond_results in conditions.items():
        valid = [r for r in cond_results if r.complexity_index is not None]

        if valid:
            cis = [r.complexity_index for r in valid]
            sims = [r.max_similarity for r in valid if r.max_similarity]
            crits = [r.critical_pairs for r in valid]
            times = [r.elapsed for r in valid]

            avg_ci = statistics.mean(cis)
            std_ci = statistics.stdev(cis) if len(cis) > 1 else 0
            avg_sim = statistics.mean(sims) if sims else 0
            avg_crit = statistics.mean(crits)
            avg_time = statistics.mean(times)

            condition_stats[cond_name] = {
                "avg_ci": avg_ci,
                "std_ci": std_ci,
                "avg_sim": avg_sim,
                "avg_crit": avg_crit,
                "n_valid": len(valid),
                "n_total": len(cond_results),
            }

            print(f"{cond_name:<15} {avg_ci:>10.2f} {std_ci:>10.2f} {avg_sim:>12.3f} {avg_crit:>10.1f} {avg_time:>9.1f}s {len(valid):>4}/{len(cond_results)}")
        else:
            print(f"{cond_name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>10} {0:>4}/{len(cond_results)}")

    print("-" * 100)

    # Key comparisons
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    def get_avg_ci(name):
        return condition_stats.get(name, {}).get("avg_ci")

    sonnet = get_avg_ci("Sonnet")
    sonnet_mcp = get_avg_ci("Sonnet+MCP")
    opus = get_avg_ci("Opus")
    opus_mcp = get_avg_ci("Opus+MCP")

    findings = []

    if sonnet and sonnet_mcp:
        improvement = (sonnet - sonnet_mcp) / sonnet * 100 if sonnet > 0 else 0
        findings.append(f"1. MCP improves Sonnet by {improvement:.0f}% (CI: {sonnet:.2f} → {sonnet_mcp:.2f})")

    if opus and opus_mcp:
        improvement = (opus - opus_mcp) / opus * 100 if opus > 0 else 0
        findings.append(f"2. MCP improves Opus by {improvement:.0f}% (CI: {opus:.2f} → {opus_mcp:.2f})")

    if sonnet_mcp and opus:
        if sonnet_mcp < opus:
            findings.append(f"3. Sonnet+MCP BEATS Opus alone ({sonnet_mcp:.2f} vs {opus:.2f}) → Tools > Reasoning")
        else:
            findings.append(f"3. Opus alone beats Sonnet+MCP ({opus:.2f} vs {sonnet_mcp:.2f})")

    if sonnet_mcp and opus_mcp:
        diff = abs(sonnet_mcp - opus_mcp)
        if diff < 0.5:
            findings.append(f"4. Both models converge with MCP (Sonnet+MCP={sonnet_mcp:.2f}, Opus+MCP={opus_mcp:.2f})")
        else:
            better = "Opus+MCP" if opus_mcp < sonnet_mcp else "Sonnet+MCP"
            findings.append(f"4. {better} is best overall")

    for f in findings:
        print(f"\n{f}")

    # Per-OMIP breakdown
    print("\n" + "=" * 100)
    print("PER-OMIP BREAKDOWN")
    print("=" * 100)

    for omip_num in OMIP_PANELS:
        omip_results = [r for r in results if r.omip == omip_num]
        panel = get_omip_panel(omip_num)
        n_markers = len(get_markers_only(panel))

        print(f"\nOMIP-{omip_num:03d} ({n_markers} markers):")
        print(f"  {'Condition':<15} {'Avg CI':>10} {'Runs':>8}")
        print(f"  {'-'*35}")

        for cond_name in ["Sonnet", "Sonnet+MCP", "Opus", "Opus+MCP"]:
            cond_omip = [r for r in omip_results if r.condition == cond_name]
            valid = [r for r in cond_omip if r.complexity_index is not None]
            if valid:
                avg = statistics.mean([r.complexity_index for r in valid])
                print(f"  {cond_name:<15} {avg:>10.2f} {len(valid):>8}")
            else:
                print(f"  {cond_name:<15} {'N/A':>10} {0:>8}")

    return condition_stats


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Run comprehensive test
    results = run_comprehensive_test(n_runs=3)

    # Print summary
    stats = print_summary(results)

    # Save raw results
    output_file = Path(__file__).parent / "comprehensive_results.json"
    with open(output_file, "w") as f:
        json.dump([{
            "condition": r.condition,
            "model": r.model,
            "use_tools": r.use_tools,
            "omip": r.omip,
            "run": r.run,
            "complexity_index": r.complexity_index,
            "max_similarity": r.max_similarity,
            "critical_pairs": r.critical_pairs,
            "high_risk_pairs": r.high_risk_pairs,
            "n_assignments": r.n_assignments,
            "elapsed": r.elapsed,
            "tool_calls": r.tool_calls,
        } for r in results], f, indent=2)

    print(f"\n\nRaw results saved to: {output_file}")


if __name__ == "__main__":
    main()
