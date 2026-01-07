#!/usr/bin/env python3
"""
Run the gating benchmark on test cases.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python run_benchmark.py

    # Or with .env file:
    python run_benchmark.py

    # With detailed per-test reports:
    python run_benchmark.py --verbose

    # Test specific condition:
    python run_benchmark.py --strategy cot --context standard

    # Run with MCP/skills ablation:
    python run_benchmark.py --ablation mcp
    python run_benchmark.py --ablation skills
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env if it exists
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.curation.omip_extractor import load_all_test_cases
from src.experiments.prompts import build_prompt
from src.evaluation.scorer import GatingScorer, compute_aggregate_metrics
from src.analysis.detailed_report import LiveReportWriter, generate_strategy_comparison_report

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)


def run_single_test(
    client,
    test_case,
    model="claude-sonnet-4-20250514",
    context_level="standard",
    strategy="cot",
    system_prompt=None,
):
    """Run a single test case and return the result."""

    # Build prompt
    prompt = build_prompt(test_case, template_name=strategy, context_level=context_level)

    # Call API
    print(f"  Calling {model}...", end=" ", flush=True)
    start = datetime.now()

    messages = [{"role": "user", "content": prompt}]

    # Optional system prompt for MCP/skills ablation
    kwargs = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": messages,
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    message = client.messages.create(**kwargs)

    elapsed = (datetime.now() - start).total_seconds()
    response = message.content[0].text

    # Get token usage
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    print(f"done ({elapsed:.1f}s, {input_tokens}+{output_tokens} tokens)")

    # Score
    scorer = GatingScorer()
    result = scorer.score(
        response=response,
        test_case=test_case,
        model=model,
        condition=f"{context_level}_{strategy}",
    )

    return result, input_tokens, output_tokens, response


def get_mcp_system_prompt():
    """
    System prompt that simulates MCP with flowkit integration.

    In actual MCP implementation, this would be replaced with
    tool definitions for flowkit operations.
    """
    return """You are a flow cytometry analysis expert with access to FlowKit tools.

## Available MCP Tools (flowkit integration):

### read_workspace(path: str) -> WorkspaceInfo
Reads a FlowJo workspace (.wsp) file and returns gating hierarchy information.

### get_gating_hierarchy(workspace: str) -> GatingHierarchy
Extracts the complete gating hierarchy from a workspace file as GatingML.

### list_populations(workspace: str) -> list[Population]
Lists all defined populations/gates in a workspace.

### get_gate_statistics(workspace: str, gate: str) -> GateStats
Returns statistics for a specific gate (count, percentage, parent).

## Instructions:
When analyzing flow cytometry panels, you should:
1. Consider standard QC gating workflow (Time -> Cells -> Singlets -> Live)
2. Use your knowledge of immunology to predict appropriate lineage gates
3. Structure output as hierarchical JSON with gate names, markers, and children

Always include these critical QC gates:
- Time gate (fluorescence stability)
- FSC/SSC cell gate (debris exclusion)
- Singlet gate (doublet exclusion via FSC-A vs FSC-H)
- Live/Dead gate (viability)
- CD45+ gate (for immune cell panels)
"""


def get_skills_system_prompt():
    """
    System prompt that provides flow cytometry domain skills.

    This simulates having a flow cytometry skill/knowledge base.
    """
    return """You are a flow cytometry analysis expert with specialized immunology skills.

## Flow Cytometry Gating Skills:

### Standard QC Workflow
Every gating strategy MUST start with these QC gates in order:
1. **Time Gate**: Plot Time vs any parameter, exclude unstable regions
2. **Cell Gate**: FSC-A vs SSC-A, exclude debris (low FSC/SSC corner)
3. **Singlet Gate**: FSC-A vs FSC-H, select diagonal population (removes doublets)
4. **Live Gate**: Viability dye negative (7-AAD-, Zombie-, DAPI-, etc.)

### Lineage Identification
After QC, identify major populations:
- **CD45+**: All leukocytes (immune cells)
- **CD45-**: Non-immune (epithelial/tumor via EpCAM+, endothelial via CD31+)
- **CD235a-**: Exclude RBCs if not lysed

### T Cell Hierarchy
CD3+ (T Cells)
├── CD4+ Helper T Cells
│   ├── Naive (CD45RA+CCR7+)
│   ├── Central Memory (CD45RA-CCR7+)
│   ├── Effector Memory (CD45RA-CCR7-)
│   └── TEMRA (CD45RA+CCR7-)
├── CD8+ Cytotoxic T Cells
│   └── [Same memory subsets]
└── Tregs (CD4+CD25+CD127low)

### B Cell Hierarchy
CD19+ or CD20+ (B Cells)
├── Naive (IgD+CD27-)
├── Memory (IgD-CD27+)
└── Plasmablasts (CD38++CD27++)

### NK Cell Hierarchy
CD3-CD56+ (NK Cells)
├── CD56bright (cytokine-producing)
└── CD56dim CD16+ (cytotoxic)

### Myeloid Hierarchy
CD14+ or CD11b+ (Myeloid)
├── Classical Monocytes (CD14++CD16-)
├── Non-classical Monocytes (CD14+CD16++)
└── Dendritic Cells (HLA-DR+CD11c+)

## Critical Rules:
1. ALWAYS include Time, Singlets, and Live/Dead gates
2. Gate names should match standard nomenclature
3. Structure must be hierarchical (children nested under parents)
4. Only use markers present in the provided panel
"""


def run_benchmark(args):
    """Main benchmark runner."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        print("Usage: ANTHROPIC_API_KEY=sk-ant-... python run_benchmark.py")
        sys.exit(1)

    # Initialize client
    client = Anthropic(api_key=api_key)

    # Load test cases
    ground_truth_dir = Path(__file__).parent / "data" / "ground_truth"
    test_cases = load_all_test_cases(ground_truth_dir)

    if not test_cases:
        print("No test cases found!")
        sys.exit(1)

    # Limit test cases if specified
    if args.limit:
        test_cases = test_cases[:args.limit]

    # Determine condition name
    condition = f"{args.context}_{args.strategy}"
    if args.ablation:
        condition = f"{condition}_{args.ablation}"

    # Get system prompt for ablation
    system_prompt = None
    if args.ablation == "mcp":
        system_prompt = get_mcp_system_prompt()
    elif args.ablation == "skills":
        system_prompt = get_skills_system_prompt()

    print(f"\n{'='*60}")
    print("FLOW GATING BENCHMARK")
    print(f"{'='*60}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Model: {args.model}")
    print(f"Condition: {condition}")
    if args.ablation:
        print(f"Ablation: {args.ablation}")
    print()

    # Initialize report writer
    reports_dir = Path(__file__).parent / "results" / "reports"
    report_writer = LiveReportWriter(
        output_dir=reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S"),
        console_output=True,
        verbose=args.verbose,
    )

    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, test_case in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {test_case.test_case_id} ({test_case.n_colors} colors)")

        try:
            result, in_tok, out_tok, response = run_single_test(
                client,
                test_case,
                model=args.model,
                context_level=args.context,
                strategy=args.strategy,
                system_prompt=system_prompt,
            )
            results.append(result)
            total_input_tokens += in_tok
            total_output_tokens += out_tok

            # Generate per-test report
            report_writer.record_result(result, test_case, response)

        except Exception as e:
            print(f"  → Error: {e}")
            import traceback
            if args.verbose:
                traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    metrics = compute_aggregate_metrics(results)

    print(f"\nTotal test cases: {metrics.get('total', 0)}")
    print(f"Successfully parsed: {metrics.get('valid', 0)}")
    print(f"Parse success rate: {metrics.get('parse_success_rate', 0):.1%}")

    print(f"\nMetrics (mean):")
    print(f"  Hierarchy F1:        {metrics.get('hierarchy_f1_mean', 0):.3f}")
    print(f"  Structure Accuracy:  {metrics.get('structure_accuracy_mean', 0):.3f}")
    print(f"  Critical Gate Recall:{metrics.get('critical_gate_recall_mean', 0):.3f}")
    print(f"  Hallucination Rate:  {metrics.get('hallucination_rate_mean', 0):.3f}")

    # Cost estimation
    # Claude Sonnet pricing: $3/M input, $15/M output
    input_cost = total_input_tokens * 3 / 1_000_000
    output_cost = total_output_tokens * 15 / 1_000_000
    total_cost = input_cost + output_cost

    print(f"\n{'='*60}")
    print("TOKEN USAGE & COST")
    print(f"{'='*60}")
    print(f"Input tokens:  {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Total tokens:  {total_input_tokens + total_output_tokens:,}")
    print(f"\nEstimated cost: ${total_cost:.4f}")
    print(f"  Input:  ${input_cost:.4f}")
    print(f"  Output: ${output_cost:.4f}")

    # Finalize reports
    print(f"\n{'='*60}")
    print("GENERATING REPORTS")
    print(f"{'='*60}")

    final_reports = report_writer.finalize()
    for report_type, content in final_reports.items():
        print(f"  Generated: {report_type}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "n_test_cases": len(test_cases),
            "model": args.model,
            "condition": condition,
            "ablation": args.ablation,
            "metrics": metrics,
            "token_usage": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens,
            },
            "cost_usd": total_cost,
            "results": [r.to_dict() for r in results],
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Print location of detailed reports
    if report_writer.output_dir:
        print(f"Detailed reports: {report_writer.output_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run flow gating benchmark")

    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--strategy",
        choices=["cot", "direct", "few_shot"],
        default="cot",
        help="Prompting strategy (default: cot)",
    )
    parser.add_argument(
        "--context",
        choices=["minimal", "standard", "rich"],
        default="standard",
        help="Context level (default: standard)",
    )
    parser.add_argument(
        "--ablation",
        choices=["mcp", "skills", None],
        default=None,
        help="Ablation study: mcp (flowkit integration) or skills (domain knowledge)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases (for quick testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-test reports",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
