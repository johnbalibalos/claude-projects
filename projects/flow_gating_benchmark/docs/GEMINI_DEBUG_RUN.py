#!/usr/bin/env python3
"""
Gemini-Only Debug Run

Tests the benchmark workflow using only Gemini models (free with credits).
Use this to debug and validate the pipeline before running the full experiment.

Configuration:
- 3 test models: Gemini 2.0 Flash, 2.5 Flash, 2.5 Pro
- 8 test cases (same as full run)
- 3 bootstrap runs (reduced for faster debugging)
- Judge: Gemini 2.5 Pro only

Estimated cost: ~$15-20 (all Gemini credits, $0 Anthropic)

Usage:
    # Dry run
    python docs/GEMINI_DEBUG_RUN.py --dry-run

    # Quick debug (1 bootstrap, 2 test cases)
    python docs/GEMINI_DEBUG_RUN.py --quick

    # Full debug run
    python docs/GEMINI_DEBUG_RUN.py

    # Single test case debug
    python docs/GEMINI_DEBUG_RUN.py --test-cases OMIP-077
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env
env_file = PROJECT_ROOT.parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value)


# =============================================================================
# CONFIGURATION (Gemini-only debug)
# =============================================================================

@dataclass
class DebugConfig:
    """Debug configuration using only Gemini models."""

    # Test cases (same 8 as full run)
    test_cases: list[str] = field(default_factory=lambda: [
        # Simple (7-12 colors)
        "OMIP-001",  # 7 colors, Basic T cell
        "OMIP-006",  # 10 colors, B cell memory
        # Medium (12-15 colors)
        "OMIP-003",  # 12 colors, B cell memory subsets
        "OMIP-077",  # 14 colors, All major leukocytes
        "OMIP-022",  # 15 colors, T cell memory/function
        # Complex (19-27 colors)
        "OMIP-074",  # 19 colors, B cell IgG/IgA subclasses
        "OMIP-064",  # 27 colors, T/NK/ILC/MAIT/γδT
        "OMIP-101",  # 27 colors, Fixed whole blood
    ])

    # Gemini models only (no Anthropic cost)
    models: list[str] = field(default_factory=lambda: [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ])

    # Reasoning strategies (wot not implemented in prompts.py)
    reasoning_types: list[str] = field(default_factory=lambda: [
        "direct",
        "cot",
    ])

    # Context levels
    context_levels: list[str] = field(default_factory=lambda: [
        "minimal",
        "standard",
        "rich",
    ])

    # RAG modes
    rag_modes: list[str] = field(default_factory=lambda: [
        "none",
        "oracle",
    ])

    # Bootstrap runs (reduced for debug)
    n_bootstrap: int = 3

    # Parallelization
    n_workers: int = 8  # Parallel API calls

    # CLI mode for Anthropic models (uses Max subscription instead of API)
    use_cli_for_anthropic: bool = False
    cli_delay_seconds: float = 2.0  # Delay between CLI calls to avoid rate limits
    cli_model_map: dict = field(default_factory=lambda: {
        "claude-sonnet": "sonnet",
        "claude-opus": "opus",
    })

    # Model parameters
    max_tokens: int = 4096
    temperature: float = 0.0

    # Judge configuration
    enable_judge: bool = True
    judge_model: str = "gemini-2.5-pro"
    judge_template: str = "comprehensive"
    use_claude_code_judge: bool = False  # Use Max subscription via CLI instead of API

    # Output
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results" / "gemini_debug")
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints" / "gemini_debug")

    @property
    def n_conditions(self) -> int:
        return (
            len(self.models) *
            len(self.reasoning_types) *
            len(self.context_levels) *
            len(self.rag_modes)
        )

    @property
    def total_api_calls(self) -> int:
        return self.n_conditions * len(self.test_cases) * self.n_bootstrap


# =============================================================================
# COST ESTIMATION
# =============================================================================

MODEL_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    # Anthropic models (API pricing, $0 when using CLI)
    "claude-sonnet": {"input": 3.0, "output": 15.0},
    "claude-opus": {"input": 15.0, "output": 75.0},
}

# Models that can be run via CLI (Max subscription)
CLI_MODELS = {"claude-sonnet", "claude-opus"}

TOKEN_ESTIMATES = {
    "direct": {"input": 1500, "output": 400},
    "cot": {"input": 2000, "output": 800},
    "wot": {"input": 2500, "output": 1200},
}

CONTEXT_MULTIPLIERS = {"minimal": 1.0, "standard": 1.3, "rich": 1.8}
RAG_TOKENS = {"none": 0, "oracle": 500}


def estimate_cost(config: DebugConfig) -> dict[str, Any]:
    """Estimate total cost (Gemini + Anthropic, $0 for CLI models)."""
    avg_input = sum(TOKEN_ESTIMATES[r]["input"] for r in config.reasoning_types) / len(config.reasoning_types)
    avg_output = sum(TOKEN_ESTIMATES[r]["output"] for r in config.reasoning_types) / len(config.reasoning_types)
    avg_context_mult = sum(CONTEXT_MULTIPLIERS[c] for c in config.context_levels) / len(config.context_levels)
    avg_rag = sum(RAG_TOKENS[r] for r in config.rag_modes) / len(config.rag_modes)

    avg_input = avg_input * avg_context_mult + avg_rag

    calls_per_model = config.total_api_calls // len(config.models) if config.models else 0

    cost_by_model = {}
    total_cost = 0.0
    cli_models_used = []

    for model in config.models:
        pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})

        # CLI models are $0 when using CLI mode
        if model in CLI_MODELS and config.use_cli_for_anthropic:
            cost_by_model[model] = 0.0
            cli_models_used.append(model)
        else:
            input_cost = (calls_per_model * avg_input / 1_000_000) * pricing["input"]
            output_cost = (calls_per_model * avg_output / 1_000_000) * pricing["output"]
            model_cost = input_cost + output_cost
            cost_by_model[model] = model_cost
            total_cost += model_cost

    # Judge cost
    judge_cost = 0.0
    judge_calls = 0
    if config.enable_judge:
        n_conditions_per_model = config.n_conditions // len(config.models)
        judge_calls = n_conditions_per_model * len(config.test_cases) * len(config.models)

        if config.use_claude_code_judge:
            # Using Max subscription via CLI - no API cost
            judge_cost = 0.0
        else:
            judge_pricing = MODEL_PRICING.get(config.judge_model, {"input": 0, "output": 0})
            judge_input = 3500
            judge_output = 500
            judge_cost = (
                (judge_calls * judge_input / 1_000_000) * judge_pricing["input"] +
                (judge_calls * judge_output / 1_000_000) * judge_pricing["output"]
            )

    return {
        "n_conditions": config.n_conditions,
        "n_test_cases": len(config.test_cases),
        "n_bootstrap": config.n_bootstrap,
        "total_api_calls": config.total_api_calls,
        "judge_calls": judge_calls,
        "avg_input_tokens": int(avg_input),
        "avg_output_tokens": int(avg_output),
        "cost_by_model": cost_by_model,
        "test_cost": total_cost,
        "judge_cost": judge_cost,
        "total_cost": total_cost + judge_cost,
        "cli_models": cli_models_used,
        "use_cli": config.use_cli_for_anthropic,
    }


def print_cost_estimate(config: DebugConfig):
    """Print formatted cost estimate."""
    est = estimate_cost(config)

    print("=" * 60)
    print("GEMINI DEBUG RUN - COST ESTIMATE")
    print("=" * 60)
    print(f"Test cases:        {est['n_test_cases']}")
    print(f"Conditions:        {est['n_conditions']} ({est['n_conditions'] // len(config.models)} per model)")
    print(f"Models:            {len(config.models)} (Gemini only)")
    print(f"Bootstrap runs:    {est['n_bootstrap']}")
    print(f"Total API calls:   {est['total_api_calls']:,}")
    print()

    print("TEST COSTS BY MODEL:")
    print("-" * 40)
    for model, cost in est["cost_by_model"].items():
        name = model.replace("gemini-", "Gemini ").replace("claude-", "Claude ")
        if model in est.get("cli_models", []):
            print(f"  {name:20} $   0.00  (via CLI)")
        else:
            print(f"  {name:20} ${cost:>7.2f}")
    print()

    if config.enable_judge:
        print(f"JUDGE ({config.judge_model}):")
        print("-" * 40)
        print(f"  Calls: {est['judge_calls']}")
        if config.use_claude_code_judge:
            print("  Cost:  $   0.00  (Max subscription via CLI)")
        else:
            print(f"  Cost:  ${est['judge_cost']:>7.2f}")
        print()

    print("=" * 40)
    print(f"  TOTAL COST:      ${est['total_cost']:>7.2f}")
    if est.get("cli_models") or config.use_claude_code_judge:
        cli_note = []
        if est.get("cli_models"):
            cli_note.append(f"tests: {', '.join(est['cli_models'])}")
        if config.use_claude_code_judge:
            cli_note.append("judge")
        print(f"  ANTHROPIC COST:  $   0.00  (via CLI: {', '.join(cli_note)})")
    else:
        print("  ANTHROPIC COST:  $   0.00  (Gemini only!)")
    print("=" * 40)

    # Show CLI mode warning
    if config.use_cli_for_anthropic:
        print()
        print("⚠️  CLI MODE ENABLED")
        print(f"  - Delay between calls: {config.cli_delay_seconds}s")
        print(f"  - Parallel workers: {config.n_workers} (reduced to avoid rate limits)")
        print("  - Using Max subscription instead of API key")


# =============================================================================
# GEMINI API CLIENT
# =============================================================================

def call_gemini(model: str, prompt: str, config: DebugConfig) -> tuple[str, int, int, float]:
    """Call Gemini API using new google-genai SDK."""
    from google import genai
    from google.genai import types

    # Create client with explicit API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return ("[ERROR] GOOGLE_API_KEY not set", 0, 0, 0.0)
    client = genai.Client(api_key=api_key)

    # Configure generation settings
    generation_config = types.GenerateContentConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        # Relax safety settings for biomedical content
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ],
    )

    start = datetime.now()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=generation_config,
        )
        latency = (datetime.now() - start).total_seconds()

        # Check for blocked response
        if not response.text:
            # Try to get block reason
            if response.candidates and response.candidates[0].finish_reason:
                reason = response.candidates[0].finish_reason
                return (f"[BLOCKED] finish_reason={reason}", 0, 0, latency)
            return ("[BLOCKED] No response text", 0, 0, latency)

        # Get token counts
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        return (response.text, input_tokens, output_tokens, latency)

    except Exception as e:
        latency = (datetime.now() - start).total_seconds()
        return (f"[ERROR] {e}", 0, 0, latency)


def call_anthropic_cli(model: str, prompt: str, config: DebugConfig) -> tuple[str, int, int, float]:
    """
    Call Anthropic model via Claude Code CLI (uses Max subscription, not API).

    Returns: (response_text, input_tokens, output_tokens, latency)
    Note: Token counts are estimated since CLI doesn't return them.
    """
    import subprocess
    import time

    # Map model names to CLI model names
    cli_model = config.cli_model_map.get(model, "sonnet")

    start = datetime.now()
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", cli_model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes for complex gating responses
        )
        latency = (datetime.now() - start).total_seconds()

        if result.returncode != 0:
            return (f"[CLI ERROR] {result.stderr}", 0, 0, latency)

        response = result.stdout

        # Estimate tokens (CLI doesn't return actual counts)
        # Rough estimate: 1 token ≈ 4 chars
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        # Add delay to avoid rate limits
        if config.cli_delay_seconds > 0:
            time.sleep(config.cli_delay_seconds)

        return (response, input_tokens, output_tokens, latency)

    except subprocess.TimeoutExpired:
        latency = (datetime.now() - start).total_seconds()
        return ("[CLI ERROR] Timeout after 180s", 0, 0, latency)
    except FileNotFoundError:
        latency = (datetime.now() - start).total_seconds()
        return ("[CLI ERROR] Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code", 0, 0, latency)
    except Exception as e:
        latency = (datetime.now() - start).total_seconds()
        return (f"[CLI ERROR] {e}", 0, 0, latency)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

@dataclass
class TrialResult:
    """Result from a single trial."""
    test_case_id: str
    model: str
    condition_name: str
    bootstrap_run: int
    raw_response: str
    parsed_hierarchy: dict | None
    parse_success: bool
    hierarchy_f1: float
    structure_accuracy: float
    critical_gate_recall: float
    hallucination_rate: float
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    matching_gates: list[str] = field(default_factory=list)
    missing_gates: list[str] = field(default_factory=list)
    extra_gates: list[str] = field(default_factory=list)


@dataclass
class ConditionResults:
    """Aggregated results for a condition."""
    test_case_id: str
    model: str
    condition_name: str
    trials: list[TrialResult] = field(default_factory=list)
    mean_f1: float = 0.0
    std_f1: float = 0.0
    mean_structure: float = 0.0
    mean_critical: float = 0.0
    judge_result: dict | None = None


def generate_conditions(config: DebugConfig) -> list[dict]:
    """Generate all condition combinations."""
    conditions = []
    for model, reasoning, context, rag in product(
        config.models,
        config.reasoning_types,
        config.context_levels,
        config.rag_modes,
    ):
        model_short = model.replace("gemini-", "").replace(".", "")
        condition_name = f"{model_short}_{reasoning}_{context}_{rag}"
        conditions.append({
            "model": model,
            "reasoning": reasoning,
            "context": context,
            "rag": rag,
            "condition_name": condition_name,
        })
    return conditions


def run_trial(
    test_case: dict,
    condition: dict,
    bootstrap_run: int,
    config: DebugConfig,
) -> TrialResult:
    """Run a single trial."""
    from evaluation.scorer import GatingScorer
    from experiments.prompts import build_prompt

    prompt = build_prompt(
        test_case,
        template_name=condition["reasoning"],
        context_level=condition["context"],
        rag_mode=condition["rag"],
    )

    model = condition["model"]

    # Choose API backend based on model type
    if model in CLI_MODELS and config.use_cli_for_anthropic:
        # Use Claude Code CLI for Anthropic models (Max subscription)
        response, input_tokens, output_tokens, latency = call_anthropic_cli(
            model, prompt, config
        )
    elif model.startswith("gemini"):
        # Use Gemini API
        response, input_tokens, output_tokens, latency = call_gemini(
            model, prompt, config
        )
    else:
        # Fallback: try Gemini API (will fail for Anthropic without --use-cli)
        response, input_tokens, output_tokens, latency = call_gemini(
            model, prompt, config
        )

    scorer = GatingScorer()
    result = scorer.score(
        response=response,
        test_case=test_case,
        model=condition["model"],
        condition=condition["condition_name"],
    )

    return TrialResult(
        test_case_id=test_case.test_case_id,
        model=condition["model"],
        condition_name=condition["condition_name"],
        bootstrap_run=bootstrap_run,
        raw_response=response,
        parsed_hierarchy=result.parsed_hierarchy,
        parse_success=result.parse_success,
        hierarchy_f1=result.evaluation.hierarchy_f1 if result.evaluation else 0.0,
        structure_accuracy=result.evaluation.structure_accuracy if result.evaluation else 0.0,
        critical_gate_recall=result.evaluation.critical_gate_recall if result.evaluation else 0.0,
        hallucination_rate=result.evaluation.hallucination_rate if result.evaluation else 0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=latency,
        matching_gates=result.evaluation.matching_gates if result.evaluation else [],
        missing_gates=result.evaluation.missing_gates if result.evaluation else [],
        extra_gates=result.evaluation.extra_gates if result.evaluation else [],
    )


# =============================================================================
# JUDGE
# =============================================================================

JUDGE_TEMPLATE = """You are an expert flow cytometry immunologist evaluating LLM-predicted gating hierarchies.

## Task Context

**Panel**: {panel_description}
**Colors**: {n_colors}

**Ground Truth Hierarchy**:
```
{ground_truth_tree}
```

## Model Outputs ({model_name}, {n_runs} runs)

{all_outputs}

## Evaluation

Score each criterion from 0-3:
1. **Gate Completeness**: Are critical gates present?
2. **Hierarchy Structure**: Are parent-child relationships correct?
3. **Marker Logic**: Are marker combinations valid?
4. **Consistency**: How similar are the outputs across runs?

Output as JSON:
```json
{{
  "scores": {{
    "gate_completeness": <0-3>,
    "hierarchy_structure": <0-3>,
    "marker_logic": <0-3>,
    "consistency": <0-3>
  }},
  "weighted_total": <0-3>,
  "systematic_errors": ["errors in 2+ runs"],
  "rationale": "brief summary"
}}
```"""


def call_claude_code(prompt: str, model: str = "opus") -> str:
    """
    Call Claude via Claude Code CLI (uses Max subscription, not API).

    Args:
        prompt: The prompt to send
        model: Model to use (opus, sonnet)

    Returns:
        Response text from Claude
    """
    import subprocess

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return f"[CLI ERROR] {result.stderr}"
        return result.stdout
    except subprocess.TimeoutExpired:
        return "[CLI ERROR] Timeout after 120s"
    except FileNotFoundError:
        return "[CLI ERROR] Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
    except Exception as e:
        return f"[CLI ERROR] {e}"


def run_judge(
    condition_result: ConditionResults,
    test_case: dict,
    ground_truth_tree: str,
    config: DebugConfig,
) -> dict:
    """Run judge evaluation on combined outputs."""
    outputs = []
    for i, trial in enumerate(condition_result.trials, 1):
        if trial.parsed_hierarchy:
            h = json.dumps(trial.parsed_hierarchy, indent=2)[:1500]
        else:
            h = f"[PARSE FAILED]\n{trial.raw_response[:300]}"
        outputs.append(f"### Run {i}\n```\n{h}\n```")

    panel_entries = test_case.panel.entries if hasattr(test_case, 'panel') else []

    prompt = JUDGE_TEMPLATE.format(
        panel_description=", ".join(e.marker for e in panel_entries[:10]),
        n_colors=len(panel_entries),
        ground_truth_tree=ground_truth_tree,
        model_name=condition_result.model.replace("gemini-", "Gemini "),
        n_runs=len(condition_result.trials),
        all_outputs="\n\n".join(outputs),
    )

    # Choose judge backend
    if config.use_claude_code_judge:
        # Use Max subscription via Claude Code CLI (saves ~$22 for Opus judge)
        response = call_claude_code(prompt, model="opus")
    else:
        # Use Gemini API
        response, _, _, _ = call_gemini(config.judge_model, prompt, config)

    # Parse JSON
    import re
    json_match = re.search(r'```json\s*([\s\S]*?)```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    return {"raw_response": response, "parse_error": True}


# =============================================================================
# REPORTS
# =============================================================================

def generate_debug_report(results: list[ConditionResults], config: DebugConfig, output_dir: Path):
    """Generate debug summary report."""
    # Check for error modes
    pro_failures = sum(1 for cr in results if "2.5-pro" in cr.model and cr.mean_f1 == 0.0)

    lines = [
        "# Gemini Debug Run Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        f"- Models: {', '.join(config.models)}",
        f"- Test cases: {len(config.test_cases)}",
        f"- Bootstrap: {config.n_bootstrap}",
        f"- Judge: {config.judge_model}",
        "",
    ]

    # Add error notes if any
    if pro_failures > 0:
        lines.extend([
            "## ⚠️ Error Notes",
            "",
            "### Gemini 2.5 Pro Safety Filter Issue",
            f"- **Affected conditions**: {pro_failures}",
            "- **Error type**: `finish_reason=2` (SAFETY)",
            "- **Description**: Gemini 2.5 Pro triggered safety filters on biomedical flow cytometry content.",
            "- **Impact**: All 2.5-pro responses were blocked, resulting in F1=0.0",
            "- **Recommendation**: Use Flash models (2.0/2.5) for biomedical content or configure safety settings.",
            "",
        ])

    lines.extend([
        "## Results Summary",
        "",
        "| Model | Avg F1 | Avg Structure | Avg Critical | Notes |",
        "|-------|--------|---------------|--------------|-------|",
    ])

    for model in config.models:
        model_results = [r for r in results if r.model == model]
        if not model_results:
            continue
        avg_f1 = sum(r.mean_f1 for r in model_results) / len(model_results)
        avg_struct = sum(r.mean_structure for r in model_results) / len(model_results)
        avg_crit = sum(r.mean_critical for r in model_results) / len(model_results)
        name = model.replace("gemini-", "")
        note = "⚠️ Safety filtered" if "2.5-pro" in model and avg_f1 == 0.0 else ""
        lines.append(f"| {name} | {avg_f1:.3f} | {avg_struct:.3f} | {avg_crit:.3f} | {note} |")

    lines.extend([
        "",
        "## Detailed Results",
        "",
        "| Condition | Test Case | F1 | Parse OK | Judge Score |",
        "|-----------|-----------|-----|----------|-------------|",
    ])

    for cr in sorted(results, key=lambda r: (r.model, r.test_case_id)):
        judge_score = "N/A"
        if cr.judge_result and not cr.judge_result.get("parse_error"):
            judge_score = f"{cr.judge_result.get('weighted_total', 'N/A')}"
        parse_ok = "✓" if all(t.parse_success for t in cr.trials) else "✗"
        lines.append(f"| {cr.condition_name[:25]} | {cr.test_case_id} | {cr.mean_f1:.3f} | {parse_ok} | {judge_score} |")

    report_path = output_dir / "debug_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport saved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gemini-only debug run")
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs only")
    parser.add_argument("--quick", action="store_true", help="Quick debug (1 bootstrap, 2 cases)")
    parser.add_argument("--test-cases", nargs="+", help="Specific test cases")
    parser.add_argument("--no-judge", action="store_true", help="Skip judge")
    parser.add_argument("--bootstrap", type=int, help="Override bootstrap count")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--claude-judge", action="store_true",
                        help="Use Claude Code CLI for judge (Max subscription, saves ~$22)")
    parser.add_argument("--use-cli", action="store_true",
                        help="Use Claude Code CLI for Anthropic test models (Max subscription)")
    parser.add_argument("--cli-delay", type=float, default=2.0,
                        help="Delay between CLI calls in seconds (default: 2.0)")
    parser.add_argument("--add-anthropic", action="store_true",
                        help="Add Claude Sonnet/Opus to test models (requires --use-cli or API key)")
    args = parser.parse_args()

    config = DebugConfig()

    if args.quick:
        config.n_bootstrap = 1
        config.test_cases = ["OMIP-077", "OMIP-074"]  # 1 medium, 1 complex

    if args.test_cases:
        config.test_cases = args.test_cases

    if args.bootstrap:
        config.n_bootstrap = args.bootstrap

    if args.no_judge:
        config.enable_judge = False

    if args.claude_judge:
        config.use_claude_code_judge = True
        config.judge_model = "claude-opus (via CLI)"  # For display purposes

    if args.use_cli:
        config.use_cli_for_anthropic = True
        config.cli_delay_seconds = args.cli_delay
        # Reduce parallelization when using CLI to avoid rate limits
        config.n_workers = 1  # Sequential for CLI calls

    if args.add_anthropic:
        # Add Anthropic models
        config.models.extend(["claude-sonnet", "claude-opus"])

    print_cost_estimate(config)

    if args.dry_run:
        print("\nDry run complete. Run without --dry-run to execute.")
        return

    # Check API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("\nERROR: GOOGLE_API_KEY not set in environment")
        return

    if not args.yes:
        print()
        response = input("Proceed with debug run? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            return

    # Load test cases
    from curation.omip_extractor import load_all_test_cases
    test_cases_list = load_all_test_cases(PROJECT_ROOT / "data" / "ground_truth")
    test_cases = {tc.test_case_id: tc for tc in test_cases_list}

    # Check for missing
    available = {k.replace("-", "_").lower() for k in test_cases.keys()}
    missing = [tc for tc in config.test_cases if tc.replace("-", "_").lower() not in available]

    if missing:
        print(f"\nWARNING: Missing ground truth: {', '.join(missing)}")
        config.test_cases = [tc for tc in config.test_cases if tc not in missing]
        if not config.test_cases:
            print("No test cases available!")
            return

    config.output_dir.mkdir(parents=True, exist_ok=True)

    conditions = generate_conditions(config)
    print(f"\nRunning {len(conditions)} conditions × {len(config.test_cases)} cases × {config.n_bootstrap} bootstrap")

    all_results = []

    for test_id in config.test_cases:
        test_case = None
        for key, tc in test_cases.items():
            if test_id.replace("-", "_").lower() in key.replace("-", "_").lower():
                test_case = tc
                break

        if not test_case:
            continue

        print(f"\n{'='*50}")
        print(f"Test Case: {test_id}")
        print(f"{'='*50}")

        for condition in conditions:
            print(f"\n  {condition['condition_name']}")

            cr = ConditionResults(
                test_case_id=test_id,
                model=condition["model"],
                condition_name=condition["condition_name"],
            )

            # Run bootstrap trials in parallel
            def run_bootstrap(b, tc=test_case, cond=condition):
                return run_trial(tc, cond, b, config)

            print(f"    Running {config.n_bootstrap} bootstrap trials ({config.n_workers} parallel)...", end=" ", flush=True)
            with ThreadPoolExecutor(max_workers=config.n_workers) as executor:
                futures = {executor.submit(run_bootstrap, b): b for b in range(config.n_bootstrap)}
                for future in as_completed(futures):
                    try:
                        trial = future.result()
                        cr.trials.append(trial)
                    except Exception as e:
                        print(f"ERROR: {e}", end=" ")

            if cr.trials:
                f1s = [t.hierarchy_f1 for t in cr.trials]
                avg_f1 = sum(f1s) / len(f1s)
                avg_time = sum(t.latency_seconds for t in cr.trials) / len(cr.trials)
                print(f"avg F1={avg_f1:.3f} ({avg_time:.1f}s/call)")

            if cr.trials:
                f1s = [t.hierarchy_f1 for t in cr.trials]
                cr.mean_f1 = sum(f1s) / len(f1s)
                cr.std_f1 = (sum((x - cr.mean_f1)**2 for x in f1s) / len(f1s)) ** 0.5
                cr.mean_structure = sum(t.structure_accuracy for t in cr.trials) / len(cr.trials)
                cr.mean_critical = sum(t.critical_gate_recall for t in cr.trials) / len(cr.trials)

            all_results.append(cr)

    # Run judge
    if config.enable_judge:
        print(f"\n{'='*50}")
        print("Running Judge Evaluation")
        print(f"{'='*50}")

        from analysis.detailed_report import gate_to_tree_string

        for cr in all_results:
            if not cr.trials:
                continue

            test_case = None
            for key, tc in test_cases.items():
                if cr.test_case_id.replace("-", "_").lower() in key.replace("-", "_").lower():
                    test_case = tc
                    break

            if not test_case:
                continue

            print(f"  Judging {cr.condition_name[:30]}...", end=" ", flush=True)

            gt = test_case.gating_hierarchy
            gt_root = gt.root if hasattr(gt, 'root') else gt
            gt_tree = "\n".join(gate_to_tree_string(gt_root))

            try:
                cr.judge_result = run_judge(cr, test_case, gt_tree, config)
                print("done")
            except Exception as e:
                print(f"ERROR: {e}")

    # Save results
    results_path = config.output_dir / f"debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Track error modes
    error_notes = []
    pro_failures = sum(1 for cr in all_results if "2.5-pro" in cr.model and cr.mean_f1 == 0.0)
    if pro_failures > 0:
        error_notes.append({
            "model": "gemini-2.5-pro",
            "error_type": "safety_filter",
            "description": "Gemini 2.5 Pro triggered safety filters on biomedical gating content (finish_reason=2). Responses were blocked, resulting in F1=0.",
            "affected_conditions": pro_failures,
            "recommendation": "Use Flash models for biomedical content or adjust safety settings."
        })

    results_data = {
        "config": asdict(config),
        "error_notes": error_notes,
        "results": [
            {
                "test_case_id": cr.test_case_id,
                "model": cr.model,
                "condition_name": cr.condition_name,
                "mean_f1": cr.mean_f1,
                "std_f1": cr.std_f1,
                "mean_structure": cr.mean_structure,
                "mean_critical": cr.mean_critical,
                "judge_result": cr.judge_result,
                "n_trials": len(cr.trials),
                "n_failed": config.n_bootstrap - len(cr.trials) if hasattr(cr, 'trials') else 0,
            }
            for cr in all_results
        ],
    }

    # Handle Path serialization
    results_data["config"]["output_dir"] = str(config.output_dir)
    results_data["config"]["checkpoint_dir"] = str(config.checkpoint_dir)

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved: {results_path}")

    generate_debug_report(all_results, config, config.output_dir)

    print("\n" + "=" * 50)
    print("DEBUG RUN COMPLETE")
    print("=" * 50)
    print(f"Total conditions tested: {len(all_results)}")
    print("Anthropic cost: $0.00 (Gemini only)")


if __name__ == "__main__":
    main()
