#!/usr/bin/env python3
"""
Run panel design ablation study using local models via LiteLLM/Ollama.

This script provides placeholders for running experiments with:
- Local Llama models (via Ollama)
- Local DeepSeek models (via Ollama)
- Other LiteLLM-supported backends

Prerequisites:
1. pip install litellm
2. Install Ollama: curl https://ollama.ai/install.sh | sh
3. Pull desired model: ollama pull llama3.1:70b
4. Start Ollama: ollama serve

Usage:
    python scripts/run_local_experiment.py --model ollama/llama3.1:70b
    python scripts/run_local_experiment.py --model ollama/deepseek-r1:32b
    python scripts/run_local_experiment.py --list-models
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def list_models():
    """List all supported models."""
    from flow_panel_optimizer.evaluation.litellm_runner import (
        LiteLLMAblationRunner,
        get_model_config,
    )

    print("Supported Models for Local Experiments")
    print("=" * 60)

    for model in LiteLLMAblationRunner.list_supported_models():
        config = get_model_config(model)
        tools_status = "tools" if config.supports_tools else "no-tools"
        provider = config.provider.upper()
        print(f"  [{provider}] {model} ({tools_status})")

    print()
    print("Local Model Setup (Ollama):")
    print("-" * 60)
    print("1. Install: curl https://ollama.ai/install.sh | sh")
    print("2. Pull model: ollama pull <model-name>")
    print("3. Run: ollama serve")
    print()
    print("Recommended local models for scientific reasoning:")
    print("  - ollama/llama3.1:70b    (best accuracy, requires ~40GB VRAM)")
    print("  - ollama/llama3.1:8b     (faster, requires ~8GB VRAM)")
    print("  - ollama/deepseek-r1:32b (reasoning focus, ~20GB VRAM)")
    print("  - ollama/deepseek-r1:7b  (smaller reasoning, ~8GB VRAM)")


def check_prerequisites(model: str):
    """Check if prerequisites are met for running the model."""
    try:
        import litellm
        print("[OK] litellm is installed")
    except ImportError:
        print("[MISSING] litellm not installed")
        print("  Run: pip install litellm")
        return False

    if model.startswith("ollama/"):
        from flow_panel_optimizer.evaluation.litellm_runner import (
            LiteLLMAblationRunner,
        )
        model_name = model.replace("ollama/", "")
        if LiteLLMAblationRunner.check_ollama_available(model_name):
            print(f"[OK] Ollama model {model_name} is available")
        else:
            print(f"[MISSING] Ollama model {model_name} not available")
            print(f"  Run: ollama pull {model_name}")
            print("  Then: ollama serve")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run panel design ablation with local models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/llama3.1:8b",
        help="Model to use (e.g., ollama/llama3.1:70b)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all supported models"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check prerequisites without running"
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=4,
        help="Number of test cases per type (default: 4)"
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    print("=" * 60)
    print("FLOW PANEL OPTIMIZER - LOCAL MODEL EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print()

    # Check prerequisites
    print("Checking prerequisites...")
    if not check_prerequisites(args.model):
        print()
        print("Prerequisites not met. Please install missing components.")
        sys.exit(1)

    if args.check:
        print()
        print("All prerequisites met!")
        return

    # Import after prerequisite check
    from flow_panel_optimizer.evaluation.litellm_runner import LiteLLMAblationRunner
    from flow_panel_optimizer.evaluation.test_cases import build_ablation_test_suite
    from flow_panel_optimizer.evaluation.conditions import CORE_CONDITIONS

    # Build test suite
    print()
    print("Building test suite...")
    suite = build_ablation_test_suite(
        n_in_dist=args.n_cases,
        n_near_dist=max(1, args.n_cases // 2),
        n_out_dist=max(1, args.n_cases // 2),
        n_adversarial=1,
    )

    print(f"Test suite: {suite.name}")
    print(f"  Total cases: {len(suite.test_cases)}")
    print()

    # Initialize runner
    runner = LiteLLMAblationRunner(model=args.model)

    # Get applicable conditions (non-MCP for models without tool support)
    conditions = CORE_CONDITIONS
    if not runner.config.supports_tools:
        conditions = [c for c in conditions if not c.mcp_enabled]
        print(f"Note: Model does not support tools, running {len(conditions)} conditions")

    print()
    print(f"Running {len(suite.test_cases)} test cases × {len(conditions)} conditions...")
    print("-" * 60)

    # Run study
    results = runner.run_full_study(
        test_suite=suite,
        conditions=conditions,
        verbose=True
    )

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_safe = args.model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"local_{model_safe}_{timestamp}.json"
    results.to_json(results_file)

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Results saved to: {results_file}")

    # Print summary
    df = results.to_dataframe()
    print()
    print("--- By Condition ---")
    print(df.groupby("condition")["accuracy"].agg(["mean", "std", "count"]).round(3))

    print()
    print("--- By Case Type ---")
    print(df.groupby(["condition", "case_type"])["accuracy"].mean().round(3))


if __name__ == "__main__":
    main()
