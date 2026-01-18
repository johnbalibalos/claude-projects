#!/usr/bin/env python3
"""
Convenience wrapper to run the full gating benchmark pipeline.

This is a thin wrapper around the standalone scripts:
    predict.py → score.py → judge.py

For most use cases, prefer running the individual scripts directly:
    python scripts/predict.py --output results/predictions.json --force
    python scripts/score.py --input results/predictions.json
    python scripts/judge.py --input results/scores.json --force

This wrapper is useful when you want to:
    - Run all phases with a single command
    - Use shared output directory naming conventions
    - Track provenance across all phases

Usage:
    python scripts/run_modular_pipeline.py --phase all --models gemini-2.0-flash --force
    python scripts/run_modular_pipeline.py --phase predict --max-cases 1 --force
    python scripts/run_modular_pipeline.py --phase score
    python scripts/run_modular_pipeline.py --phase judge
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pipeline_utils import PROJECT_ROOT

# Import for provenance tracking only
from utils.provenance import ExperimentContext
from experiments.llm_judge import JUDGE_STYLES


def run_script(script_name: str, args: list[str]) -> int:
    """Run a pipeline script as subprocess."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run gating benchmark pipeline (wrapper for predict.py → score.py → judge.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["predict", "score", "judge", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "verified",
        help="Directory with test cases (default: data/verified)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "modular_pipeline",
        help="Output directory (default: results/modular_pipeline)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-sonnet-cli"],
        help="Models to test (default: claude-sonnet-cli)",
    )
    parser.add_argument(
        "--context-levels",
        nargs="+",
        default=["minimal", "standard"],
        help="Context levels (default: minimal standard)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1,
        help="Bootstrap runs per condition (default: 1)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit test cases (for testing)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=6000,
        help="Max output tokens (default: 6000)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-pro",
        help="Judge model (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--judge-style",
        type=str,
        choices=JUDGE_STYLES,
        default="default",
        help=f"Judge style (default: default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mock API calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cost confirmation",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Define file paths
    predictions_file = args.output / "predictions.json"
    scores_file = args.output / "scores.json"
    judge_file = args.output / "judge.json"

    # Create provenance context
    experiment_config = {
        "phase": args.phase,
        "dry_run": args.dry_run,
        "test_cases_dir": str(args.test_cases),
        "judge_model": args.judge_model,
        "judge_style": args.judge_style,
    }
    ctx = ExperimentContext.create(
        ground_truth_dir=args.test_cases,
        config=experiment_config,
        models=args.models,
        n_bootstrap=args.n_bootstrap,
    )
    ctx.save(args.output)
    ctx.print_summary()

    start_time = datetime.now()

    try:
        # Phase: predict
        if args.phase in ["predict", "all"]:
            predict_args = [
                "--test-cases", str(args.test_cases),
                "--output", str(predictions_file),
                "--checkpoint-dir", str(args.output / "checkpoints"),
                "--models", *args.models,
                "--context-levels", *args.context_levels,
                "--n-bootstrap", str(args.n_bootstrap),
                "--max-tokens", str(args.max_tokens),
                "--run-id", ctx.run_id,
            ]
            if args.max_cases:
                predict_args += ["--max-cases", str(args.max_cases)]
            if args.dry_run:
                predict_args.append("--dry-run")
            if args.resume:
                predict_args.append("--resume")
            if args.force:
                predict_args.append("--force")

            ret = run_script("predict.py", predict_args)
            if ret != 0:
                raise RuntimeError(f"predict.py failed with code {ret}")

        # Phase: score
        if args.phase in ["score", "all"]:
            score_args = [
                "--input", str(predictions_file),
                "--output", str(scores_file),
                "--test-cases", str(args.test_cases),
                "--checkpoint-dir", str(args.output / "checkpoints"),
            ]

            ret = run_script("score.py", score_args)
            if ret != 0:
                raise RuntimeError(f"score.py failed with code {ret}")

        # Phase: judge
        if args.phase in ["judge", "all"]:
            judge_args = [
                "--input", str(scores_file),
                "--output", str(judge_file),
                "--test-cases", str(args.test_cases),
                "--checkpoint-dir", str(args.output / "checkpoints"),
                "--judge-model", args.judge_model,
                "--judge-style", args.judge_style,
            ]
            if args.dry_run:
                judge_args.append("--dry-run")
            if args.force:
                judge_args.append("--force")

            ret = run_script("judge.py", judge_args)
            if ret != 0:
                raise RuntimeError(f"judge.py failed with code {ret}")

        # Mark complete
        elapsed = datetime.now() - start_time
        ctx.mark_completed(0, 0, 0, args.output)
        print(f"\n{'='*60}")
        print(f"Pipeline complete in {elapsed.total_seconds():.1f}s")
        print(f"{'='*60}")
        print(f"\nOutputs:")
        if predictions_file.exists():
            print(f"  Predictions: {predictions_file}")
        if scores_file.exists():
            print(f"  Scores:      {scores_file}")
        if judge_file.exists():
            print(f"  Judge:       {judge_file}")

    except Exception as e:
        ctx.mark_failed(args.output, str(e))
        print(f"\nPipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
