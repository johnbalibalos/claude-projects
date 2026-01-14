#!/usr/bin/env python3
"""
Modular Gating Benchmark Pipeline

Demonstrates the decoupled architecture:
    PredictionCollector → BatchScorer → LLMJudge → Report

This enables:
- Running predictions and scoring separately
- Rerunning scoring on cached predictions
- Running LLM judge independently on scored results

Usage:
    python scripts/run_modular_pipeline.py --phase predict --dry-run
    python scripts/run_modular_pipeline.py --phase score
    python scripts/run_modular_pipeline.py --phase judge
    python scripts/run_modular_pipeline.py --phase all --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from curation.omip_extractor import load_all_test_cases  # noqa: E402
from experiments.batch_scorer import (  # noqa: E402
    BatchScorer,
    ScoringResult,
    compute_aggregate_stats,
)
from experiments.conditions import get_all_conditions  # noqa: E402
from experiments.llm_judge import (  # noqa: E402
    JUDGE_STYLES,
    JudgeConfig,
    JudgeResult,
    LLMJudge,
    compute_judge_stats,
)
from experiments.prediction_collector import (  # noqa: E402
    CollectorConfig,
    Prediction,
    PredictionCollector,
)
from utils.provenance import ExperimentContext  # noqa: E402


def print_phase(name: str):
    """Print a phase header."""
    print()
    print("=" * 60)
    print(f"PHASE: {name}")
    print("=" * 60)


def progress_callback(current: int, total: int, item):
    """Generic progress callback."""
    pct = (current / total) * 100 if total > 0 else 0
    if hasattr(item, 'error') and item.error:
        print(f"  [{current}/{total}] ({pct:.0f}%) ERROR: {item.error[:50]}")
    else:
        print(f"  [{current}/{total}] ({pct:.0f}%) {getattr(item, 'test_case_id', 'unknown')}")


def run_predict(
    test_cases_dir: Path,
    output_dir: Path,
    models: list[str],
    n_bootstrap: int,
    dry_run: bool,
    resume: bool,
    max_cases: int | None = None,
    run_id: str = "",
    references: list[str] | None = None,
) -> list[Prediction]:
    """Phase 1: Collect predictions from LLMs."""
    print_phase("PREDICTION COLLECTION")

    # Load test cases
    test_cases = load_all_test_cases(test_cases_dir)
    if max_cases is not None:
        test_cases = test_cases[:max_cases]
    print(f"Loaded {len(test_cases)} test cases")

    # Generate conditions
    conditions = get_all_conditions(
        models=models,
        context_levels=["minimal", "standard"],
        prompt_strategies=["direct", "cot"],
        references=references or ["none"],
    )
    print(f"Generated {len(conditions)} conditions")

    # Configure collector
    config = CollectorConfig(
        n_bootstrap=n_bootstrap,
        cli_delay_seconds=0.5,
        checkpoint_dir=output_dir / "checkpoints",
        dry_run=dry_run,
        run_id=run_id,  # Link predictions to experiment context
        # Per-provider parallelism (defaults: gemini=50, anthropic=50, openai=50)
    )

    collector = PredictionCollector(test_cases, conditions, config)
    print(f"Total calls to make: {collector.total_calls}")

    # Collect predictions
    predictions = collector.collect(resume=resume, progress_callback=progress_callback)

    print(f"\nCollected {len(predictions)} predictions")
    errors = [p for p in predictions if p.error]
    print(f"  Errors: {len(errors)}")

    # Save predictions
    output_file = output_dir / "predictions.json"
    with open(output_file, "w") as f:
        json.dump([p.to_dict() for p in predictions], f, indent=2)
    print(f"  Saved to: {output_file}")

    return predictions


def run_score(
    test_cases_dir: Path,
    output_dir: Path,
    predictions: list[Prediction] | None = None,
) -> list[ScoringResult]:
    """Phase 2: Score predictions against ground truth."""
    print_phase("BATCH SCORING")

    # Load test cases
    test_cases = load_all_test_cases(test_cases_dir)
    print(f"Loaded {len(test_cases)} test cases")

    # Load predictions if not provided
    if predictions is None:
        predictions_file = output_dir / "predictions.json"
        if not predictions_file.exists():
            print(f"ERROR: No predictions file found: {predictions_file}")
            return []

        with open(predictions_file) as f:
            data = json.load(f)
            predictions = [Prediction.from_dict(p) for p in data]
        print(f"Loaded {len(predictions)} predictions from file")

    # Score predictions
    scorer = BatchScorer(test_cases, checkpoint_dir=output_dir / "checkpoints")
    results = scorer.score_all(predictions)

    print(f"\nScored {len(results)} predictions")

    # Compute stats
    stats = compute_aggregate_stats(results)

    print("\nOverall Statistics:")
    overall = stats.get("overall", {})
    f1_stats = overall.get("hierarchy_f1", {})
    print(f"  Hierarchy F1: {f1_stats.get('mean', 0):.3f} ± {f1_stats.get('std', 0):.3f}")
    print(f"  Parse success rate: {overall.get('parse_success_rate', 0):.1%}")
    print(f"  Errors: {overall.get('error_count', 0)}")

    print("\nBy Model:")
    for model, model_stats in stats.get("by_model", {}).items():
        f1 = model_stats.get("hierarchy_f1", {})
        print(f"  {model}: F1={f1.get('mean', 0):.3f} (n={model_stats.get('n', 0)})")

    # Save results
    output_file = output_dir / "scoring_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "results": [r.to_dict() for r in results],
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved to: {output_file}")

    return results


def run_judge(
    test_cases_dir: Path,
    output_dir: Path,
    scoring_results: list[ScoringResult] | None = None,
    dry_run: bool = False,
    judge_model: str = "gemini-2.5-pro",
    judge_style: str = "default",
) -> list[JudgeResult]:
    """Phase 3: LLM judge evaluation."""
    print_phase("LLM JUDGE EVALUATION")

    # Load scoring results if not provided
    if scoring_results is None:
        results_file = output_dir / "scoring_results.json"
        if not results_file.exists():
            print(f"ERROR: No scoring results found: {results_file}")
            return []

        with open(results_file) as f:
            data = json.load(f)
            scoring_results = [ScoringResult.from_dict(r) for r in data["results"]]
        print(f"Loaded {len(scoring_results)} scoring results from file")

    # Configure judge
    # Use high parallelism for Gemini (50) - flash model can handle it
    config = JudgeConfig(
        model=judge_model,
        parallel_workers=50,
        checkpoint_dir=output_dir / "checkpoints",
        dry_run=dry_run,
        prompt_style=judge_style,
    )

    judge = LLMJudge(test_cases_dir, config)

    print(f"Judging {len(scoring_results)} results...")
    print(f"  Model: {config.model}")
    print(f"  Style: {config.prompt_style}")
    print(f"  Dry run: {dry_run}")

    results = judge.judge_all(scoring_results, progress_callback=progress_callback)

    print(f"\nJudged {len(results)} results")

    # Compute stats
    stats = compute_judge_stats(results)

    print("\nJudge Statistics:")
    overall = stats.get("overall", {})
    for key in ["completeness", "accuracy", "scientific", "overall"]:
        s = overall.get(key, {})
        print(f"  {key}: {s.get('mean', 0):.3f} ± {s.get('std', 0):.3f}")

    print("\nBy Model:")
    for model, model_stats in stats.get("by_model", {}).items():
        s = model_stats.get("overall", {})
        print(f"  {model}: {s.get('mean', 0):.3f} (n={model_stats.get('n', 0)})")

    if stats.get("common_issues"):
        print("\nCommon Issues:")
        for issue in stats["common_issues"][:5]:
            print(f"  - {issue[:60]}...")

    # Save results - include style in filename for multi-judge comparison
    if judge_style == "default":
        output_file = output_dir / "judge_results.json"
    else:
        output_file = output_dir / f"judge_results_{judge_style}.json"
    with open(output_file, "w") as f:
        json.dump({
            "results": [r.to_dict() for r in results],
            "stats": stats,
            "judge_style": judge_style,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run modular gating benchmark pipeline")
    parser.add_argument(
        "--phase",
        choices=["predict", "score", "judge", "all"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "verified",
        help="Directory with test cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "modular_pipeline",
        help="Output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-sonnet-cli"],
        help="Models to test",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1,
        help="Number of bootstrap runs",
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
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of test cases (for quick testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip cost confirmation hook",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-pro",
        help="Model to use for LLM judge (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--judge-style",
        type=str,
        choices=JUDGE_STYLES,
        default="default",
        help=f"Judge prompt style: {', '.join(JUDGE_STYLES)} (default: default)",
    )
    parser.add_argument(
        "--reference",
        nargs="+",
        default=["none"],
        dest="references",
        help="Reference modes: none, hipc (static HIPC injection). Default: none",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Create and save experiment context for provenance tracking
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

    predictions = None
    scoring_results = None

    try:
        if args.phase in ["predict", "all"]:
            predictions = run_predict(
                test_cases_dir=args.test_cases,
                output_dir=args.output,
                models=args.models,
                n_bootstrap=args.n_bootstrap,
                dry_run=args.dry_run,
                resume=args.resume,
                max_cases=args.max_cases,
                run_id=ctx.run_id,  # Pass run_id for provenance
                references=args.references,
            )

        if args.phase in ["score", "all"]:
            scoring_results = run_score(
                test_cases_dir=args.test_cases,
                output_dir=args.output,
                predictions=predictions,
            )

        if args.phase in ["judge", "all"]:
            run_judge(
                test_cases_dir=args.test_cases,
                output_dir=args.output,
                scoring_results=scoring_results,
                dry_run=args.dry_run,
                judge_model=args.judge_model,
                judge_style=args.judge_style,
            )

        # Mark experiment as completed with prediction stats
        if predictions is not None:
            errors = len([p for p in predictions if p.error])
            ctx.mark_completed(
                total=len(predictions),
                successful=len(predictions) - errors,
                failed=errors,
                output_dir=args.output,
            )
        else:
            # For score/judge-only runs, just mark completed
            ctx.mark_completed(0, 0, 0, args.output)

        print("\n✓ Pipeline complete!")

    except Exception as e:
        ctx.mark_failed(args.output, str(e))
        print(f"\n✗ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
