#!/usr/bin/env python3
"""Test the DrugDevBench pipeline with mock evaluation.

This script runs the full pipeline on sample data without making API calls,
then generates a PDF report with results.

Usage:
    python scripts/test_pipeline.py
"""

import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drugdevbench.data import load_annotations, PromptCondition, FigureType
from drugdevbench.data.schemas import AblationResult, BenchmarkResult, EvaluationResponse
from drugdevbench.models import MockEvaluator, EvaluatorConfig
from drugdevbench.prompts import build_system_prompt
from drugdevbench.evaluation import (
    score_response,
    compute_benchmark_metrics,
    create_ablation_report,
    generate_quick_summary,
)


def run_mock_ablation(
    annotations: list,
    models: list[str],
    conditions: list[PromptCondition],
    max_figures: int | None = None,
) -> dict[str, AblationResult]:
    """Run mock ablation study on annotations.

    Args:
        annotations: List of annotation objects
        models: List of model names to simulate
        conditions: List of conditions to test
        max_figures: Maximum number of figures to evaluate

    Returns:
        Dictionary of AblationResult by model
    """
    print(f"\n{'='*60}")
    print("RUNNING MOCK ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Models: {models}")
    print(f"Conditions: {[c.value for c in conditions]}")
    print(f"Annotations: {len(annotations)}")

    if max_figures:
        annotations = annotations[:max_figures]
        print(f"Limited to: {max_figures} figures")

    results = {}

    for model in models:
        print(f"\n--- Evaluating: {model} ---")
        evaluator = MockEvaluator(EvaluatorConfig(default_model=model))

        condition_results = {}
        all_responses = []

        for condition in conditions:
            print(f"  Condition: {condition.value}", end=" ")

            responses = []
            for ann in annotations:
                figure = ann.figure

                # Get system prompt for this condition
                system_prompt = build_system_prompt(
                    condition=condition,
                    figure_type=figure.figure_type,
                )

                for question in ann.questions:
                    response = evaluator.evaluate(
                        figure_id=figure.figure_id,
                        question_id=question.question_id,
                        image_path=figure.image_path,
                        question=question.question_text,
                        system_prompt=system_prompt,
                        condition=condition,
                        model=model,
                        gold_answer=question.gold_answer,
                        question_type=question.question_type,
                    )
                    responses.append(response)

            # Score responses
            scores = []
            for resp in responses:
                scoring = score_response(
                    response_text=resp.response_text,
                    gold_answer=resp.gold_answer,
                    question_type=resp.metadata.get("question_type", "factual_extraction"),
                )
                scores.append(scoring.score)

            # Compute metrics for this condition
            import numpy as np
            mean_score = np.mean(scores) if scores else 0.0
            std_score = np.std(scores) if scores else 0.0
            total_cost = sum(r.cost_usd for r in responses)

            condition_results[condition.value] = BenchmarkResult(
                run_id=str(uuid.uuid4())[:8],
                model=model,
                condition=condition,
                mean_score=mean_score,
                std_score=std_score,
                n_figures=len(annotations),
                n_questions=len(responses),
                total_cost_usd=total_cost,
                total_time_s=0.0,
            )

            print(f"-> {mean_score:.3f} (n={len(responses)})")
            all_responses.extend(responses)

        # Calculate improvements over vanilla baseline
        improvements = {}
        if "vanilla" in condition_results:
            baseline = condition_results["vanilla"].mean_score
            if baseline > 0:
                for cond_name, result in condition_results.items():
                    if cond_name != "vanilla":
                        improvements[cond_name] = (
                            (result.mean_score - baseline) / baseline * 100
                        )

        results[model] = AblationResult(
            run_id=str(uuid.uuid4())[:8],
            model=model,
            results_by_condition=condition_results,
            improvements=improvements,
        )

        print(f"\n  Total evaluations for {model}: {evaluator.call_count}")

    return results


def main():
    print("=" * 60)
    print("DRUGDEVBENCH PIPELINE TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    project_dir = Path(__file__).parent.parent
    annotations_path = project_dir / "data" / "annotations" / "sample_annotations.jsonl"
    output_dir = project_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # Load annotations
    print(f"\nLoading annotations from: {annotations_path}")
    annotations = load_annotations(annotations_path)

    if not annotations:
        print("ERROR: No annotations found!")
        print("Run 'python scripts/generate_sample_figures.py' first.")
        sys.exit(1)

    print(f"Loaded {len(annotations)} annotations")

    # Count by figure type
    type_counts = defaultdict(int)
    for ann in annotations:
        type_counts[ann.figure.figure_type.value] += 1

    print("\nFigures by type:")
    for fig_type, count in sorted(type_counts.items()):
        print(f"  {fig_type}: {count}")

    # Define test parameters
    models = ["claude-haiku-mock", "gemini-flash-mock"]
    conditions = list(PromptCondition)

    # Run mock ablation (limit to 50 figures for quick test)
    results = run_mock_ablation(
        annotations=annotations,
        models=models,
        conditions=conditions,
        max_figures=50,
    )

    # Generate text summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    summary = generate_quick_summary(results)
    print(summary)

    # Save text summary
    summary_path = output_dir / "pipeline_test_summary.txt"
    summary_path.write_text(summary)
    print(f"\nSaved summary to: {summary_path}")

    # Generate PDF report
    print("\nGenerating PDF report...")
    report_path = output_dir / "pipeline_test_report.pdf"

    try:
        create_ablation_report(
            results=results,
            output_path=report_path,
            title="DrugDevBench Pipeline Test Report",
        )
        print(f"Generated PDF report: {report_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
