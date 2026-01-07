"""Ablation study runner for systematic prompt evaluation."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from drugdevbench.data.schemas import (
    Annotation,
    AblationResult,
    BenchmarkResult,
    EvaluationResponse,
    FigureType,
    PromptCondition,
)
from drugdevbench.evaluation.metrics import compute_ablation_metrics, compute_benchmark_metrics
from drugdevbench.evaluation.rubric import score_response
from drugdevbench.models import DrugDevBenchEvaluator, EvaluatorConfig
from drugdevbench.prompts import build_system_prompt


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""

    conditions: list[PromptCondition] = field(
        default_factory=lambda: [
            PromptCondition.VANILLA,
            PromptCondition.BASE_ONLY,
            PromptCondition.PERSONA_ONLY,
            PromptCondition.BASE_PLUS_SKILL,
            PromptCondition.FULL_STACK,
        ]
    )
    models: list[str] = field(default_factory=lambda: ["claude-haiku"])
    figure_types: list[FigureType] | None = None  # None = all types
    max_figures: int | None = None  # None = all figures
    max_questions_per_figure: int | None = None
    output_dir: Path = Path("results")
    save_responses: bool = True
    verbose: bool = True


class AblationRunner:
    """Runner for ablation studies."""

    def __init__(
        self,
        evaluator: DrugDevBenchEvaluator | None = None,
        config: AblationConfig | None = None,
    ):
        """Initialize the ablation runner.

        Args:
            evaluator: Evaluator instance (creates default if None)
            config: Ablation configuration
        """
        self.evaluator = evaluator or DrugDevBenchEvaluator(
            EvaluatorConfig(use_cache=True)
        )
        self.config = config or AblationConfig()
        self.console = Console()

    def run(
        self,
        annotations: list[Annotation],
    ) -> dict[str, AblationResult]:
        """Run the ablation study.

        Args:
            annotations: List of figure annotations with questions

        Returns:
            Dictionary of AblationResult by model
        """
        # Filter annotations by figure type if specified
        if self.config.figure_types:
            annotations = [
                a for a in annotations if a.figure.figure_type in self.config.figure_types
            ]

        # Limit number of figures if specified
        if self.config.max_figures:
            annotations = annotations[: self.config.max_figures]

        run_id = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_by_model: dict[str, AblationResult] = {}

        # Progress tracking
        total_tasks = (
            len(self.config.models)
            * len(self.config.conditions)
            * sum(len(a.questions) for a in annotations)
        )

        if self.config.verbose:
            self.console.print(f"\n[bold]Starting ablation study: {run_id}[/bold]")
            self.console.print(f"Models: {self.config.models}")
            self.console.print(f"Conditions: {[c.value for c in self.config.conditions]}")
            self.console.print(f"Figures: {len(annotations)}")
            self.console.print(f"Total evaluations: {total_tasks}\n")

        with Progress() as progress:
            task = progress.add_task("[cyan]Running ablation...", total=total_tasks)

            for model in self.config.models:
                results_by_condition: dict[PromptCondition, BenchmarkResult] = {}

                for condition in self.config.conditions:
                    responses = self._run_condition(
                        annotations=annotations,
                        model=model,
                        condition=condition,
                        progress=progress,
                        task=task,
                    )

                    # Compute metrics for this condition
                    benchmark_result = compute_benchmark_metrics(
                        responses=responses,
                        run_id=f"{run_id}_{model}_{condition.value}",
                        model=model,
                        condition=condition,
                    )
                    results_by_condition[condition] = benchmark_result

                    # Save responses if configured
                    if self.config.save_responses:
                        self._save_responses(
                            responses, run_id, model, condition.value
                        )

                # Compute ablation metrics for this model
                ablation_result = compute_ablation_metrics(
                    results_by_condition=results_by_condition,
                    run_id=run_id,
                    model=model,
                )
                results_by_model[model] = ablation_result

        # Print summary
        if self.config.verbose:
            self._print_summary(results_by_model)

        # Save results
        self._save_results(results_by_model, run_id)

        return results_by_model

    def _run_condition(
        self,
        annotations: list[Annotation],
        model: str,
        condition: PromptCondition,
        progress: Progress,
        task: TaskID,
    ) -> list[EvaluationResponse]:
        """Run evaluation for a single condition.

        Args:
            annotations: Figure annotations
            model: Model to use
            condition: Prompt condition
            progress: Progress bar
            task: Progress task ID

        Returns:
            List of evaluation responses
        """
        responses = []

        for annotation in annotations:
            figure = annotation.figure
            questions = annotation.questions

            # Limit questions if configured
            if self.config.max_questions_per_figure:
                questions = questions[: self.config.max_questions_per_figure]

            # Build system prompt for this condition and figure type
            system_prompt = build_system_prompt(
                condition=condition,
                figure_type=figure.figure_type,
            )

            for question in questions:
                # Run evaluation
                response = self.evaluator.evaluate(
                    figure_id=figure.figure_id,
                    question_id=question.question_id,
                    image_path=figure.image_path,
                    question=question.question_text,
                    system_prompt=system_prompt,
                    condition=condition,
                    model=model,
                    gold_answer=question.gold_answer,
                )

                # Score the response
                scoring_result = score_response(
                    response_text=response.response_text,
                    gold_answer=question.gold_answer,
                    question_type=question.question_type,
                )

                response.score = scoring_result.score
                response.scoring_rationale = scoring_result.rationale
                response.metadata["question_type"] = question.question_type.value

                responses.append(response)
                progress.advance(task)

        return responses

    def _save_responses(
        self,
        responses: list[EvaluationResponse],
        run_id: str,
        model: str,
        condition: str,
    ) -> None:
        """Save responses to a JSON file."""
        output_dir = self.config.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"responses_{model}_{condition}.json"
        with open(output_path, "w") as f:
            json.dump(
                [r.model_dump(mode="json") for r in responses],
                f,
                indent=2,
                default=str,
            )

    def _save_results(
        self,
        results: dict[str, AblationResult],
        run_id: str,
    ) -> None:
        """Save ablation results to a JSON file."""
        output_dir = self.config.output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "ablation_results.json"
        with open(output_path, "w") as f:
            json.dump(
                {model: r.model_dump(mode="json") for model, r in results.items()},
                f,
                indent=2,
                default=str,
            )

        if self.config.verbose:
            self.console.print(f"\n[green]Results saved to {output_path}[/green]")

    def _print_summary(self, results: dict[str, AblationResult]) -> None:
        """Print a summary table of results."""
        for model, ablation in results.items():
            self.console.print(f"\n[bold]Results for {model}[/bold]")

            table = Table(title="Ablation Results")
            table.add_column("Condition", style="cyan")
            table.add_column("Mean Score", justify="right")
            table.add_column("Std", justify="right")
            table.add_column("N", justify="right")
            table.add_column("Improvement", justify="right")

            for condition_name, result in ablation.results_by_condition.items():
                improvement = ablation.improvements.get(condition_name, 0)
                imp_str = f"{improvement:+.1f}%" if condition_name != "vanilla" else "-"

                table.add_row(
                    condition_name,
                    f"{result.mean_score:.3f}",
                    f"{result.std_score:.3f}",
                    str(result.n_questions),
                    imp_str,
                )

            self.console.print(table)


def run_ablation_study(
    annotations: list[Annotation],
    config: AblationConfig | None = None,
) -> dict[str, AblationResult]:
    """Convenience function to run an ablation study.

    Args:
        annotations: List of figure annotations
        config: Optional ablation configuration

    Returns:
        Dictionary of AblationResult by model
    """
    runner = AblationRunner(config=config)
    return runner.run(annotations)
