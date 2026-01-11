"""
Experiment runner for the gating benchmark.

Orchestrates running LLM predictions across all conditions and collecting results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from curation.omip_extractor import load_all_test_cases
from curation.schemas import TestCase
from evaluation.scorer import GatingScorer, ScoringResult

from .conditions import ExperimentCondition, get_standard_conditions
from .llm_client import LLMClient, create_client
from .prompts import build_prompt

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str
    test_cases_dir: str
    output_dir: str
    conditions: list[ExperimentCondition] = field(default_factory=get_standard_conditions)
    max_tokens: int = 4096
    temperature: float = 0.0
    dry_run: bool = False
    log_level: str = "INFO"
    n_runs: int = 1
    checkpoint_every: int = 10


@dataclass
class ExperimentResult:
    """Result of a complete experiment run."""

    config: ExperimentConfig
    start_time: datetime
    end_time: datetime | None = None
    results: list[ScoringResult] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    total_api_calls: int = 0
    total_tokens: int = 0
    run_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_name": self.config.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "n_results": len(self.results),
            "n_errors": len(self.errors),
            "total_api_calls": self.total_api_calls,
            "total_tokens": self.total_tokens,
            "run_number": self.run_number,
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
        }


@dataclass
class MultiRunResult:
    """Aggregated result from multiple experiment runs."""

    config: ExperimentConfig
    runs: list[ExperimentResult]
    aggregate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def compute_statistics(self) -> None:
        """Compute aggregate statistics with confidence intervals."""
        import statistics

        metrics_by_condition: dict[str, dict[str, list[float]]] = {}

        for run in self.runs:
            for result in run.results:
                condition = result.condition
                if condition not in metrics_by_condition:
                    metrics_by_condition[condition] = {
                        "hierarchy_f1": [],
                        "structure_accuracy": [],
                        "critical_gate_recall": [],
                    }
                metrics_by_condition[condition]["hierarchy_f1"].append(result.hierarchy_f1)
                metrics_by_condition[condition]["structure_accuracy"].append(result.structure_accuracy)
                metrics_by_condition[condition]["critical_gate_recall"].append(result.critical_gate_recall)

        for condition, metrics in metrics_by_condition.items():
            self.aggregate_metrics[condition] = {}
            for metric_name, values in metrics.items():
                if len(values) >= 2:
                    mean = statistics.mean(values)
                    std = statistics.stdev(values)
                    n = len(values)
                    ci_margin = 1.96 * (std / (n ** 0.5))
                    self.aggregate_metrics[condition][f"{metric_name}_mean"] = mean
                    self.aggregate_metrics[condition][f"{metric_name}_std"] = std
                    self.aggregate_metrics[condition][f"{metric_name}_ci_low"] = mean - ci_margin
                    self.aggregate_metrics[condition][f"{metric_name}_ci_high"] = mean + ci_margin
                elif len(values) == 1:
                    self.aggregate_metrics[condition][f"{metric_name}_mean"] = values[0]
                    self.aggregate_metrics[condition][f"{metric_name}_std"] = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config.name,
            "n_runs": len(self.runs),
            "runs": [r.to_dict() for r in self.runs],
            "aggregate_metrics": self.aggregate_metrics,
        }

    def format_summary(self) -> str:
        lines = [
            "=" * 70,
            f"MULTI-RUN EXPERIMENT SUMMARY ({len(self.runs)} runs)",
            "=" * 70,
            "",
        ]
        for condition, metrics in sorted(self.aggregate_metrics.items()):
            lines.append(f"\n{condition}:")
            for key in ["hierarchy_f1", "structure_accuracy", "critical_gate_recall"]:
                mean = metrics.get(f"{key}_mean", 0)
                ci_low = metrics.get(f"{key}_ci_low", mean)
                ci_high = metrics.get(f"{key}_ci_high", mean)
                lines.append(f"  {key}: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        return "\n".join(lines)


class ExperimentRunner:
    """Runs experiments across test cases and conditions."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scorer = GatingScorer()
        self._clients: dict[str, LLMClient] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _get_client(self, model: str) -> LLMClient:
        """Get or create an LLM client for the given model."""
        if model not in self._clients:
            self._clients[model] = create_client(model, dry_run=self.config.dry_run)
        return self._clients[model]

    def run(self) -> ExperimentResult:
        """Run the complete experiment."""
        result = ExperimentResult(config=self.config, start_time=datetime.now())

        test_cases = load_all_test_cases(self.config.test_cases_dir)
        logger.info(f"Loaded {len(test_cases)} test cases")

        if not test_cases:
            logger.warning("No test cases found!")
            result.end_time = datetime.now()
            return result

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total = len(test_cases) * len(self.config.conditions)
        logger.info(f"Running {total} total evaluations")

        for i, condition in enumerate(self.config.conditions):
            logger.info(f"Condition {i+1}/{len(self.config.conditions)}: {condition.name}")

            for j, test_case in enumerate(test_cases):
                logger.info(f"  Test case {j+1}/{len(test_cases)}: {test_case.test_case_id}")

                try:
                    scoring_result = self._run_single(test_case, condition)
                    result.results.append(scoring_result)
                    result.total_api_calls += 1

                    if scoring_result.parse_success:
                        log_msg = (
                            f"    F1={scoring_result.hierarchy_f1:.3f}, "
                            f"Structure={scoring_result.structure_accuracy:.3f}"
                        )
                        if scoring_result.is_task_failure:
                            log_msg += f" [TASK FAILURE: {scoring_result.task_failure_type.value}]"
                        logger.info(log_msg)
                    else:
                        logger.warning(f"    Parse failed: {scoring_result.parse_error}")

                except Exception as e:
                    result.errors.append({
                        "test_case_id": test_case.test_case_id,
                        "condition": condition.name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    })
                    logger.error(f"    Error: {e}")

                if result.total_api_calls % self.config.checkpoint_every == 0:
                    self._save_checkpoint(result, output_dir)

        result.end_time = datetime.now()
        self._save_final(result, output_dir)
        return result

    def run_multi(self, n_runs: int | None = None) -> MultiRunResult:
        """Run the experiment multiple times for statistical significance."""
        n_runs = n_runs or self.config.n_runs
        logger.info(f"Starting multi-run experiment with {n_runs} runs")

        runs: list[ExperimentResult] = []
        for run_num in range(1, n_runs + 1):
            logger.info(f"\n{'='*60}\nRUN {run_num}/{n_runs}\n{'='*60}\n")
            result = self._run_single_experiment(run_number=run_num)
            runs.append(result)

            if result.results:
                avg_f1 = sum(r.hierarchy_f1 for r in result.results) / len(result.results)
                logger.info(f"Run {run_num} complete: avg F1 = {avg_f1:.3f}")

        multi_result = MultiRunResult(config=self.config, runs=runs)
        multi_result.compute_statistics()

        output_dir = Path(self.config.output_dir)
        self._save_multi_run_results(multi_result, output_dir)
        return multi_result

    def _run_single_experiment(self, run_number: int = 1) -> ExperimentResult:
        """Run a single experiment iteration."""
        result = ExperimentResult(
            config=self.config,
            start_time=datetime.now(),
            run_number=run_number,
        )

        test_cases = load_all_test_cases(self.config.test_cases_dir)
        if not test_cases:
            result.end_time = datetime.now()
            return result

        for condition in self.config.conditions:
            for test_case in test_cases:
                try:
                    scoring_result = self._run_single(test_case, condition)
                    result.results.append(scoring_result)
                    result.total_api_calls += 1
                except Exception as e:
                    result.errors.append({
                        "test_case_id": test_case.test_case_id,
                        "condition": condition.name,
                        "error": str(e),
                        "run_number": run_number,
                    })

        result.end_time = datetime.now()
        return result

    def _run_single(self, test_case: TestCase, condition: ExperimentCondition) -> ScoringResult:
        """Run a single test case with a condition."""
        prompt = build_prompt(
            test_case=test_case,
            template_name=condition.prompt_strategy,
            context_level=condition.context_level,
            rag_mode=condition.rag_mode,
        )

        client = self._get_client(condition.model)
        response = client.call(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return self.scorer.score(
            response=response.content,
            test_case=test_case,
            model=condition.model,
            condition=condition.name,
        )

    def _save_checkpoint(self, result: ExperimentResult, output_dir: Path) -> None:
        path = output_dir / "checkpoint.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _save_final(self, result: ExperimentResult, output_dir: Path) -> None:
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"experiment_results_{timestamp}.json"

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Results saved to: {path}")

        summary_path = output_dir / f"experiment_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(self._generate_summary(result))
        logger.info(f"Summary saved to: {summary_path}")

    def _save_multi_run_results(self, result: MultiRunResult, output_dir: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        path = output_dir / f"multirun_results_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Multi-run results saved to: {path}")

        summary_path = output_dir / f"multirun_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(result.format_summary())

    def _generate_summary(self, result: ExperimentResult) -> str:
        from evaluation.scorer import compute_metrics_by_condition, compute_metrics_by_model

        lines = [
            "=" * 60,
            "EXPERIMENT SUMMARY",
            "=" * 60,
            "",
            f"Experiment: {self.config.name}",
            f"Start: {result.start_time}",
            f"End: {result.end_time}",
            f"Duration: {result.end_time - result.start_time if result.end_time else 'N/A'}",
            "",
            f"Total API calls: {result.total_api_calls}",
            f"Successful: {len(result.results)}",
            f"Errors: {len(result.errors)}",
            "",
            "-" * 60,
            "METRICS BY MODEL",
            "-" * 60,
        ]

        by_model = compute_metrics_by_model(result.results)
        for model, metrics in by_model.items():
            lines.append(f"\n{model}:")
            lines.append(f"  Hierarchy F1: {metrics.get('hierarchy_f1_mean', 0):.3f}")
            lines.append(f"  Structure Accuracy: {metrics.get('structure_accuracy_mean', 0):.3f}")
            lines.append(f"  Critical Gate Recall: {metrics.get('critical_gate_recall_mean', 0):.3f}")

        lines.extend(["", "-" * 60, "METRICS BY CONDITION", "-" * 60])

        by_condition = compute_metrics_by_condition(result.results)
        for condition, metrics in by_condition.items():
            lines.append(f"\n{condition}:")
            lines.append(f"  F1: {metrics.get('hierarchy_f1_mean', 0):.3f}")
            lines.append(f"  Parse Rate: {metrics.get('parse_success_rate', 0):.1%}")
            lines.append(f"  Task Failure Rate: {metrics.get('task_failure_rate', 0):.1%}")
            failures_by_type = metrics.get("task_failures_by_type", {})
            if any(v > 0 for v in failures_by_type.values()):
                lines.append(f"    Meta-questions: {failures_by_type.get('meta_questions', 0)}")
                lines.append(f"    Refusals: {failures_by_type.get('refusals', 0)}")
                lines.append(f"    Instructions: {failures_by_type.get('instructions', 0)}")

        return "\n".join(lines)


def run_experiment(
    test_cases_dir: str,
    output_dir: str,
    name: str = "gating_benchmark",
    conditions: list[ExperimentCondition] | None = None,
    dry_run: bool = False,
) -> ExperimentResult:
    """Convenience function to run an experiment."""
    config = ExperimentConfig(
        name=name,
        test_cases_dir=test_cases_dir,
        output_dir=output_dir,
        conditions=conditions or get_standard_conditions(),
        dry_run=dry_run,
    )
    runner = ExperimentRunner(config)
    return runner.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run gating benchmark experiment")
    parser.add_argument("--test-cases", required=True, help="Directory with test cases")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--name", default="gating_benchmark", help="Experiment name")
    parser.add_argument("--dry-run", action="store_true", help="Mock API calls")

    args = parser.parse_args()

    result = run_experiment(
        test_cases_dir=args.test_cases,
        output_dir=args.output,
        name=args.name,
        dry_run=args.dry_run,
    )

    print("\nExperiment complete!")
    print(f"Results: {len(result.results)}")
    print(f"Errors: {len(result.errors)}")
