"""
Experiment runner for the gating benchmark.

Orchestrates running LLM predictions across all conditions
and collecting results.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from curation.schemas import TestCase
from curation.omip_extractor import load_all_test_cases
from evaluation.scorer import GatingScorer, ScoringResult
from .conditions import ExperimentCondition, get_standard_conditions
from .prompts import build_prompt

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


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
    n_runs: int = 1  # Number of runs for statistical significance
    random_seed: int | None = None  # For reproducibility across runs


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
    run_number: int = 1  # Which run this is (for multi-run experiments)

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
    """Aggregated result from multiple experiment runs with confidence intervals."""

    config: ExperimentConfig
    runs: list[ExperimentResult]
    aggregate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def compute_statistics(self) -> None:
        """Compute aggregate statistics with confidence intervals."""
        import statistics

        # Collect metrics by condition across all runs
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

        # Compute mean, std, and 95% CI for each condition
        for condition, metrics in metrics_by_condition.items():
            self.aggregate_metrics[condition] = {}
            for metric_name, values in metrics.items():
                if len(values) >= 2:
                    mean = statistics.mean(values)
                    std = statistics.stdev(values)
                    n = len(values)
                    # 95% CI = mean ± 1.96 * (std / sqrt(n))
                    ci_margin = 1.96 * (std / (n ** 0.5))
                    self.aggregate_metrics[condition][f"{metric_name}_mean"] = mean
                    self.aggregate_metrics[condition][f"{metric_name}_std"] = std
                    self.aggregate_metrics[condition][f"{metric_name}_ci_low"] = mean - ci_margin
                    self.aggregate_metrics[condition][f"{metric_name}_ci_high"] = mean + ci_margin
                    self.aggregate_metrics[condition][f"{metric_name}_n"] = n
                elif len(values) == 1:
                    self.aggregate_metrics[condition][f"{metric_name}_mean"] = values[0]
                    self.aggregate_metrics[condition][f"{metric_name}_std"] = 0.0
                    self.aggregate_metrics[condition][f"{metric_name}_n"] = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_name": self.config.name,
            "n_runs": len(self.runs),
            "runs": [r.to_dict() for r in self.runs],
            "aggregate_metrics": self.aggregate_metrics,
        }

    def format_summary(self) -> str:
        """Format a human-readable summary with confidence intervals."""
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
                n = metrics.get(f"{key}_n", 0)
                lines.append(
                    f"  {key}: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}] (n={n})"
                )

        return "\n".join(lines)


class ExperimentRunner:
    """
    Runs experiments across test cases and conditions.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.scorer = GatingScorer()
        self._setup_logging()
        self._setup_clients()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_clients(self) -> None:
        """Initialize API clients."""
        self.anthropic_client = None
        self.openai_client = None
        self._ollama_client = None

        if Anthropic and os.environ.get("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic()
            logger.info("Anthropic client initialized")

        if OpenAI and os.environ.get("OPENAI_API_KEY"):
            self.openai_client = OpenAI()
            logger.info("OpenAI client initialized")

        # Ollama configuration
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def ollama_client(self):
        """Lazy initialization of Ollama HTTP client."""
        if self._ollama_client is None and HTTPX_AVAILABLE:
            self._ollama_client = httpx.Client(timeout=300.0)
            logger.info(f"Ollama client initialized at {self.ollama_base_url}")
        return self._ollama_client

    def run(self) -> ExperimentResult:
        """
        Run the complete experiment.

        Returns:
            ExperimentResult with all outcomes
        """
        result = ExperimentResult(
            config=self.config,
            start_time=datetime.now(),
        )

        # Load test cases
        test_cases = load_all_test_cases(self.config.test_cases_dir)
        logger.info(f"Loaded {len(test_cases)} test cases")

        if not test_cases:
            logger.warning("No test cases found!")
            result.end_time = datetime.now()
            return result

        # Ensure output directory exists
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run all conditions
        total = len(test_cases) * len(self.config.conditions)
        logger.info(f"Running {total} total evaluations")

        for i, condition in enumerate(self.config.conditions):
            logger.info(f"Condition {i+1}/{len(self.config.conditions)}: {condition.name}")

            for j, test_case in enumerate(test_cases):
                logger.info(
                    f"  Test case {j+1}/{len(test_cases)}: {test_case.test_case_id}"
                )

                try:
                    scoring_result = self._run_single(test_case, condition)
                    result.results.append(scoring_result)
                    result.total_api_calls += 1

                    # Log progress
                    if scoring_result.parse_success:
                        logger.info(
                            f"    F1={scoring_result.hierarchy_f1:.3f}, "
                            f"Structure={scoring_result.structure_accuracy:.3f}"
                        )
                    else:
                        logger.warning(f"    Parse failed: {scoring_result.parse_error}")

                except Exception as e:
                    error = {
                        "test_case_id": test_case.test_case_id,
                        "condition": condition.name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    result.errors.append(error)
                    logger.error(f"    Error: {e}")

                # Save intermediate results
                if (result.total_api_calls % 10) == 0:
                    self._save_intermediate(result, output_dir)

        result.end_time = datetime.now()

        # Save final results
        self._save_final(result, output_dir)

        return result

    def run_multi(self, n_runs: int | None = None) -> MultiRunResult:
        """
        Run the experiment multiple times for statistical significance.

        Args:
            n_runs: Number of runs (defaults to config.n_runs)

        Returns:
            MultiRunResult with aggregated statistics and confidence intervals
        """
        n_runs = n_runs or self.config.n_runs
        logger.info(f"Starting multi-run experiment with {n_runs} runs")

        runs: list[ExperimentResult] = []

        for run_num in range(1, n_runs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"RUN {run_num}/{n_runs}")
            logger.info(f"{'='*60}\n")

            # Run single experiment
            result = self._run_single_experiment(run_number=run_num)
            runs.append(result)

            # Log intermediate stats
            if result.results:
                avg_f1 = sum(r.hierarchy_f1 for r in result.results) / len(result.results)
                logger.info(f"Run {run_num} complete: avg F1 = {avg_f1:.3f}")

        # Aggregate results
        multi_result = MultiRunResult(
            config=self.config,
            runs=runs,
        )
        multi_result.compute_statistics()

        # Save multi-run results
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

        # Load test cases
        test_cases = load_all_test_cases(self.config.test_cases_dir)

        if not test_cases:
            logger.warning("No test cases found!")
            result.end_time = datetime.now()
            return result

        # Run all conditions
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

    def _save_multi_run_results(self, result: MultiRunResult, output_dir: Path) -> None:
        """Save multi-run experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        path = output_dir / f"multirun_results_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Multi-run results saved to: {path}")

        # Save summary
        summary_path = output_dir / f"multirun_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(result.format_summary())
        logger.info(f"Multi-run summary saved to: {summary_path}")

    def _run_single(
        self,
        test_case: TestCase,
        condition: ExperimentCondition,
    ) -> ScoringResult:
        """
        Run a single test case with a condition.

        Args:
            test_case: Test case to evaluate
            condition: Experimental condition

        Returns:
            ScoringResult
        """
        # Build prompt
        prompt = build_prompt(
            test_case=test_case,
            template_name=condition.prompt_strategy,
            context_level=condition.context_level,
        )

        # Get response
        if self.config.dry_run:
            response = self._mock_response(test_case)
        else:
            response = self._call_model(condition.model, prompt)

        # Score
        return self.scorer.score(
            response=response,
            test_case=test_case,
            model=condition.model,
            condition=condition.name,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def _call_model(self, model: str, prompt: str) -> str:
        """
        Call an LLM with retry logic.

        Args:
            model: Model identifier
            prompt: Prompt to send

        Returns:
            Model response text
        """
        if "claude" in model.lower():
            return self._call_anthropic(model, prompt)
        elif "gpt" in model.lower():
            return self._call_openai(model, prompt)
        else:
            # Assume Ollama for all other models (local LLMs)
            return self._call_ollama(model, prompt)

    def _call_anthropic(self, model: str, prompt: str) -> str:
        """Call Anthropic API."""
        if self.anthropic_client is None:
            raise RuntimeError("Anthropic client not initialized")

        message = self.anthropic_client.messages.create(
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    def _call_openai(self, model: str, prompt: str) -> str:
        """Call OpenAI API."""
        if self.openai_client is None:
            raise RuntimeError("OpenAI client not initialized")

        response = self.openai_client.chat.completions.create(
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content

    def _call_ollama(self, model: str, prompt: str) -> str:
        """
        Call Ollama API for local models.

        Args:
            model: Ollama model name (e.g., 'llama3.1:8b', 'qwen2.5:72b')
            prompt: Prompt to send

        Returns:
            Model response text
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not installed. Run: pip install httpx")

        if self.ollama_client is None:
            raise RuntimeError("Ollama client not available")

        response = self.ollama_client.post(
            f"{self.ollama_base_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        return data["message"].get("content", "")

    def _mock_response(self, test_case: TestCase) -> str:
        """Generate mock response for dry run."""
        # Return a simple mock hierarchy
        return json.dumps({
            "name": "All Events",
            "children": [
                {
                    "name": "Singlets",
                    "markers": ["FSC-A", "FSC-H"],
                    "children": [
                        {
                            "name": "Live",
                            "markers": ["Live/Dead"],
                            "children": []
                        }
                    ]
                }
            ]
        })

    def _save_intermediate(self, result: ExperimentResult, output_dir: Path) -> None:
        """Save intermediate results."""
        path = output_dir / "intermediate_results.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def _save_final(self, result: ExperimentResult, output_dir: Path) -> None:
        """Save final results."""
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"experiment_results_{timestamp}.json"

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Results saved to: {path}")

        # Also save a summary
        summary_path = output_dir / f"experiment_summary_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(self._generate_summary(result))

        logger.info(f"Summary saved to: {summary_path}")

    def _generate_summary(self, result: ExperimentResult) -> str:
        """Generate a text summary of results."""
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

        # Group by model
        from evaluation.scorer import compute_metrics_by_model
        by_model = compute_metrics_by_model(result.results)

        for model, metrics in by_model.items():
            lines.append(f"\n{model}:")
            lines.append(f"  Hierarchy F1: {metrics.get('hierarchy_f1_mean', 0):.3f}")
            lines.append(f"  Structure Accuracy: {metrics.get('structure_accuracy_mean', 0):.3f}")
            lines.append(f"  Critical Gate Recall: {metrics.get('critical_gate_recall_mean', 0):.3f}")

        lines.append("")
        lines.append("-" * 60)
        lines.append("METRICS BY CONDITION")
        lines.append("-" * 60)

        from evaluation.scorer import compute_metrics_by_condition
        by_condition = compute_metrics_by_condition(result.results)

        for condition, metrics in by_condition.items():
            lines.append(f"\n{condition}:")
            lines.append(f"  F1: {metrics.get('hierarchy_f1_mean', 0):.3f}")
            lines.append(f"  Parse Rate: {metrics.get('parse_success_rate', 0):.1%}")

        return "\n".join(lines)


def run_experiment(
    test_cases_dir: str,
    output_dir: str,
    name: str = "gating_benchmark",
    conditions: list[ExperimentCondition] | None = None,
    dry_run: bool = False,
) -> ExperimentResult:
    """
    Convenience function to run an experiment.

    Args:
        test_cases_dir: Directory with test case JSON files
        output_dir: Where to save results
        name: Experiment name
        conditions: Conditions to run (default: standard)
        dry_run: Whether to mock API calls

    Returns:
        ExperimentResult
    """
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

    print(f"\nExperiment complete!")
    print(f"Results: {len(result.results)}")
    print(f"Errors: {len(result.errors)}")
