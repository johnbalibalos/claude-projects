"""
Hypothesis pipeline orchestrator.

Combines all components to run experiments with:
- Multiple reasoning strategies (CoT, WoT, direct, etc.)
- Multiple RAG configurations
- Multiple context levels
- Multiple tool configurations

Generates a condition matrix and runs all combinations with checkpointing.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

# Import from sibling package
sys.path.insert(0, str(Path(__file__).parent.parent))
from checkpoint import CheckpointedRunner

from .base import ContextBuilder, Evaluator, PromptStrategy, ToolRegistry
from .context import get_context_builder

# Use shared model_client library
sys.path.insert(0, str(Path(__file__).parent.parent / "model_client"))
# Import PipelineConfig from config module (avoid circular import at runtime)
# Use TYPE_CHECKING for type hints only
from typing import TYPE_CHECKING

from model_client import AnthropicClient, ClientConfig, ModelResponse, OpenAIClient

from .models import (
    ContextLevel,
    ExperimentResults,
    HypothesisCondition,
    RAGMode,
    ReasoningType,
    TrialInput,
    TrialResult,
)
from .rag import NoRAGProvider, RAGProvider, get_provider_for_mode
from .strategies import get_strategy

if TYPE_CHECKING:
    from .config import PipelineConfig


class ProgressReporter:
    """Handles progress reporting for pipeline execution."""

    def __init__(self, name: str, conditions: list, trial_count: int, total: int):
        self.name = name
        self.conditions = conditions
        self.trial_count = trial_count
        self.total = total

    def print_header(self) -> None:
        """Print experiment header."""
        print(f"\n{'='*60}")
        print(f"HYPOTHESIS PIPELINE: {self.name}")
        print(f"{'='*60}")
        print(f"Trial inputs: {self.trial_count}")
        print(f"Conditions: {len(self.conditions)}")
        print(f"Total trials: {self.total}")
        print("\nCondition matrix:")
        for cond in self.conditions:
            print(f"  - {cond.name}")
        print(f"{'='*60}\n")

    def print_progress(self, completed: int, cond_name: str, trial_id: str) -> None:
        """Print progress line start."""
        pct = (completed + 1) / self.total * 100
        print(f"[{pct:5.1f}%] {cond_name} / {trial_id}...", end=" ", flush=True)

    def print_result(self, result: TrialResult) -> None:
        """Print trial result."""
        if result.error:
            print(f"ERROR: {result.error[:50]}")
        else:
            score_str = ", ".join(
                f"{k}={v:.2f}" for k, v in list(result.scores.items())[:3]
            )
            print(f"{score_str}, {result.latency_seconds:.1f}s")


class HypothesisPipeline:
    """
    Main pipeline orchestrator for hypothesis testing.

    Generates condition matrix from configuration and runs all combinations.
    Supports checkpointing for long-running experiments.
    """

    def __init__(
        self,
        config: PipelineConfig,
        evaluator: Evaluator,
        trial_inputs: list[TrialInput],
        tool_registry: ToolRegistry | None = None,
        rag_providers: dict[RAGMode, RAGProvider] | None = None,
        context_builders: dict[ContextLevel, ContextBuilder] | None = None,
        strategies: dict[ReasoningType, PromptStrategy] | None = None,
        output_schema: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
            evaluator: Evaluator for scoring responses
            trial_inputs: List of trial inputs to test
            tool_registry: Registry of available tools
            rag_providers: Custom RAG providers by mode
            context_builders: Custom context builders by level
            strategies: Custom strategies by reasoning type
            output_schema: Expected output format
            examples: Few-shot examples
        """
        self.config = config
        self.evaluator = evaluator
        self.trial_inputs = trial_inputs
        self.tool_registry = tool_registry or ToolRegistry()
        self.output_schema = output_schema
        self.examples = examples

        # Initialize components
        self._init_rag_providers(rag_providers)
        self._init_context_builders(context_builders)
        self._init_strategies(strategies)

        # Model clients are created on-demand via create_client()

        # Generate conditions
        self.conditions = self._generate_conditions()

        # Setup checkpointing
        self.checkpoint = CheckpointedRunner(
            f"{config.name}_pipeline",
            checkpoint_dir=config.checkpoint_dir,
        )

    def _init_rag_providers(self, custom: dict[RAGMode, RAGProvider] | None) -> None:
        """Initialize RAG providers."""
        self.rag_providers: dict[RAGMode, RAGProvider] = {
            RAGMode.NONE: NoRAGProvider(),
        }
        if custom:
            self.rag_providers.update(custom)

        # Create providers from config
        for mode in self.config.rag_modes:
            if mode not in self.rag_providers:
                mode_config = self.config.rag_configs.get(mode.value, {})
                self.rag_providers[mode] = get_provider_for_mode(mode, **mode_config)

    def _init_context_builders(self, custom: dict[ContextLevel, ContextBuilder] | None) -> None:
        """Initialize context builders."""
        self.context_builders: dict[ContextLevel, ContextBuilder] = {}
        if custom:
            self.context_builders.update(custom)

        # Create builders from config
        for level in self.config.context_levels:
            if level not in self.context_builders:
                level_config = self.config.context_configs.get(level.value, {})
                self.context_builders[level] = get_context_builder(level, **level_config)

    def _init_strategies(self, custom: dict[ReasoningType, PromptStrategy] | None) -> None:
        """Initialize prompt strategies."""
        self.strategies: dict[ReasoningType, PromptStrategy] = {}
        if custom:
            self.strategies.update(custom)

        # Create strategies from config
        for reasoning_type in self.config.reasoning_types:
            if reasoning_type not in self.strategies:
                type_config = self.config.strategy_configs.get(reasoning_type.value, {})
                self.strategies[reasoning_type] = get_strategy(reasoning_type, **type_config)

    def _generate_conditions(self) -> list[HypothesisCondition]:
        """Generate all condition combinations from config."""
        conditions = []

        # Use temperatures list if specified, otherwise use single temperature value
        temperatures = self.config.temperatures or [self.config.temperature]

        # Cartesian product of all dimensions (now including temperature)
        for model, reasoning, context, rag, tools, temp in product(
            self.config.models,
            self.config.reasoning_types,
            self.config.context_levels,
            self.config.rag_modes,
            self.config.tool_configs,
            temperatures,
        ):
            # Generate condition name
            name_parts = [
                model.split("-")[1] if "-" in model else model,
                reasoning.value,
                context.value,
                rag.value,
            ]
            if tools:
                name_parts.append("tools")
            # Only add temperature to name if we're varying it
            if len(temperatures) > 1:
                name_parts.append(f"t{temp:.1f}")

            condition = HypothesisCondition(
                name="_".join(name_parts),
                model=model,
                reasoning_type=reasoning,
                reasoning_config=self.config.strategy_configs.get(reasoning.value, {}),
                context_level=context,
                context_config=self.config.context_configs.get(context.value, {}),
                rag_mode=rag,
                rag_config=self.config.rag_configs.get(rag.value, {}),
                tools_enabled=bool(tools),
                tool_names=tools,
                max_tokens=self.config.max_tokens,
                temperature=temp,
            )
            conditions.append(condition)

        return conditions

    def _build_full_prompt(
        self,
        trial_input: TrialInput,
        condition: HypothesisCondition,
    ) -> tuple[str, list[str]]:
        """
        Build the complete prompt for a trial.

        Returns:
            Tuple of (prompt, rag_documents)
        """
        # 1. Get RAG documents
        rag_provider = self.rag_providers.get(condition.rag_mode, NoRAGProvider())
        rag_documents = rag_provider.get_context_for_trial(trial_input)

        # 2. Get tool descriptions
        tool_descriptions = []
        if condition.tools_enabled:
            tools = self.tool_registry.get_tools(condition.tool_names)
            tool_descriptions = [f"{t.name}: {t.description}" for t in tools]

        # 3. Build context
        context_builder = self.context_builders.get(condition.context_level)
        if not context_builder:
            context = trial_input.prompt
        else:
            context = context_builder.build_context(
                trial_input,
                rag_documents=rag_documents,
                tool_descriptions=tool_descriptions,
            )

        # 4. Apply strategy to build final prompt
        strategy = self.strategies.get(condition.reasoning_type)
        if not strategy:
            prompt = context
        else:
            prompt = strategy.build_prompt(
                base_prompt=trial_input.prompt,
                context=context,
                output_schema=self.output_schema,
                examples=self.examples,
            )

        return prompt, rag_documents

    def _call_model(
        self,
        prompt: str,
        condition: HypothesisCondition,
    ) -> tuple[str, list[dict], int, int]:
        """
        Call the model.

        Returns:
            Tuple of (response, tool_calls, input_tokens, output_tokens)
        """
        # Create client based on model type
        config = ClientConfig()
        if "claude" in condition.model.lower():
            client = AnthropicClient(config)
        elif "gpt" in condition.model.lower():
            client = OpenAIClient(config)
        else:
            raise ValueError(f"Unknown model: {condition.model}")

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Get tools if enabled
        kwargs = {}
        if condition.tools_enabled:
            kwargs["tools"] = self.tool_registry.get_anthropic_tools(condition.tool_names)

        response: ModelResponse = client.generate(
            messages,
            model=condition.model,
            max_tokens=condition.max_tokens,
            temperature=condition.temperature,
            **kwargs,
        )

        return response.content, [], response.usage.input_tokens, response.usage.output_tokens

    def _run_trial(
        self,
        trial_input: TrialInput,
        condition: HypothesisCondition,
    ) -> TrialResult:
        """Run a single trial."""
        start_time = datetime.now()

        try:
            # Build prompt
            prompt, rag_documents = self._build_full_prompt(trial_input, condition)

            # Call model
            raw_response, tool_calls, input_tokens, output_tokens = self._call_model(
                prompt, condition
            )

            # Extract answer using strategy
            strategy = self.strategies.get(condition.reasoning_type)
            if strategy:
                final_answer = strategy.extract_final_answer(raw_response)
            else:
                final_answer = raw_response

            # Evaluate
            extracted = self.evaluator.extract(final_answer)
            scores = self.evaluator.score(extracted, trial_input.ground_truth)

            end_time = datetime.now()

            return TrialResult(
                trial_id=trial_input.id,
                condition_name=condition.name,
                start_time=start_time,
                end_time=end_time,
                latency_seconds=(end_time - start_time).total_seconds(),
                raw_response=raw_response,
                extracted_output=extracted,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                scores=scores,
                context_used=prompt[:1000],  # Truncate for storage
                rag_documents=rag_documents,
            )

        except Exception as e:
            end_time = datetime.now()
            return TrialResult(
                trial_id=trial_input.id,
                condition_name=condition.name,
                start_time=start_time,
                end_time=end_time,
                latency_seconds=(end_time - start_time).total_seconds(),
                error=str(e),
            )

    def run(self, verbose: bool = True) -> ExperimentResults:
        """
        Run the complete experiment.

        Args:
            verbose: Whether to print progress

        Returns:
            Complete experiment results
        """
        results = ExperimentResults(
            experiment_name=self.config.name,
            started_at=datetime.now(),
            conditions=self.conditions,
        )

        all_pairs = [
            (trial, cond)
            for trial in self.trial_inputs
            for cond in self.conditions
        ]
        total = len(all_pairs)

        # Setup progress reporting
        reporter = ProgressReporter(
            self.config.name, self.conditions, len(self.trial_inputs), total
        ) if verbose else None
        if reporter:
            reporter.print_header()

        # Iterate with checkpointing
        for trial, cond in self.checkpoint.iterate(
            all_pairs,
            key_fn=lambda x: f"{x[0].id}_{x[1].name}",
        ):
            key = f"{trial.id}_{cond.name}"
            completed, _ = self.checkpoint.progress()

            if reporter:
                reporter.print_progress(completed, cond.name, trial.id)

            result = self._run_trial(trial, cond)
            results.trials.append(result)
            self.checkpoint.save_result(key, result.to_dict())

            if reporter:
                reporter.print_result(result)

        # Load any previously completed trials not in current run
        for _key, data in self.checkpoint.get_all_results().items():
            if not any(
                t.trial_id == data["trial_id"] and t.condition_name == data["condition_name"]
                for t in results.trials
            ):
                # Reconstruct TrialResult
                trial_result = TrialResult(
                    trial_id=data["trial_id"],
                    condition_name=data["condition_name"],
                    start_time=datetime.fromisoformat(data["start_time"]),
                    end_time=datetime.fromisoformat(data["end_time"]),
                    latency_seconds=data["latency_seconds"],
                    raw_response=data.get("raw_response", ""),
                    extracted_output=data.get("extracted_output"),
                    tool_calls=data.get("tool_calls", []),
                    input_tokens=data.get("input_tokens", 0),
                    output_tokens=data.get("output_tokens", 0),
                    scores=data.get("scores", {}),
                    error=data.get("error"),
                )
                results.trials.append(trial_result)

        results.completed_at = datetime.now()

        # Compute aggregated metrics
        results.metrics_by_condition = self._compute_metrics(results)

        # Save results
        self._save_results(results)

        return results

    def _compute_metrics(self, results: ExperimentResults) -> dict[str, dict[str, float]]:
        """Compute aggregated metrics by condition."""
        metrics = {}

        for cond in self.conditions:
            trials = results.filter_by_condition(cond.name)
            if not trials:
                continue

            successful = [t for t in trials if t.success]
            cond_metrics = {
                "n_trials": len(trials),
                "n_successful": len(successful),
                "success_rate": len(successful) / len(trials) if trials else 0,
            }

            # Aggregate scores
            if successful:
                all_metrics = set()
                for t in successful:
                    all_metrics.update(t.scores.keys())

                for metric in all_metrics:
                    scores = [t.scores[metric] for t in successful if metric in t.scores]
                    if scores:
                        cond_metrics[f"{metric}_mean"] = sum(scores) / len(scores)
                        cond_metrics[f"{metric}_min"] = min(scores)
                        cond_metrics[f"{metric}_max"] = max(scores)

                # Latency
                latencies = [t.latency_seconds for t in successful]
                cond_metrics["latency_mean"] = sum(latencies) / len(latencies)

                # Tokens
                input_tokens = [t.input_tokens for t in successful]
                output_tokens = [t.output_tokens for t in successful]
                cond_metrics["input_tokens_mean"] = sum(input_tokens) / len(input_tokens)
                cond_metrics["output_tokens_mean"] = sum(output_tokens) / len(output_tokens)

            metrics[cond.name] = cond_metrics

        return metrics

    def _save_results(self, results: ExperimentResults) -> None:
        """Save results to disk."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = results.started_at.strftime("%Y%m%d_%H%M%S")
        path = self.config.output_dir / f"{self.config.name}_results_{timestamp}.json"

        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        print(f"\nResults saved to: {path}")

    def generate_report(self, results: ExperimentResults) -> str:
        """Generate a markdown report from results."""
        lines = [
            f"# Hypothesis Pipeline Report: {results.experiment_name}",
            "",
            f"**Started:** {results.started_at}",
            f"**Completed:** {results.completed_at}",
            f"**Total Trials:** {len(results.trials)}",
            "",
            "## Condition Matrix",
            "",
        ]

        # Condition summary table
        lines.append("| Condition | Reasoning | Context | RAG | Tools | Success Rate |")
        lines.append("|-----------|-----------|---------|-----|-------|--------------|")

        for cond in self.conditions:
            metrics = results.metrics_by_condition.get(cond.name, {})
            success_rate = metrics.get("success_rate", 0)
            lines.append(
                f"| {cond.name} | {cond.reasoning_type.value} | "
                f"{cond.context_level.value} | {cond.rag_mode.value} | "
                f"{'Yes' if cond.tools_enabled else 'No'} | {success_rate:.1%} |"
            )

        lines.append("")
        lines.append("## Results by Condition")
        lines.append("")

        for cond in self.conditions:
            metrics = results.metrics_by_condition.get(cond.name, {})
            if not metrics:
                continue

            lines.append(f"### {cond.name}")
            lines.append("")

            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.3f}")
                else:
                    lines.append(f"- **{key}:** {value}")

            lines.append("")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_simple_pipeline(
    name: str,
    trial_inputs: list[TrialInput],
    evaluator: Evaluator,
    models: list[str] | None = None,
    reasoning_types: list[ReasoningType] | None = None,
    context_levels: list[ContextLevel] | None = None,
    **kwargs: Any,
) -> HypothesisPipeline:
    """
    Create a simple pipeline with common defaults.

    Args:
        name: Experiment name
        trial_inputs: Trial inputs to test
        evaluator: Evaluator for scoring
        models: Models to test
        reasoning_types: Reasoning strategies to test
        context_levels: Context levels to test
        **kwargs: Additional pipeline config

    Returns:
        Configured pipeline
    """
    config = PipelineConfig(
        name=name,
        models=models or ["claude-sonnet-4-20250514"],
        reasoning_types=reasoning_types or [ReasoningType.DIRECT, ReasoningType.COT],
        context_levels=context_levels or [ContextLevel.MINIMAL, ContextLevel.STANDARD],
        **kwargs,
    )

    return HypothesisPipeline(
        config=config,
        evaluator=evaluator,
        trial_inputs=trial_inputs,
    )
