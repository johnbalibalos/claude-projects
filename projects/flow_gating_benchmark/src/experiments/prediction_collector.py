"""
Decoupled prediction collection from LLMs.

This module separates the concern of collecting raw LLM predictions
from scoring and judging. Enables modular pipeline architecture:

    PredictionCollector → BatchScorer → LLMJudge → ResultsAggregator

Checkpointing uses JSONL format for lock-free parallel writes.
Each worker appends predictions independently.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from curation.schemas import TestCase
from utils.serializable import SerializableMixin

from .conditions import ExperimentCondition
from .llm_client import create_client
from .prompts import build_prompt


@dataclass
class Prediction(SerializableMixin):
    """Raw LLM prediction before scoring.

    Provenance is tracked via run_id which references the manifest.json
    in the output directory. This avoids redundant metadata in every row.
    """

    test_case_id: str
    model: str
    condition: str
    bootstrap_run: int
    raw_response: str
    tokens_used: int
    timestamp: datetime
    prompt: str = ""
    error: str | None = None
    run_id: str = ""  # References manifest.json for full provenance

    @property
    def key(self) -> tuple:
        """Unique key for deduplication."""
        return (self.bootstrap_run, self.test_case_id, self.model, self.condition)


@dataclass
class CollectorConfig:
    """Configuration for prediction collection."""

    n_bootstrap: int = 3
    cli_delay_seconds: float = 0.5
    max_tokens: int = 6000  # Output token limit for predictions
    checkpoint_dir: Path | None = None
    dry_run: bool = False
    run_id: str = ""  # Set by RunManifest, references manifest.json

    # Per-provider parallel workers (rate limit aware)
    parallel_workers_gemini: int = 50
    parallel_workers_anthropic: int = 50  # Tier 2: 1000 RPM
    parallel_workers_openai: int = 50

    def get_parallel_workers(self, model: str) -> int:
        """Get parallel worker count for a model based on provider."""
        if model.startswith("gemini"):
            return self.parallel_workers_gemini
        elif model.startswith("claude") and not model.endswith("-cli"):
            return self.parallel_workers_anthropic
        elif model.startswith("gpt"):
            return self.parallel_workers_openai
        else:
            return 10  # Conservative default for unknown providers


class PredictionCollector:
    """Collects raw predictions from LLMs without scoring.

    Supports:
    - Concurrent CLI + API execution
    - Checkpoint/resume
    - Progress tracking
    - Cache-optimized iteration order
    """

    def __init__(
        self,
        test_cases: list[TestCase],
        conditions: list[ExperimentCondition],
        config: CollectorConfig | None = None,
    ):
        self.test_cases = test_cases
        self.conditions = conditions
        self.config = config or CollectorConfig()

        # Split conditions by execution type
        self.cli_conditions = [c for c in conditions if c.model.endswith("-cli")]
        self.api_conditions = [c for c in conditions if not c.model.endswith("-cli")]

        # Checkpoint paths (JSONL for lock-free parallel writes)
        self._checkpoint_dir = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir else None
        self._jsonl_path = self._checkpoint_dir / "predictions.jsonl" if self._checkpoint_dir else None

        # Track predictions
        self._predictions: list[Prediction] = []
        self._completed: set[tuple] = set()
        self._lock = threading.Lock()  # Only for in-memory tracking, not file writes
        self._progress_counter = 0

    @property
    def total_calls(self) -> int:
        """Total number of calls to make."""
        return len(self.test_cases) * len(self.conditions) * self.config.n_bootstrap

    def load_checkpoint(self) -> list[Prediction]:
        """Load predictions from JSONL checkpoint.

        Predictions with errors are excluded from the completed set,
        so they will be retried on resume.
        """
        predictions = self._load_jsonl() if self._jsonl_path and self._jsonl_path.exists() else []

        # Only mark error-free predictions as completed (retry errors on resume)
        successful = [p for p in predictions if not p.error]
        errored = [p for p in predictions if p.error]
        self._completed = {p.key for p in successful}
        self._predictions = successful

        if predictions:
            print(f"Loaded {len(predictions)} predictions from checkpoint")
            if errored:
                print(f"  {len(errored)} with errors will be retried")

        return successful

    def _load_jsonl(self) -> list[Prediction]:
        """Load predictions from JSONL file."""
        if not self._jsonl_path:
            return []

        predictions = []
        seen_keys = set()

        with open(self._jsonl_path) as f:
            for line_content in f:
                line_content = line_content.strip()
                if not line_content:
                    continue
                try:
                    data = json.loads(line_content)
                    pred = Prediction.from_dict(data)
                    # Deduplicate by key (keep latest)
                    if pred.key not in seen_keys:
                        predictions.append(pred)
                        seen_keys.add(pred.key)
                except (json.JSONDecodeError, KeyError):
                    continue  # Skip malformed lines

        return predictions

    def _append_jsonl(self, prediction: Prediction) -> None:
        """Append a single prediction to JSONL file (atomic, lock-free)."""
        if not self._jsonl_path or not self._checkpoint_dir:
            return

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(prediction.to_dict()) + "\n"

        # Atomic append - each write is independent
        with open(self._jsonl_path, "a") as f:
            f.write(line)

    def save_checkpoint(self, predictions: list[Prediction] | None = None) -> None:
        """Save predictions to JSON file (final output, not for checkpointing)."""
        if not self._checkpoint_dir:
            return
        if predictions is None:
            predictions = self._predictions
        output_path = self._checkpoint_dir.parent / "predictions.json"
        with open(output_path, "w") as f:
            json.dump([p.to_dict() for p in predictions], f, indent=2)

    def _add_prediction(self, prediction: Prediction) -> None:
        """Add prediction to memory and append to JSONL checkpoint."""
        # Append to JSONL immediately (lock-free file write)
        self._append_jsonl(prediction)

        # Update in-memory tracking (needs lock)
        with self._lock:
            self._predictions.append(prediction)

    def collect(
        self,
        resume: bool = False,
        progress_callback: Callable[[int, int, Prediction], None] | None = None,
    ) -> list[Prediction]:
        """Collect predictions from all models and conditions.

        Args:
            resume: Whether to resume from checkpoint
            progress_callback: Optional callback(current, total, prediction)

        Returns:
            List of all predictions
        """
        # Load checkpoint if resuming
        if resume:
            self._predictions = self.load_checkpoint()
            self._progress_counter = len(self._completed)

        # Collect from CLI and API concurrently
        # Predictions are added directly to self._predictions via _add_prediction

        if self.api_conditions:
            with ThreadPoolExecutor(max_workers=1) as bg_executor:
                # Start API in background
                api_future = bg_executor.submit(
                    self._collect_api_batch,
                    progress_callback,
                )

                # Run CLI in foreground
                if self.cli_conditions:
                    self._collect_cli_batch(progress_callback)

                # Wait for API
                api_future.result()

        elif self.cli_conditions:
            self._collect_cli_batch(progress_callback)

        # Save final checkpoint
        self.save_checkpoint()

        return self._predictions

    def _collect_cli_batch(
        self,
        progress_callback: Callable[[int, int, Prediction], None] | None = None,
    ) -> None:
        """Collect predictions from CLI models (sequential, rate-limited).

        Predictions are added directly to self._predictions with periodic checkpointing.
        """
        for condition in self.cli_conditions:
            client = create_client(condition.model, dry_run=self.config.dry_run)

            for test_case in self.test_cases:
                # Build prompt once for all bootstrap runs (cache optimization)
                prompt = build_prompt(
                    test_case,
                    template_name=condition.prompt_strategy,
                    context_level=condition.context_level,
                    rag_mode=condition.rag_mode,
                )

                for bootstrap_run in range(1, self.config.n_bootstrap + 1):
                    key = (bootstrap_run, test_case.test_case_id, condition.model, condition.name)

                    with self._lock:
                        if key in self._completed:
                            self._progress_counter += 1
                            continue

                    prediction = self._make_call(
                        test_case, condition, bootstrap_run, prompt, client
                    )

                    # Add to predictions with periodic checkpoint
                    self._add_prediction(prediction)

                    with self._lock:
                        self._completed.add(key)
                        self._progress_counter += 1
                        current = self._progress_counter

                    if progress_callback:
                        progress_callback(current, self.total_calls, prediction)

                    # Rate limit
                    time.sleep(self.config.cli_delay_seconds)

    def _collect_api_batch(
        self,
        progress_callback: Callable[[int, int, Prediction], None] | None = None,
    ) -> None:
        """Collect predictions from API models (parallel).

        Predictions are added directly to self._predictions with periodic checkpointing.
        Groups tasks by provider and uses per-provider worker limits.
        """
        # Build pending tasks grouped by provider
        pending_by_provider: dict[str, list[tuple]] = {}
        with self._lock:
            for condition in self.api_conditions:
                for test_case in self.test_cases:
                    for bootstrap_run in range(1, self.config.n_bootstrap + 1):
                        key = (bootstrap_run, test_case.test_case_id, condition.model, condition.name)
                        if key not in self._completed:
                            # Determine provider from model name
                            if condition.model.startswith("gemini"):
                                provider = "gemini"
                            elif condition.model.startswith("claude"):
                                provider = "anthropic"
                            elif condition.model.startswith("gpt"):
                                provider = "openai"
                            else:
                                provider = "other"
                            pending_by_provider.setdefault(provider, []).append(
                                (test_case, condition, bootstrap_run)
                            )

        if not pending_by_provider:
            return

        # Process each provider with its own worker pool
        for pending_tasks in pending_by_provider.values():
            # Get worker count for this provider (use first model in group)
            sample_model = pending_tasks[0][1].model
            workers = self.config.get_parallel_workers(sample_model)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._make_single_api_call,
                        task[0],  # test_case
                        task[1],  # condition
                        task[2],  # bootstrap_run
                    ): task
                    for task in pending_tasks
                }

                for future in as_completed(futures):
                    task = futures[future]
                    test_case, condition, bootstrap_run = task

                    try:
                        prediction = future.result()

                        # Add to predictions with periodic checkpoint
                        self._add_prediction(prediction)

                        with self._lock:
                            self._completed.add(prediction.key)
                            self._progress_counter += 1
                            current = self._progress_counter

                        if progress_callback:
                            progress_callback(current, self.total_calls, prediction)

                    except Exception as e:
                        prediction = Prediction(
                            test_case_id=test_case.test_case_id,
                            model=condition.model,
                            condition=condition.name,
                            bootstrap_run=bootstrap_run,
                            raw_response="",
                            tokens_used=0,
                            timestamp=datetime.now(),
                            error=str(e),
                            run_id=self.config.run_id,
                        )
                        # Also checkpoint error predictions
                        self._add_prediction(prediction)

    def _make_single_api_call(
        self,
        test_case: TestCase,
        condition: ExperimentCondition,
        bootstrap_run: int,
    ) -> Prediction:
        """Make a single API call. Used for parallel execution."""
        client = create_client(condition.model, dry_run=self.config.dry_run)

        prompt = build_prompt(
            test_case,
            template_name=condition.prompt_strategy,
            context_level=condition.context_level,
            rag_mode=condition.rag_mode,
        )

        return self._make_call(test_case, condition, bootstrap_run, prompt, client)

    def _make_call(
        self,
        test_case: TestCase,
        condition: ExperimentCondition,
        bootstrap_run: int,
        prompt: str,
        client: Any,
    ) -> Prediction:
        """Make a single LLM call and return prediction."""
        try:
            response = client.call(prompt, max_tokens=self.config.max_tokens)

            return Prediction(
                test_case_id=test_case.test_case_id,
                model=condition.model,
                condition=condition.name,
                bootstrap_run=bootstrap_run,
                raw_response=response.content,
                tokens_used=response.tokens_used,
                timestamp=datetime.now(),
                prompt=prompt[:500],  # Store truncated prompt for debugging
                run_id=self.config.run_id,
            )

        except Exception as e:
            return Prediction(
                test_case_id=test_case.test_case_id,
                model=condition.model,
                condition=condition.name,
                bootstrap_run=bootstrap_run,
                raw_response="",
                tokens_used=0,
                timestamp=datetime.now(),
                error=str(e),
                run_id=self.config.run_id,
            )
