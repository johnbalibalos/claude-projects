"""
Decoupled prediction collection from LLMs.

This module separates the concern of collecting raw LLM predictions
from scoring and judging. Enables modular pipeline architecture:

    PredictionCollector → BatchScorer → LLMJudge → ResultsAggregator
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from curation.schemas import TestCase
from utils.checkpoint import CheckpointManager

from .conditions import ExperimentCondition
from .llm_client import create_client
from .prompts import build_prompt


@dataclass
class Prediction:
    """Raw LLM prediction before scoring."""

    test_case_id: str
    model: str
    condition: str
    bootstrap_run: int
    raw_response: str
    tokens_used: int
    timestamp: datetime
    prompt: str = ""
    error: str | None = None

    @property
    def key(self) -> tuple:
        """Unique key for deduplication."""
        return (self.bootstrap_run, self.test_case_id, self.model, self.condition)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_case_id": self.test_case_id,
            "model": self.model,
            "condition": self.condition,
            "bootstrap_run": self.bootstrap_run,
            "raw_response": self.raw_response,
            "prompt": self.prompt,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prediction:
        """Create from dictionary."""
        return cls(
            test_case_id=data["test_case_id"],
            model=data["model"],
            condition=data["condition"],
            bootstrap_run=data["bootstrap_run"],
            raw_response=data["raw_response"],
            prompt=data.get("prompt", ""),
            tokens_used=data.get("tokens_used", 0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            error=data.get("error"),
        )


@dataclass
class CollectorConfig:
    """Configuration for prediction collection."""

    n_bootstrap: int = 3
    cli_delay_seconds: float = 2.0
    parallel_workers: int = 5
    checkpoint_dir: Path | None = None
    dry_run: bool = False


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

        # Checkpoint manager
        self._checkpoint = CheckpointManager(self.config.checkpoint_dir)

        # Track predictions
        self._predictions: list[Prediction] = []
        self._completed: set[tuple] = set()
        self._lock = threading.Lock()
        self._progress_counter = 0

    @property
    def total_calls(self) -> int:
        """Total number of calls to make."""
        return len(self.test_cases) * len(self.conditions) * self.config.n_bootstrap

    def load_checkpoint(self) -> list[Prediction]:
        """Load predictions from checkpoint files."""
        predictions, completed = self._checkpoint.load_with_keys(
            "predictions.json",
            Prediction,
            key_fn=lambda p: p.key,
        )

        self._completed = completed

        if predictions:
            print(f"Loaded {len(predictions)} predictions from checkpoint")

        return predictions

    def save_checkpoint(self, predictions: list[Prediction]) -> None:
        """Save predictions to checkpoint file."""
        self._checkpoint.save(predictions, "predictions.json")

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
        cli_predictions = []
        api_predictions = []

        if self.api_conditions:
            with ThreadPoolExecutor(max_workers=1) as bg_executor:
                # Start API in background
                api_future = bg_executor.submit(
                    self._collect_api_batch,
                    progress_callback,
                )

                # Run CLI in foreground
                if self.cli_conditions:
                    cli_predictions = self._collect_cli_batch(progress_callback)

                # Wait for API
                api_predictions = api_future.result()

        elif self.cli_conditions:
            cli_predictions = self._collect_cli_batch(progress_callback)

        # Merge all predictions
        all_predictions = self._predictions + cli_predictions + api_predictions

        # Save final checkpoint
        self.save_checkpoint(all_predictions)

        return all_predictions

    def _collect_cli_batch(
        self,
        progress_callback: Callable[[int, int, Prediction], None] | None = None,
    ) -> list[Prediction]:
        """Collect predictions from CLI models (sequential, rate-limited)."""
        predictions = []

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
                    predictions.append(prediction)

                    with self._lock:
                        self._completed.add(key)
                        self._progress_counter += 1
                        current = self._progress_counter

                    if progress_callback:
                        progress_callback(current, self.total_calls, prediction)

                    # Rate limit
                    time.sleep(self.config.cli_delay_seconds)

        return predictions

    def _collect_api_batch(
        self,
        progress_callback: Callable[[int, int, Prediction], None] | None = None,
    ) -> list[Prediction]:
        """Collect predictions from API models (parallel)."""
        predictions = []

        # Build pending tasks
        pending_tasks = []
        with self._lock:
            for condition in self.api_conditions:
                for test_case in self.test_cases:
                    for bootstrap_run in range(1, self.config.n_bootstrap + 1):
                        key = (bootstrap_run, test_case.test_case_id, condition.model, condition.name)
                        if key not in self._completed:
                            pending_tasks.append((test_case, condition, bootstrap_run))

        if not pending_tasks:
            return predictions

        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
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
                    predictions.append(prediction)

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
                    )
                    predictions.append(prediction)

        return predictions

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
            response = client.call(prompt)

            return Prediction(
                test_case_id=test_case.test_case_id,
                model=condition.model,
                condition=condition.name,
                bootstrap_run=bootstrap_run,
                raw_response=response.content,
                tokens_used=response.tokens_used,
                timestamp=datetime.now(),
                prompt=prompt[:500],  # Store truncated prompt for debugging
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
            )
