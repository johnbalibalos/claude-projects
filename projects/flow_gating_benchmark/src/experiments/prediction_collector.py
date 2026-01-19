"""
Simplified prediction collection from LLMs.

Supports three execution modes:
- Sequential: CLI models (rate-limited)
- Parallel: API models via ThreadPoolExecutor
- Batch: Anthropic batch API (50% cheaper, async)

Usage:
    predictions = collect_predictions(tasks, config, checkpoint_path)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    run_id: str = ""

    @property
    def key(self) -> tuple:
        """Unique key for deduplication."""
        return (self.bootstrap_run, self.test_case_id, self.model, self.condition)


@dataclass
class CollectorConfig:
    """Configuration for prediction collection."""

    n_bootstrap: int = 1
    max_tokens: int = 6000
    workers: int = 50
    cli_delay: float = 0.5
    checkpoint_dir: Path | None = None
    dry_run: bool = False
    run_id: str = ""
    use_batch: bool = False  # Use Anthropic batch API when available


# Type alias for task tuple
Task = tuple[TestCase, ExperimentCondition, int]  # (test_case, condition, bootstrap_run)


def make_key(task: Task) -> tuple:
    """Create unique key from task tuple."""
    tc, cond, run = task
    return (run, tc.test_case_id, cond.model, cond.name)


def load_checkpoint(path: Path) -> set[tuple]:
    """Load completed keys from JSONL checkpoint."""
    if not path.exists():
        return set()

    completed = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Only count successful predictions (no error)
                if not data.get("error"):
                    key = (
                        data["bootstrap_run"],
                        data["test_case_id"],
                        data["model"],
                        data["condition"],
                    )
                    completed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_checkpoint(path: Path, prediction: Prediction) -> None:
    """Append prediction to JSONL checkpoint (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(prediction.to_dict()) + "\n")


def predict_one(
    test_case: TestCase,
    condition: ExperimentCondition,
    bootstrap_run: int,
    config: CollectorConfig,
) -> Prediction:
    """Make a single LLM prediction."""
    client = create_client(condition.model, dry_run=config.dry_run)

    prompt = build_prompt(
        test_case,
        template_name=condition.prompt_strategy,
        context_level=condition.context_level,
        reference=condition.reference,
    )

    try:
        response = client.call(prompt, max_tokens=config.max_tokens)
        return Prediction(
            test_case_id=test_case.test_case_id,
            model=condition.model,
            condition=condition.name,
            bootstrap_run=bootstrap_run,
            raw_response=response.content,
            tokens_used=response.tokens_used,
            timestamp=datetime.now(),
            prompt=prompt[:500],
            run_id=config.run_id,
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
            run_id=config.run_id,
        )


def collect_sequential(
    tasks: list[Task],
    config: CollectorConfig,
    checkpoint_path: Path,
    progress: Callable[[Prediction], None] | None = None,
) -> list[Prediction]:
    """Collect predictions sequentially (for CLI models)."""
    predictions = []
    for tc, cond, run in tasks:
        pred = predict_one(tc, cond, run, config)
        append_checkpoint(checkpoint_path, pred)
        predictions.append(pred)
        if progress:
            progress(pred)
        time.sleep(config.cli_delay)
    return predictions


def collect_parallel(
    tasks: list[Task],
    config: CollectorConfig,
    checkpoint_path: Path,
    progress: Callable[[Prediction], None] | None = None,
) -> list[Prediction]:
    """Collect predictions in parallel (for API models)."""
    predictions = []

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {
            pool.submit(predict_one, tc, cond, run, config): (tc, cond, run)
            for tc, cond, run in tasks
        }

        for future in as_completed(futures):
            pred = future.result()
            append_checkpoint(checkpoint_path, pred)
            predictions.append(pred)
            if progress:
                progress(pred)

    return predictions


def collect_batch(
    tasks: list[Task],
    config: CollectorConfig,
    checkpoint_path: Path,
    progress: Callable[[Prediction], None] | None = None,
) -> list[Prediction]:
    """Collect predictions via Anthropic batch API (50% cheaper).

    Batch API is async - we submit all requests, then poll for results.
    See: https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
    """
    try:
        import anthropic
    except ImportError:
        print("Warning: anthropic package not installed, falling back to parallel")
        return collect_parallel(tasks, config, checkpoint_path, progress)

    client = anthropic.Anthropic()
    predictions = []

    # Build batch requests
    requests = []
    task_map = {}  # custom_id -> task

    for tc, cond, run in tasks:
        prompt = build_prompt(
            tc,
            template_name=cond.prompt_strategy,
            context_level=cond.context_level,
            reference=cond.reference,
        )

        custom_id = f"{tc.test_case_id}_{cond.name}_{run}"
        task_map[custom_id] = (tc, cond, run, prompt)

        # Resolve model name (remove -cli suffix if present)
        model = cond.model.replace("-cli", "")
        if model.startswith("claude-") and not model.startswith("claude-3"):
            # Map short names to full model IDs
            model_map = {
                "claude-sonnet": "claude-sonnet-4-20250514",
                "claude-opus": "claude-opus-4-20250514",
            }
            model = model_map.get(model, model)

        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    if not requests:
        return predictions

    # Submit batch
    print(f"Submitting batch of {len(requests)} requests...")
    try:
        batch = client.batches.create(requests=requests)
        batch_id = batch.id
        print(f"Batch ID: {batch_id}")
    except Exception as e:
        print(f"Batch submission failed: {e}, falling back to parallel")
        return collect_parallel(tasks, config, checkpoint_path, progress)

    # Poll for completion
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.processing_status

        if status == "ended":
            break

        completed = batch.request_counts.succeeded + batch.request_counts.errored
        total = batch.request_counts.processing + completed
        print(f"Batch status: {status} ({completed}/{total})")
        time.sleep(10)

    # Retrieve results
    print("Retrieving batch results...")
    for result in client.batches.results(batch_id):
        custom_id = result.custom_id
        tc, cond, run, prompt = task_map[custom_id]

        if result.result.type == "succeeded":
            response = result.result.message
            pred = Prediction(
                test_case_id=tc.test_case_id,
                model=cond.model,
                condition=cond.name,
                bootstrap_run=run,
                raw_response=response.content[0].text,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                timestamp=datetime.now(),
                prompt=prompt[:500],
                run_id=config.run_id,
            )
        else:
            pred = Prediction(
                test_case_id=tc.test_case_id,
                model=cond.model,
                condition=cond.name,
                bootstrap_run=run,
                raw_response="",
                tokens_used=0,
                timestamp=datetime.now(),
                error=str(result.result.error),
                run_id=config.run_id,
            )

        append_checkpoint(checkpoint_path, pred)
        predictions.append(pred)
        if progress:
            progress(pred)

    return predictions


def collect_predictions(
    test_cases: list[TestCase],
    conditions: list[ExperimentCondition],
    config: CollectorConfig,
    resume: bool = False,
    progress_callback: Callable[[int, int, Prediction], None] | None = None,
) -> list[Prediction]:
    """Collect predictions with automatic execution mode selection.

    Args:
        test_cases: Test cases to run
        conditions: Experimental conditions
        config: Collection configuration
        resume: Whether to resume from checkpoint
        progress_callback: Optional callback(current, total, prediction)

    Returns:
        List of all predictions
    """
    # Build all tasks
    tasks: list[Task] = [
        (tc, cond, run)
        for tc in test_cases
        for cond in conditions
        for run in range(1, config.n_bootstrap + 1)
    ]

    # Setup checkpoint
    checkpoint_path = (
        config.checkpoint_dir / "predictions.jsonl"
        if config.checkpoint_dir
        else Path("/tmp/predictions.jsonl")
    )

    # Load completed if resuming
    completed = load_checkpoint(checkpoint_path) if resume else set()
    pending = [t for t in tasks if make_key(t) not in completed]

    total = len(tasks)
    done = len(completed)

    if resume and completed:
        print(f"Resuming: {done}/{total} already completed")

    # Group by execution mode
    cli_tasks = []
    batch_tasks = []
    parallel_tasks = []

    for t in pending:
        model = t[1].model
        if model.endswith("-cli"):
            cli_tasks.append(t)
        elif config.use_batch and model.startswith("claude"):
            batch_tasks.append(t)
        else:
            parallel_tasks.append(t)

    predictions = []

    # Progress wrapper
    def on_prediction(pred: Prediction):
        nonlocal done
        done += 1
        if progress_callback:
            progress_callback(done, total, pred)

    # Execute each mode
    if cli_tasks:
        print(f"Processing {len(cli_tasks)} CLI tasks (sequential)...")
        predictions.extend(
            collect_sequential(cli_tasks, config, checkpoint_path, on_prediction)
        )

    if batch_tasks:
        print(f"Processing {len(batch_tasks)} batch tasks (Anthropic batch API)...")
        predictions.extend(
            collect_batch(batch_tasks, config, checkpoint_path, on_prediction)
        )

    if parallel_tasks:
        print(f"Processing {len(parallel_tasks)} API tasks ({config.workers} workers)...")
        predictions.extend(
            collect_parallel(parallel_tasks, config, checkpoint_path, on_prediction)
        )

    # Save final JSON (in addition to JSONL checkpoint)
    if config.checkpoint_dir:
        output_path = config.checkpoint_dir.parent / "predictions.json"
        all_predictions = list(completed) if resume else []
        # Reload all from checkpoint for complete output
        all_preds = []
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            all_preds.append(Prediction.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError):
                            continue
        with open(output_path, "w") as f:
            json.dump([p.to_dict() for p in all_preds], f, indent=2)

    return predictions


# Legacy compatibility - keep class interface for existing code
class PredictionCollector:
    """Legacy wrapper around collect_predictions function."""

    def __init__(
        self,
        test_cases: list[TestCase],
        conditions: list[ExperimentCondition],
        config: CollectorConfig | None = None,
    ):
        self.test_cases = test_cases
        self.conditions = conditions
        self.config = config or CollectorConfig()

    @property
    def total_calls(self) -> int:
        return len(self.test_cases) * len(self.conditions) * self.config.n_bootstrap

    def collect(
        self,
        resume: bool = False,
        progress_callback: Callable[[int, int, Prediction], None] | None = None,
    ) -> list[Prediction]:
        return collect_predictions(
            self.test_cases,
            self.conditions,
            self.config,
            resume=resume,
            progress_callback=progress_callback,
        )
