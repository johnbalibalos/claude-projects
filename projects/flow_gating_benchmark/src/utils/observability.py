"""
Observability utilities for long-running experiment pipelines.

Provides:
- Structured logging with error surfacing
- Progress tracking with ETA
- Fail-fast error thresholds
- Real-time error summaries
- Slack/webhook notifications (optional)

Usage:
    from utils.observability import PipelineMonitor, setup_logging

    # At start of script
    setup_logging(verbose=args.verbose)

    # Wrap your pipeline
    monitor = PipelineMonitor(
        total_tasks=len(tasks),
        fail_threshold=0.1,  # Stop if >10% fail
        alert_on_errors=["rate_limit", "token_limit"],
    )

    for task in tasks:
        with monitor.track(task.id):
            result = run_task(task)
            if result.error:
                monitor.record_error(result.error, category="api")

    monitor.print_summary()
"""

from __future__ import annotations

import atexit
import logging
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

# Optional rich library for better terminal output
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels with semantic meaning for experiments."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"  # Pipeline-stopping errors


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
    use_rich: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Configure logging with sensible defaults for experiment pipelines.

    Features:
    - Rich console output with colors (if available)
    - Separate file logging for full debug output
    - Timestamps and module names for traceability
    - Dry-run mode: minimal logging, no file output

    Args:
        verbose: If True, show DEBUG level; otherwise INFO
        log_file: Optional path to write full logs (always DEBUG level)
        use_rich: Use rich library for prettier output if available
        dry_run: If True, skip file logging and reduce output (fast mode)
    """
    level = logging.DEBUG if verbose else logging.INFO

    # In dry run mode, only show warnings and above (minimal output)
    if dry_run:
        level = logging.WARNING

    handlers: list[logging.Handler] = []

    # Console handler
    if use_rich and RICH_AVAILABLE and not dry_run:
        console_handler = RichHandler(
            level=level,
            show_time=True,
            show_path=verbose,
            markup=True,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
    handlers.append(console_handler)

    # File handler (skip in dry run mode - no disk I/O)
    if log_file and not dry_run:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        ))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture everything, handlers filter
        handlers=handlers,
        force=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


@dataclass
class ErrorRecord:
    """Record of an error that occurred during pipeline execution."""
    timestamp: datetime
    task_id: str
    category: str
    message: str
    traceback: str | None = None


@dataclass
class PipelineStats:
    """Statistics for pipeline monitoring."""
    started_at: datetime = field(default_factory=datetime.now)
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    errors_by_category: Counter = field(default_factory=Counter)
    task_durations: list[float] = field(default_factory=list)
    recent_errors: list[ErrorRecord] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return self.completed + self.failed + self.skipped

    @property
    def failure_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.failed / self.total_processed

    @property
    def avg_duration(self) -> float:
        if not self.task_durations:
            return 0.0
        return sum(self.task_durations) / len(self.task_durations)

    def eta(self, remaining: int) -> timedelta | None:
        if self.avg_duration == 0 or remaining == 0:
            return None
        return timedelta(seconds=self.avg_duration * remaining)


class FailThresholdExceeded(Exception):
    """Raised when too many tasks have failed."""
    def __init__(self, failure_rate: float, threshold: float, recent_errors: list[ErrorRecord]):
        self.failure_rate = failure_rate
        self.threshold = threshold
        self.recent_errors = recent_errors
        super().__init__(
            f"Failure rate {failure_rate:.1%} exceeds threshold {threshold:.1%}. "
            f"Recent errors: {[e.message[:50] for e in recent_errors[-3:]]}"
        )


class PipelineMonitor:
    """
    Monitor for long-running experiment pipelines with fail-fast and error surfacing.

    Features:
    - Progress bar with ETA
    - Automatic fail-fast when error threshold exceeded
    - Error categorization and summary
    - Real-time error logging that surfaces critical issues
    - Final summary report

    Example:
        monitor = PipelineMonitor(total_tasks=100, fail_threshold=0.1)

        for item in items:
            with monitor.track(item.id):
                try:
                    process(item)
                except RateLimitError as e:
                    monitor.record_error(str(e), category="rate_limit")
                    raise

        monitor.print_summary()
    """

    def __init__(
        self,
        total_tasks: int,
        fail_threshold: float = 0.2,
        min_samples_for_threshold: int = 10,
        alert_on_errors: list[str] | None = None,
        alert_callback: Callable[[ErrorRecord], None] | None = None,
        show_progress: bool = True,
        dry_run: bool = False,
    ):
        """
        Initialize pipeline monitor.

        Args:
            total_tasks: Total number of tasks to process
            fail_threshold: Stop pipeline if failure rate exceeds this (0.0-1.0)
            min_samples_for_threshold: Minimum tasks before enforcing threshold
            alert_on_errors: Error categories that trigger immediate alerts
            alert_callback: Optional callback for alerts (e.g., Slack webhook)
            show_progress: Show progress bar
            dry_run: If True, minimal overhead mode (no progress bar, no alerts)
        """
        self.total_tasks = total_tasks
        self.fail_threshold = fail_threshold
        self.min_samples = min_samples_for_threshold
        self.alert_categories = set(alert_on_errors or [])
        self.alert_callback = alert_callback if not dry_run else None
        self.show_progress = show_progress and not dry_run
        self.dry_run = dry_run

        self.stats = PipelineStats()
        self._current_task: str | None = None
        self._task_start: float | None = None

        # Rich progress bar (disabled in dry run mode for speed)
        self._progress: Progress | None = None
        self._progress_task: Any = None

        if self.show_progress and RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=Console(stderr=True),
            )
            self._progress.start()
            self._progress_task = self._progress.add_task(
                "Processing", total=total_tasks
            )

        # Print summary on exit (skip in dry run - no persistent state)
        if not dry_run:
            atexit.register(self._atexit_summary)

    def _atexit_summary(self) -> None:
        """Print summary if pipeline exits unexpectedly."""
        if self._progress:
            self._progress.stop()
        if self.stats.total_processed > 0 and self.stats.failed > 0:
            self._print_error_summary()

    @contextmanager
    def track(self, task_id: str):
        """
        Context manager to track a single task.

        Usage:
            with monitor.track("task_123"):
                process_task()
        """
        self._current_task = task_id
        self._task_start = time.time()

        try:
            yield
            self._mark_complete()
        except Exception as e:
            self._mark_failed(str(e))
            raise
        finally:
            self._current_task = None
            self._task_start = None

    def _mark_complete(self) -> None:
        """Mark current task as complete."""
        self.stats.completed += 1
        if self._task_start:
            self.stats.task_durations.append(time.time() - self._task_start)
        if self._progress and self._progress_task is not None:
            self._progress.update(self._progress_task, advance=1)

    def _mark_failed(self, error_msg: str) -> None:
        """Mark current task as failed."""
        self.stats.failed += 1
        if self._task_start:
            self.stats.task_durations.append(time.time() - self._task_start)
        if self._progress and self._progress_task is not None:
            self._progress.update(self._progress_task, advance=1)

    def record_error(
        self,
        message: str,
        category: str = "unknown",
        traceback_str: str | None = None,
    ) -> None:
        """
        Record an error with categorization.

        Args:
            message: Error message
            category: Error category (e.g., "rate_limit", "parse", "api")
            traceback_str: Optional traceback string
        """
        # In dry run mode, just count errors (no logging overhead)
        self.stats.errors_by_category[category] += 1

        if self.dry_run:
            return

        error = ErrorRecord(
            timestamp=datetime.now(),
            task_id=self._current_task or "unknown",
            category=category,
            message=message,
            traceback=traceback_str,
        )

        self.stats.recent_errors.append(error)

        # Keep only last 100 errors in memory
        if len(self.stats.recent_errors) > 100:
            self.stats.recent_errors = self.stats.recent_errors[-100:]

        # Log the error
        logger.error(f"[{category}] {message[:200]}")

        # Alert callback for critical errors
        if category in self.alert_categories and self.alert_callback:
            self.alert_callback(error)

        # Check fail threshold
        self._check_threshold()

    def _check_threshold(self) -> None:
        """Check if failure threshold has been exceeded."""
        if self.stats.total_processed < self.min_samples:
            return

        if self.stats.failure_rate > self.fail_threshold:
            if self._progress:
                self._progress.stop()
            raise FailThresholdExceeded(
                self.stats.failure_rate,
                self.fail_threshold,
                self.stats.recent_errors,
            )

    def skip(self, task_id: str, reason: str = "") -> None:
        """Mark a task as skipped (e.g., already processed)."""
        self.stats.skipped += 1
        if self._progress and self._progress_task is not None:
            self._progress.update(self._progress_task, advance=1)
        logger.debug(f"Skipped {task_id}: {reason}")

    def _print_error_summary(self) -> None:
        """Print summary of errors."""
        if not self.stats.errors_by_category:
            return

        print("\n" + "=" * 60, file=sys.stderr)
        print("ERROR SUMMARY", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        if RICH_AVAILABLE:
            console = Console(stderr=True)
            table = Table(title="Errors by Category")
            table.add_column("Category", style="red")
            table.add_column("Count", justify="right")

            for category, count in self.stats.errors_by_category.most_common():
                table.add_row(category, str(count))

            console.print(table)

            # Show recent errors
            if self.stats.recent_errors:
                console.print("\n[bold]Recent Errors:[/bold]")
                for error in self.stats.recent_errors[-5:]:
                    console.print(f"  [{error.category}] {error.message[:80]}")
        else:
            for category, count in self.stats.errors_by_category.most_common():
                print(f"  {category}: {count}", file=sys.stderr)

    def print_summary(self) -> None:
        """Print final pipeline summary."""
        if self._progress:
            self._progress.stop()

        elapsed = datetime.now() - self.stats.started_at

        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Duration:    {elapsed}")
        print(f"Completed:   {self.stats.completed}")
        print(f"Failed:      {self.stats.failed}")
        print(f"Skipped:     {self.stats.skipped}")
        print(f"Failure Rate: {self.stats.failure_rate:.1%}")

        if self.stats.task_durations:
            print(f"Avg Duration: {self.stats.avg_duration:.2f}s per task")

        self._print_error_summary()


# =============================================================================
# Convenience decorators
# =============================================================================

def log_errors(category: str = "unknown"):
    """
    Decorator to log errors with categorization.

    Usage:
        @log_errors(category="api")
        def call_api():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[{category}] {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


def retry_with_logging(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retry with logging.

    Unlike tenacity, this logs each retry attempt for visibility.

    Usage:
        @retry_with_logging(max_attempts=3, exceptions=(RateLimitError,))
        def call_api():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            wait_time = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        wait_time *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# Slack/Webhook notifications (optional)
# =============================================================================

def create_slack_alerter(webhook_url: str) -> Callable[[ErrorRecord], None]:
    """
    Create a Slack webhook alerter for critical errors.

    Usage:
        alerter = create_slack_alerter("https://hooks.slack.com/...")
        monitor = PipelineMonitor(..., alert_callback=alerter)
    """
    def alert(error: ErrorRecord) -> None:
        try:
            import httpx
            httpx.post(
                webhook_url,
                json={
                    "text": f":warning: Pipeline Error [{error.category}]",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Task:* {error.task_id}\n*Error:* {error.message[:500]}"
                            }
                        }
                    ]
                },
                timeout=5.0,
            )
        except Exception as e:
            logger.warning(f"Failed to send Slack alert: {e}")

    return alert
