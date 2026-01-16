"""
Standardized output utilities for experiment scripts.

Replaces ad-hoc print() statements with structured output that:
- Respects --quiet and --verbose flags
- Works in both interactive and non-interactive environments
- Separates progress, results, and errors
- Supports machine-readable output (JSON)

Usage:
    from utils.output import Console, OutputFormat

    console = Console(
        verbose=args.verbose,
        quiet=args.quiet,
        format=OutputFormat.RICH if sys.stdout.isatty() else OutputFormat.PLAIN,
    )

    # Progress (only in verbose mode, or with progress bar)
    with console.progress("Processing", total=100) as progress:
        for item in items:
            process(item)
            progress.advance()

    # Status updates (hidden in quiet mode)
    console.status(f"Loaded {n} test cases")

    # Results (always shown unless --quiet)
    console.result("accuracy", 0.95)

    # Errors (always shown)
    console.error("Failed to connect to API")

    # Final summary (structured)
    console.summary({
        "total": 100,
        "passed": 95,
        "failed": 5,
    })
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Any, Iterator

# Optional rich library
try:
    from rich.console import Console as RichConsole
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


class OutputFormat(Enum):
    """Output format modes."""

    PLAIN = "plain"  # Simple text, good for logs/CI
    RICH = "rich"  # Rich terminal output with colors/progress bars
    JSON = "json"  # Machine-readable JSON output
    SILENT = "silent"  # No output (for testing)


@dataclass
class ProgressTracker:
    """Simple progress tracker for plain output."""

    description: str
    total: int
    current: int = 0
    _last_pct: int = -1

    def advance(self, n: int = 1) -> None:
        self.current += n
        pct = int(100 * self.current / self.total) if self.total > 0 else 0
        # Only print every 10%
        if pct // 10 > self._last_pct // 10:
            self._last_pct = pct
            print(f"  {self.description}: {pct}%", file=sys.stderr)

    def finish(self) -> None:
        pass


@dataclass
class Console:
    """
    Standardized console output for experiment scripts.

    Handles verbose/quiet modes, progress bars, and structured output.
    Automatically detects terminal vs non-terminal environments.
    """

    verbose: bool = False
    quiet: bool = False
    format: OutputFormat = field(default=OutputFormat.PLAIN)
    dry_run: bool = False

    # Internal state
    _results: dict = field(default_factory=dict)
    _errors: list = field(default_factory=list)
    _rich_console: Any = None

    def __post_init__(self):
        # Auto-detect format if not specified
        if self.format == OutputFormat.PLAIN and RICH_AVAILABLE:
            if sys.stdout.isatty() and not self.dry_run:
                self.format = OutputFormat.RICH

        # Dry run implies quiet for most output
        if self.dry_run:
            self.quiet = True

        # Initialize rich console if available
        if self.format == OutputFormat.RICH and RICH_AVAILABLE:
            self._rich_console = RichConsole(stderr=True)

    def _should_output(self, level: str) -> bool:
        """Check if output should be shown based on verbosity settings."""
        if self.format == OutputFormat.SILENT:
            return False
        if level == "error":
            return True  # Always show errors
        if self.quiet:
            return False
        if level == "debug" and not self.verbose:
            return False
        return True

    def debug(self, message: str) -> None:
        """Debug message (only in verbose mode)."""
        if self._should_output("debug"):
            print(f"[DEBUG] {message}", file=sys.stderr)

    def status(self, message: str) -> None:
        """Status update (hidden in quiet mode)."""
        if self._should_output("info"):
            if self._rich_console:
                self._rich_console.print(f"[dim]{message}[/dim]")
            else:
                print(message, file=sys.stderr)

    def info(self, message: str) -> None:
        """Info message (hidden in quiet mode)."""
        if self._should_output("info"):
            print(message)

    def error(self, message: str) -> None:
        """Error message (always shown)."""
        self._errors.append(message)
        if self._rich_console:
            self._rich_console.print(f"[red]ERROR:[/red] {message}")
        else:
            print(f"ERROR: {message}", file=sys.stderr)

    def warning(self, message: str) -> None:
        """Warning message (shown unless quiet)."""
        if self._should_output("info"):
            if self._rich_console:
                self._rich_console.print(f"[yellow]WARNING:[/yellow] {message}")
            else:
                print(f"WARNING: {message}", file=sys.stderr)

    def result(self, key: str, value: Any) -> None:
        """Record a result (for final summary)."""
        self._results[key] = value
        if self.format == OutputFormat.JSON:
            pass  # Will be output in summary()
        elif self._should_output("info"):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    @contextmanager
    def progress(self, description: str, total: int) -> Iterator[Any]:
        """
        Context manager for progress tracking.

        In rich mode: Shows animated progress bar
        In plain mode: Shows percentage every 10%
        In quiet/dry-run mode: No output
        """
        if self.quiet or self.format == OutputFormat.SILENT:
            # No-op progress tracker
            tracker = ProgressTracker(description, total)
            tracker.advance = lambda n=1: None  # type: ignore
            yield tracker
            return

        if self.format == OutputFormat.RICH and RICH_AVAILABLE:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=RichConsole(stderr=True),
            )
            with progress:
                task = progress.add_task(description, total=total)

                class RichTracker:
                    def advance(self, n: int = 1) -> None:
                        progress.update(task, advance=n)

                    def finish(self) -> None:
                        pass

                yield RichTracker()
        else:
            tracker = ProgressTracker(description, total)
            print(f"{description}:", file=sys.stderr)
            yield tracker
            print(f"  {description}: done", file=sys.stderr)

    def header(self, title: str) -> None:
        """Print a section header."""
        if self._should_output("info"):
            if self._rich_console:
                self._rich_console.rule(title)
            else:
                print(f"\n{'='*60}")
                print(title)
                print(f"{'='*60}")

    def table(self, headers: list[str], rows: list[list[Any]]) -> None:
        """Print a table of results."""
        if not self._should_output("info"):
            return

        if self._rich_console:
            table = Table()
            for h in headers:
                table.add_column(h)
            for row in rows:
                table.add_row(*[str(x) for x in row])
            self._rich_console.print(table)
        else:
            # Simple ASCII table
            widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]
            header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
            print(header_line)
            print("-" * len(header_line))
            for row in rows:
                print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))

    def summary(self, data: dict | None = None) -> None:
        """
        Print final summary.

        In JSON mode, outputs all collected results as JSON.
        In other modes, prints a formatted summary.
        """
        final_data = {**self._results, **(data or {})}

        if self._errors:
            final_data["errors"] = self._errors

        if self.format == OutputFormat.JSON:
            print(json.dumps(final_data, indent=2, default=str))
        elif self._should_output("info"):
            self.header("SUMMARY")
            for key, value in final_data.items():
                if key == "errors":
                    print(f"\nErrors ({len(value)}):")
                    for err in value[:5]:  # Show first 5
                        print(f"  - {err}")
                    if len(value) > 5:
                        print(f"  ... and {len(value) - 5} more")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def create_console(
    verbose: bool = False,
    quiet: bool = False,
    dry_run: bool = False,
    json_output: bool = False,
) -> Console:
    """
    Factory function to create a Console with common CLI flags.

    Args:
        verbose: Show debug output
        quiet: Suppress non-essential output
        dry_run: Suppress output and skip file I/O
        json_output: Output results as JSON (for scripting)

    Returns:
        Configured Console instance
    """
    if json_output:
        fmt = OutputFormat.JSON
    elif dry_run or quiet:
        fmt = OutputFormat.PLAIN
    elif sys.stdout.isatty() and RICH_AVAILABLE:
        fmt = OutputFormat.RICH
    else:
        fmt = OutputFormat.PLAIN

    return Console(
        verbose=verbose,
        quiet=quiet,
        format=fmt,
        dry_run=dry_run,
    )
