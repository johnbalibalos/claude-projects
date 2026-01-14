"""
SQLite experiment tracker for flow gating benchmark.

Provides structured storage and querying for experiment results,
replacing ad-hoc JSON files with a relational database.

Usage:
    tracker = ExperimentTracker()
    run_id = tracker.start_run(config)

    # During prediction collection
    tracker.log_prediction(run_id, prediction)

    # After scoring
    tracker.log_scores(prediction_id, scores)

    # Analysis
    df = tracker.query_to_dataframe("SELECT * FROM predictions WHERE model LIKE '%sonnet%'")
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from .prediction_collector import Prediction
    from .batch_scorer import ScoringResult


SCHEMA = """
-- Experiment runs (one per pipeline invocation)
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    config_json TEXT,
    git_commit TEXT,
    notes TEXT
);

-- Individual predictions (one row per model × condition × test_case × bootstrap)
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),

    -- Input dimensions
    test_case_id TEXT NOT NULL,
    model TEXT NOT NULL,
    context_level TEXT NOT NULL,
    prompt_strategy TEXT NOT NULL,
    reference_mode TEXT DEFAULT 'none',
    bootstrap_run INTEGER DEFAULT 1,

    -- Response
    raw_response TEXT,
    tokens_used INTEGER,
    latency_ms INTEGER,
    timestamp TEXT NOT NULL,

    -- Parse outcome
    parse_success INTEGER,
    parse_error TEXT,

    -- Scores (populated by BatchScorer)
    hierarchy_f1 REAL,
    hierarchy_precision REAL,
    hierarchy_recall REAL,
    structure_accuracy REAL,
    critical_gate_recall REAL,
    hallucination_rate REAL,

    -- Error tracking
    error_type TEXT,
    error_message TEXT,

    UNIQUE(run_id, test_case_id, model, context_level, prompt_strategy, reference_mode, bootstrap_run)
);

-- LLM judge results
CREATE TABLE IF NOT EXISTS judge_scores (
    id INTEGER PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    judge_model TEXT NOT NULL,
    judge_style TEXT NOT NULL,
    score REAL,
    rationale TEXT,
    error TEXT,
    timestamp TEXT,

    UNIQUE(prediction_id, judge_model, judge_style)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model);
CREATE INDEX IF NOT EXISTS idx_predictions_test_case ON predictions(test_case_id);
CREATE INDEX IF NOT EXISTS idx_predictions_condition ON predictions(model, context_level, prompt_strategy);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_judge_style ON judge_scores(judge_style);
CREATE INDEX IF NOT EXISTS idx_judge_prediction ON judge_scores(prediction_id);
"""


@dataclass
class TrackerConfig:
    """Configuration for experiment tracker."""

    db_path: Path = Path("experiments.db")
    store_raw_responses: bool = True  # Set False to save space (responses can be large)


class ExperimentTracker:
    """SQLite-backed experiment tracker.

    Thread-safe for concurrent writes from parallel API calls.
    Uses WAL mode for better concurrent read/write performance.
    """

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.db_path = self.config.db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database with schema."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Run management
    # -------------------------------------------------------------------------

    def start_run(
        self,
        run_id: str,
        config: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> str:
        """Start a new experiment run.

        Args:
            run_id: Unique identifier for this run
            config: Configuration dict to store for reproducibility
            notes: Optional notes about this run

        Returns:
            The run_id
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, started_at, config_json, git_commit, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now().isoformat(),
                    json.dumps(config) if config else None,
                    self._get_git_commit(),
                    notes,
                ),
            )
        return run_id

    def complete_run(self, run_id: str) -> None:
        """Mark a run as completed."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET completed_at = ? WHERE run_id = ?",
                (datetime.now().isoformat(), run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        """Get run metadata."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------------------
    # Prediction logging
    # -------------------------------------------------------------------------

    def log_prediction(
        self,
        run_id: str,
        prediction: Prediction,
        latency_ms: int | None = None,
    ) -> int:
        """Log a prediction to the database.

        Uses INSERT OR REPLACE for idempotent writes (safe for retries).

        Args:
            run_id: The run this prediction belongs to
            prediction: The Prediction object to log
            latency_ms: Optional latency measurement

        Returns:
            The prediction row ID
        """
        # Parse condition string to extract components
        # Format: "model_context_strategy_reference" e.g. "claude-sonnet-cli_minimal_direct_none"
        condition_parts = prediction.condition.split("_")
        if len(condition_parts) >= 4:
            # Handle model names with underscores by finding known suffixes
            context_level = None
            prompt_strategy = None
            reference_mode = "none"

            for i, part in enumerate(condition_parts):
                if part in ("minimal", "standard", "rich"):
                    context_level = part
                    if i + 1 < len(condition_parts):
                        prompt_strategy = condition_parts[i + 1]
                    if i + 2 < len(condition_parts):
                        reference_mode = condition_parts[i + 2]
                    break

            if not context_level:
                # Fallback: assume last 3 parts are context_strategy_reference
                context_level = condition_parts[-3] if len(condition_parts) >= 3 else "standard"
                prompt_strategy = condition_parts[-2] if len(condition_parts) >= 2 else "direct"
                reference_mode = condition_parts[-1] if len(condition_parts) >= 1 else "none"
        else:
            context_level = "standard"
            prompt_strategy = "direct"
            reference_mode = "none"

        raw_response = prediction.raw_response if self.config.store_raw_responses else None

        # Classify error type
        error_type = None
        error_message = prediction.error
        if prediction.error:
            error_lower = prediction.error.lower()
            if "rate limit" in error_lower or "429" in error_lower:
                error_type = "rate_limit"
            elif "token" in error_lower and ("limit" in error_lower or "max" in error_lower):
                error_type = "token_limit"
            elif "timeout" in error_lower:
                error_type = "timeout"
            else:
                error_type = "api_error"

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    run_id, test_case_id, model, context_level, prompt_strategy,
                    reference_mode, bootstrap_run, raw_response, tokens_used,
                    latency_ms, timestamp, error_type, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    prediction.test_case_id,
                    prediction.model,
                    context_level,
                    prompt_strategy,
                    reference_mode,
                    prediction.bootstrap_run,
                    raw_response,
                    prediction.tokens_used,
                    latency_ms,
                    prediction.timestamp.isoformat() if isinstance(prediction.timestamp, datetime) else prediction.timestamp,
                    error_type,
                    error_message,
                ),
            )
            return cursor.lastrowid

    def get_completed_keys(self, run_id: str) -> set[tuple]:
        """Get set of completed prediction keys for resume logic.

        Returns:
            Set of (bootstrap_run, test_case_id, model, condition) tuples
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT bootstrap_run, test_case_id, model,
                       context_level || '_' || prompt_strategy || '_' || reference_mode as condition
                FROM predictions
                WHERE run_id = ? AND raw_response IS NOT NULL AND error_message IS NULL
                """,
                (run_id,),
            ).fetchall()

            return {
                (row["bootstrap_run"], row["test_case_id"], row["model"], row["condition"])
                for row in rows
            }

    def prediction_exists(
        self,
        run_id: str,
        test_case_id: str,
        model: str,
        context_level: str,
        prompt_strategy: str,
        bootstrap_run: int,
        reference_mode: str = "none",
    ) -> int | None:
        """Check if a prediction exists (for caching).

        Returns:
            Prediction ID if exists, None otherwise
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM predictions
                WHERE run_id = ? AND test_case_id = ? AND model = ?
                  AND context_level = ? AND prompt_strategy = ?
                  AND bootstrap_run = ? AND reference_mode = ?
                  AND raw_response IS NOT NULL
                """,
                (run_id, test_case_id, model, context_level, prompt_strategy, bootstrap_run, reference_mode),
            ).fetchone()
            return row["id"] if row else None

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def log_scores(
        self,
        prediction_id: int,
        result: ScoringResult,
    ) -> None:
        """Update prediction with scoring results.

        Args:
            prediction_id: The prediction row ID
            result: ScoringResult from BatchScorer
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE predictions SET
                    parse_success = ?,
                    parse_error = ?,
                    hierarchy_f1 = ?,
                    structure_accuracy = ?,
                    critical_gate_recall = ?,
                    hallucination_rate = ?
                WHERE id = ?
                """,
                (
                    1 if result.parse_success else 0,
                    result.error if not result.parse_success else None,
                    result.hierarchy_f1,
                    result.structure_accuracy,
                    result.critical_gate_recall,
                    result.hallucination_rate,
                    prediction_id,
                ),
            )

    def log_scores_by_key(
        self,
        run_id: str,
        result: ScoringResult,
    ) -> None:
        """Update prediction scores by matching key fields.

        Useful when you don't have the prediction_id handy.
        """
        # Parse condition to get components
        condition_parts = result.condition.split("_")
        context_level = "standard"
        prompt_strategy = "direct"
        reference_mode = "none"

        for i, part in enumerate(condition_parts):
            if part in ("minimal", "standard", "rich"):
                context_level = part
                if i + 1 < len(condition_parts):
                    prompt_strategy = condition_parts[i + 1]
                if i + 2 < len(condition_parts):
                    reference_mode = condition_parts[i + 2]
                break

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE predictions SET
                    parse_success = ?,
                    parse_error = ?,
                    hierarchy_f1 = ?,
                    structure_accuracy = ?,
                    critical_gate_recall = ?,
                    hallucination_rate = ?
                WHERE run_id = ? AND test_case_id = ? AND model = ?
                  AND context_level = ? AND prompt_strategy = ?
                  AND bootstrap_run = ? AND reference_mode = ?
                """,
                (
                    1 if result.parse_success else 0,
                    result.error if not result.parse_success else None,
                    result.hierarchy_f1,
                    result.structure_accuracy,
                    result.critical_gate_recall,
                    result.hallucination_rate,
                    run_id,
                    result.test_case_id,
                    result.model,
                    context_level,
                    prompt_strategy,
                    result.bootstrap_run,
                    reference_mode,
                ),
            )

    # -------------------------------------------------------------------------
    # Judge scores
    # -------------------------------------------------------------------------

    def log_judge_score(
        self,
        prediction_id: int,
        judge_model: str,
        judge_style: str,
        score: float | None,
        rationale: str | None = None,
        error: str | None = None,
    ) -> None:
        """Log an LLM judge score for a prediction.

        Args:
            prediction_id: The prediction this score is for
            judge_model: Model used for judging (e.g., "gemini-2.5-pro")
            judge_style: Evaluation style (default, validation, qualitative, orthogonal, binary)
            score: The judge score (0-1)
            rationale: Optional reasoning from the judge
            error: Optional error if judging failed
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO judge_scores
                (prediction_id, judge_model, judge_style, score, rationale, error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    judge_model,
                    judge_style,
                    score,
                    rationale,
                    error,
                    datetime.now().isoformat(),
                ),
            )

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as list of dicts."""
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def query_to_dataframe(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a query and return results as pandas DataFrame."""
        import pandas as pd

        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    # -------------------------------------------------------------------------
    # Convenience queries
    # -------------------------------------------------------------------------

    def compare_models(
        self,
        model_a: str,
        model_b: str,
        metric: str = "hierarchy_f1",
        run_id: str | None = None,
    ) -> list[dict]:
        """Compare two models on each test case.

        Args:
            model_a: First model pattern (supports SQL LIKE)
            model_b: Second model pattern
            metric: Metric to compare (hierarchy_f1, structure_accuracy, etc.)
            run_id: Optional run to filter by

        Returns:
            List of dicts with test_case_id, model_a_score, model_b_score, delta
        """
        run_filter = "AND p1.run_id = ?" if run_id else ""
        params = (model_a, model_b) + ((run_id,) if run_id else ())

        sql = f"""
            SELECT
                p1.test_case_id,
                p1.{metric} as model_a_score,
                p2.{metric} as model_b_score,
                p1.{metric} - p2.{metric} as delta
            FROM predictions p1
            JOIN predictions p2
                ON p1.test_case_id = p2.test_case_id
                AND p1.context_level = p2.context_level
                AND p1.prompt_strategy = p2.prompt_strategy
                AND p1.bootstrap_run = p2.bootstrap_run
            WHERE p1.model LIKE ?
              AND p2.model LIKE ?
              AND p1.parse_success = 1
              AND p2.parse_success = 1
              {run_filter}
            ORDER BY delta DESC
        """
        return self.query(sql, params)

    def get_stats_by_model(self, run_id: str | None = None) -> list[dict]:
        """Get aggregate statistics by model."""
        run_filter = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()

        sql = f"""
            SELECT
                model,
                COUNT(*) as n_predictions,
                SUM(CASE WHEN parse_success = 1 THEN 1 ELSE 0 END) as n_parsed,
                AVG(CASE WHEN parse_success = 1 THEN hierarchy_f1 END) as mean_f1,
                AVG(CASE WHEN parse_success = 1 THEN structure_accuracy END) as mean_structure,
                AVG(CASE WHEN parse_success = 1 THEN critical_gate_recall END) as mean_critical,
                SUM(CASE WHEN error_type = 'token_limit' THEN 1 ELSE 0 END) as token_limit_errors,
                SUM(CASE WHEN error_type = 'rate_limit' THEN 1 ELSE 0 END) as rate_limit_errors
            FROM predictions
            {run_filter}
            GROUP BY model
            ORDER BY mean_f1 DESC
        """
        return self.query(sql, params)

    def get_cot_effect(self, run_id: str | None = None) -> list[dict]:
        """Compare CoT vs direct prompting by model."""
        run_filter = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()

        sql = f"""
            SELECT
                model,
                prompt_strategy,
                AVG(hierarchy_f1) as mean_f1,
                AVG(structure_accuracy) as mean_structure,
                COUNT(*) as n
            FROM predictions
            {run_filter}
            {"AND" if run_filter else "WHERE"} parse_success = 1
            GROUP BY model, prompt_strategy
            ORDER BY model, prompt_strategy
        """
        return self.query(sql, params)

    def get_f1_vs_judge_correlation(
        self,
        judge_style: str = "default",
        run_id: str | None = None,
    ) -> list[dict]:
        """Get F1 and judge scores for correlation analysis."""
        run_filter = "AND p.run_id = ?" if run_id else ""
        params = (judge_style,) + ((run_id,) if run_id else ())

        sql = f"""
            SELECT
                p.model,
                p.test_case_id,
                p.hierarchy_f1,
                j.score as judge_score
            FROM predictions p
            JOIN judge_scores j ON j.prediction_id = p.id
            WHERE j.judge_style = ?
              AND p.parse_success = 1
              AND j.score IS NOT NULL
              {run_filter}
        """
        return self.query(sql, params)

    def export_to_json(self, run_id: str, output_path: Path) -> None:
        """Export run results to JSON for sharing."""
        with self._connect() as conn:
            predictions = conn.execute(
                "SELECT * FROM predictions WHERE run_id = ?",
                (run_id,),
            ).fetchall()

            judge_scores = conn.execute(
                """
                SELECT j.* FROM judge_scores j
                JOIN predictions p ON j.prediction_id = p.id
                WHERE p.run_id = ?
                """,
                (run_id,),
            ).fetchall()

        output = {
            "run_id": run_id,
            "predictions": [dict(p) for p in predictions],
            "judge_scores": [dict(j) for j in judge_scores],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
