"""Evaluation and ablation framework for DrugDevBench."""

from drugdevbench.evaluation.rubric import (
    score_response,
    ScoringResult,
    SCORING_RUBRIC,
)
from drugdevbench.evaluation.metrics import (
    compute_benchmark_metrics,
    compute_ablation_metrics,
)
from drugdevbench.evaluation.ablation import (
    AblationRunner,
    AblationConfig,
    run_ablation_study,
)
from drugdevbench.evaluation.reporting import (
    create_ablation_report,
    generate_quick_summary,
)

__all__ = [
    # Scoring
    "score_response",
    "ScoringResult",
    "SCORING_RUBRIC",
    # Metrics
    "compute_benchmark_metrics",
    "compute_ablation_metrics",
    # Ablation
    "AblationRunner",
    "AblationConfig",
    "run_ablation_study",
    # Reporting
    "create_ablation_report",
    "generate_quick_summary",
]
