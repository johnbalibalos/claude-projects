"""
Cost-Performance Pareto analysis module.

Analyzes tradeoffs between cost/efficiency and performance:
- Pareto frontier computation
- Cost-efficiency metrics
- Model comparison with cost consideration
- Recommendations for cost-constrained scenarios
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ModelResult:
    """Result for a single model/condition."""

    name: str
    performance: float  # Primary metric (accuracy, F1, etc.)
    cost_per_call: float  # USD per API call
    latency_ms: float  # Average latency in milliseconds
    input_tokens: float  # Average input tokens
    output_tokens: float  # Average output tokens

    # Optional additional metrics
    additional_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def total_tokens(self) -> float:
        return self.input_tokens + self.output_tokens

    def efficiency(self, cost_weight: float = 0.5) -> float:
        """Compute efficiency score balancing performance and cost."""
        # Normalize cost (assume $0.01 per call is baseline)
        normalized_cost = min(1.0, self.cost_per_call / 0.01)
        return self.performance * (1 - cost_weight) + (1 - normalized_cost) * cost_weight


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""

    name: str
    performance: float
    cost: float
    is_pareto_optimal: bool
    dominates: list[str]  # Names of dominated points
    dominated_by: list[str]  # Names of dominating points


@dataclass
class ParetoAnalysis:
    """Complete Pareto analysis result."""

    points: list[ParetoPoint]
    pareto_frontier: list[str]  # Names of Pareto-optimal points
    dominated_points: list[str]
    best_performance: str  # Best regardless of cost
    best_value: str  # Best performance/cost ratio
    recommended_for_budget: dict[float, str]  # Budget -> recommended model

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PARETO ANALYSIS: Cost vs Performance",
            "=" * 60,
            "",
            f"Pareto-optimal solutions: {len(self.pareto_frontier)}",
            f"Dominated solutions: {len(self.dominated_points)}",
            "",
            "Pareto Frontier:",
        ]

        frontier_points = [p for p in self.points if p.is_pareto_optimal]
        frontier_points.sort(key=lambda x: x.cost)

        for p in frontier_points:
            lines.append(f"  • {p.name}: perf={p.performance:.3f}, cost=${p.cost:.4f}")

        lines.extend([
            "",
            f"Best Performance: {self.best_performance}",
            f"Best Value (perf/cost): {self.best_value}",
            "",
            "Budget Recommendations:",
        ])

        for budget, model in sorted(self.recommended_for_budget.items()):
            lines.append(f"  ${budget:.3f}/call: {model}")

        return "\n".join(lines)


# =============================================================================
# PARETO FRONTIER COMPUTATION
# =============================================================================


def compute_pareto_frontier(
    results: list[ModelResult],
    maximize_performance: bool = True,
    minimize_cost: bool = True,
) -> ParetoAnalysis:
    """
    Compute Pareto frontier for cost-performance tradeoff.

    A point is Pareto-optimal if no other point is better on all dimensions.

    Args:
        results: List of model results
        maximize_performance: Whether higher performance is better
        minimize_cost: Whether lower cost is better

    Returns:
        ParetoAnalysis with frontier and recommendations
    """
    n = len(results)
    if n == 0:
        return ParetoAnalysis(
            points=[],
            pareto_frontier=[],
            dominated_points=[],
            best_performance="",
            best_value="",
            recommended_for_budget={},
        )

    # Build points
    points = []
    for r in results:
        points.append(ParetoPoint(
            name=r.name,
            performance=r.performance,
            cost=r.cost_per_call,
            is_pareto_optimal=True,  # Will be updated
            dominates=[],
            dominated_by=[],
        ))

    # Determine dominance relationships
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                continue

            # Check if p1 dominates p2
            p1_better_perf = (p1.performance >= p2.performance) if maximize_performance else (p1.performance <= p2.performance)
            p1_better_cost = (p1.cost <= p2.cost) if minimize_cost else (p1.cost >= p2.cost)
            p1_strictly_better = (
                (p1.performance > p2.performance if maximize_performance else p1.performance < p2.performance) or
                (p1.cost < p2.cost if minimize_cost else p1.cost > p2.cost)
            )

            if p1_better_perf and p1_better_cost and p1_strictly_better:
                p1.dominates.append(p2.name)
                p2.dominated_by.append(p1.name)
                p2.is_pareto_optimal = False

    # Build frontier and dominated lists
    pareto_frontier = [p.name for p in points if p.is_pareto_optimal]
    dominated_points = [p.name for p in points if not p.is_pareto_optimal]

    # Find best performance
    if maximize_performance:
        best_perf_point = max(points, key=lambda p: p.performance)
    else:
        best_perf_point = min(points, key=lambda p: p.performance)

    # Find best value (performance per cost)
    def value_score(p: ParetoPoint) -> float:
        if p.cost == 0:
            return float('inf') if p.performance > 0 else 0
        return p.performance / p.cost

    best_value_point = max(points, key=value_score)

    # Budget recommendations
    budgets = [0.001, 0.003, 0.01, 0.03, 0.1]
    recommended_for_budget = {}

    for budget in budgets:
        # Find best performance within budget
        within_budget = [p for p in points if p.cost <= budget]
        if within_budget:
            if maximize_performance:
                best = max(within_budget, key=lambda p: p.performance)
            else:
                best = min(within_budget, key=lambda p: p.performance)
            recommended_for_budget[budget] = best.name

    return ParetoAnalysis(
        points=points,
        pareto_frontier=pareto_frontier,
        dominated_points=dominated_points,
        best_performance=best_perf_point.name,
        best_value=best_value_point.name,
        recommended_for_budget=recommended_for_budget,
    )


# =============================================================================
# MULTI-OBJECTIVE PARETO
# =============================================================================


@dataclass
class MultiObjectiveParetoResult:
    """Result of multi-objective Pareto analysis."""

    objectives: list[str]
    points: list[dict[str, Any]]
    pareto_frontier: list[str]
    hypervolume: float  # Multi-objective quality metric


def multi_objective_pareto(
    results: list[dict[str, Any]],
    objectives: list[str],
    maximize: list[bool] | None = None,
    name_field: str = "name",
) -> MultiObjectiveParetoResult:
    """
    Compute Pareto frontier for multiple objectives.

    Args:
        results: List of result dictionaries
        objectives: List of objective field names
        maximize: List of bools indicating maximize (True) or minimize (False)
        name_field: Field containing result name

    Returns:
        MultiObjectiveParetoResult with multi-objective Pareto frontier
    """
    if maximize is None:
        maximize = [True] * len(objectives)

    n = len(results)
    is_dominated = [False] * n

    # Check dominance
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            j_at_least_as_good = True
            j_strictly_better = False

            for obj, maxim in zip(objectives, maximize):
                vi = results[i].get(obj, 0)
                vj = results[j].get(obj, 0)

                if maxim:
                    if vj < vi:
                        j_at_least_as_good = False
                    if vj > vi:
                        j_strictly_better = True
                else:
                    if vj > vi:
                        j_at_least_as_good = False
                    if vj < vi:
                        j_strictly_better = True

            if j_at_least_as_good and j_strictly_better:
                is_dominated[i] = True
                break

    pareto_frontier = [
        results[i][name_field]
        for i in range(n)
        if not is_dominated[i]
    ]

    # Compute hypervolume (simplified 2D version)
    hypervolume = 0.0
    if len(objectives) == 2:
        frontier_points = [results[i] for i in range(n) if not is_dominated[i]]
        if frontier_points:
            # Sort by first objective
            frontier_points.sort(key=lambda x: x[objectives[0]], reverse=maximize[0])
            # Compute area under frontier
            ref_point = [0, 0]  # Reference point
            for i, p in enumerate(frontier_points):
                width = p[objectives[0]] - (frontier_points[i+1][objectives[0]] if i+1 < len(frontier_points) else ref_point[0])
                height = p[objectives[1]] - ref_point[1]
                hypervolume += abs(width * height)

    return MultiObjectiveParetoResult(
        objectives=objectives,
        points=results,
        pareto_frontier=pareto_frontier,
        hypervolume=hypervolume,
    )


# =============================================================================
# COST-EFFECTIVENESS METRICS
# =============================================================================


@dataclass
class CostEffectivenessMetrics:
    """Cost-effectiveness metrics for a model."""

    name: str
    performance: float
    cost: float
    latency: float

    # Computed metrics
    performance_per_dollar: float
    performance_per_ms: float
    cost_to_reach_baseline: float | None  # Cost to match baseline performance
    speedup_vs_baseline: float | None


def compute_cost_effectiveness(
    results: list[ModelResult],
    baseline_name: str | None = None,
) -> list[CostEffectivenessMetrics]:
    """
    Compute cost-effectiveness metrics for each model.

    Args:
        results: List of model results
        baseline_name: Name of baseline model for comparison

    Returns:
        List of CostEffectivenessMetrics for each model
    """
    # Find baseline
    baseline = None
    if baseline_name:
        for r in results:
            if r.name == baseline_name:
                baseline = r
                break

    metrics = []
    for r in results:
        # Basic metrics
        perf_per_dollar = r.performance / r.cost_per_call if r.cost_per_call > 0 else float('inf')
        perf_per_ms = r.performance / r.latency_ms if r.latency_ms > 0 else float('inf')

        # Comparison to baseline
        cost_to_baseline = None
        speedup = None

        if baseline:
            if r.performance >= baseline.performance:
                # How much cheaper to match baseline performance?
                cost_to_baseline = r.cost_per_call
            else:
                # Would need multiple calls to match baseline
                calls_needed = baseline.performance / r.performance if r.performance > 0 else float('inf')
                cost_to_baseline = r.cost_per_call * calls_needed

            speedup = baseline.latency_ms / r.latency_ms if r.latency_ms > 0 else float('inf')

        metrics.append(CostEffectivenessMetrics(
            name=r.name,
            performance=r.performance,
            cost=r.cost_per_call,
            latency=r.latency_ms,
            performance_per_dollar=perf_per_dollar,
            performance_per_ms=perf_per_ms,
            cost_to_reach_baseline=cost_to_baseline,
            speedup_vs_baseline=speedup,
        ))

    return metrics


# =============================================================================
# ROI ANALYSIS
# =============================================================================


@dataclass
class ROIAnalysis:
    """Return on Investment analysis for model upgrades."""

    from_model: str
    to_model: str
    performance_gain: float
    cost_increase: float
    cost_increase_percent: float
    roi: float  # Performance gain per dollar of additional cost
    break_even_n: int | None  # Number of calls for ROI to be positive


def compute_upgrade_roi(
    cheaper_model: ModelResult,
    expensive_model: ModelResult,
    value_per_performance_point: float = 1.0,
) -> ROIAnalysis:
    """
    Compute ROI of upgrading from cheaper to more expensive model.

    Args:
        cheaper_model: The less expensive model
        expensive_model: The more expensive model
        value_per_performance_point: Monetary value of 1 point of performance

    Returns:
        ROIAnalysis with ROI metrics
    """
    perf_gain = expensive_model.performance - cheaper_model.performance
    cost_increase = expensive_model.cost_per_call - cheaper_model.cost_per_call

    if cheaper_model.cost_per_call > 0:
        cost_increase_pct = cost_increase / cheaper_model.cost_per_call * 100
    else:
        cost_increase_pct = float('inf')

    if cost_increase > 0:
        roi = (perf_gain * value_per_performance_point) / cost_increase
    else:
        roi = float('inf') if perf_gain > 0 else 0

    # Break-even analysis
    # At what N calls does the expensive model provide better total value?
    # Value = Performance * value_per_point - Cost * N
    # Expensive better when: perf_exp * v - cost_exp * N > perf_cheap * v - cost_cheap * N
    # (perf_exp - perf_cheap) * v > (cost_exp - cost_cheap) * N
    # N < (perf_exp - perf_cheap) * v / (cost_exp - cost_cheap)
    if cost_increase > 0 and perf_gain > 0:
        break_even = int((perf_gain * value_per_performance_point) / cost_increase)
    else:
        break_even = None

    return ROIAnalysis(
        from_model=cheaper_model.name,
        to_model=expensive_model.name,
        performance_gain=perf_gain,
        cost_increase=cost_increase,
        cost_increase_percent=cost_increase_pct,
        roi=roi,
        break_even_n=break_even,
    )


# =============================================================================
# SCALING ANALYSIS
# =============================================================================


@dataclass
class ScalingAnalysis:
    """Analysis of how costs scale with volume."""

    model_name: str
    cost_per_call: float
    daily_cost_at_volume: dict[int, float]  # Volume -> daily cost
    monthly_cost_at_volume: dict[int, float]
    break_even_volume: int | None  # Volume where switching models makes sense


def analyze_cost_scaling(
    results: list[ModelResult],
    daily_volumes: list[int] = [100, 1000, 10000, 100000],
) -> list[ScalingAnalysis]:
    """
    Analyze how costs scale with API call volume.

    Args:
        results: List of model results
        daily_volumes: Volume levels to analyze

    Returns:
        List of ScalingAnalysis for each model
    """
    analyses = []

    for r in results:
        daily_costs = {vol: r.cost_per_call * vol for vol in daily_volumes}
        monthly_costs = {vol: r.cost_per_call * vol * 30 for vol in daily_volumes}

        analyses.append(ScalingAnalysis(
            model_name=r.name,
            cost_per_call=r.cost_per_call,
            daily_cost_at_volume=daily_costs,
            monthly_cost_at_volume=monthly_costs,
            break_even_volume=None,  # Would need to compare to specific alternative
        ))

    return analyses


# =============================================================================
# COMPREHENSIVE REPORT
# =============================================================================


@dataclass
class CostPerformanceReport:
    """Comprehensive cost-performance analysis report."""

    pareto_analysis: ParetoAnalysis
    cost_effectiveness: list[CostEffectivenessMetrics]
    scaling_analysis: list[ScalingAnalysis]
    roi_comparisons: list[ROIAnalysis]
    recommendations: list[str]


def generate_cost_performance_report(
    results: list[ModelResult],
    baseline_name: str | None = None,
    budget_constraint: float | None = None,
) -> CostPerformanceReport:
    """
    Generate comprehensive cost-performance report.

    Args:
        results: List of model results
        baseline_name: Name of baseline model
        budget_constraint: Maximum cost per call

    Returns:
        CostPerformanceReport with full analysis
    """
    # Pareto analysis
    pareto = compute_pareto_frontier(results)

    # Cost effectiveness
    effectiveness = compute_cost_effectiveness(results, baseline_name)

    # Scaling analysis
    scaling = analyze_cost_scaling(results)

    # ROI comparisons (compare adjacent models on Pareto frontier)
    roi_comparisons = []
    frontier_results = [r for r in results if r.name in pareto.pareto_frontier]
    frontier_results.sort(key=lambda r: r.cost_per_call)

    for i in range(len(frontier_results) - 1):
        roi = compute_upgrade_roi(frontier_results[i], frontier_results[i + 1])
        roi_comparisons.append(roi)

    # Generate recommendations
    recommendations = []

    # Best value recommendation
    recommendations.append(
        f"Best value (performance/cost): {pareto.best_value}"
    )

    # Budget-constrained recommendation
    if budget_constraint:
        within_budget = [r for r in results if r.cost_per_call <= budget_constraint]
        if within_budget:
            best_in_budget = max(within_budget, key=lambda r: r.performance)
            recommendations.append(
                f"Best within ${budget_constraint:.3f} budget: {best_in_budget.name} "
                f"(performance={best_in_budget.performance:.3f})"
            )
        else:
            recommendations.append(
                f"No models within ${budget_constraint:.3f} budget"
            )

    # High-volume recommendation
    if len(results) > 1:
        cheapest = min(results, key=lambda r: r.cost_per_call)
        recommendations.append(
            f"Lowest cost option: {cheapest.name} "
            f"(${cheapest.cost_per_call:.4f}/call)"
        )

    # Performance-critical recommendation
    best_perf = max(results, key=lambda r: r.performance)
    if best_perf.name != pareto.best_value:
        recommendations.append(
            f"Maximum performance: {best_perf.name} "
            f"(performance={best_perf.performance:.3f})"
        )

    return CostPerformanceReport(
        pareto_analysis=pareto,
        cost_effectiveness=effectiveness,
        scaling_analysis=scaling,
        roi_comparisons=roi_comparisons,
        recommendations=recommendations,
    )


def print_report(report: CostPerformanceReport) -> str:
    """Generate printable version of cost-performance report."""
    lines = [
        "=" * 70,
        "COST-PERFORMANCE ANALYSIS REPORT",
        "=" * 70,
        "",
        report.pareto_analysis.summary(),
        "",
        "Cost Effectiveness Metrics:",
        "| Model | Performance | Cost | Perf/$ |",
        "|-------|-------------|------|--------|",
    ]

    for m in sorted(report.cost_effectiveness, key=lambda x: -x.performance_per_dollar):
        lines.append(
            f"| {m.name[:20]:20} | {m.performance:.3f} | "
            f"${m.cost:.4f} | {m.performance_per_dollar:.1f} |"
        )

    lines.extend([
        "",
        "Scaling Analysis (Monthly Cost at Volume):",
        "| Model | 100/day | 1K/day | 10K/day | 100K/day |",
        "|-------|---------|--------|---------|----------|",
    ])

    for s in report.scaling_analysis:
        costs = [s.monthly_cost_at_volume.get(v, 0) for v in [100, 1000, 10000, 100000]]
        lines.append(
            f"| {s.model_name[:20]:20} | ${costs[0]:.0f} | ${costs[1]:.0f} | "
            f"${costs[2]:.0f} | ${costs[3]:.0f} |"
        )

    if report.roi_comparisons:
        lines.extend([
            "",
            "Upgrade ROI Analysis:",
        ])
        for roi in report.roi_comparisons:
            lines.append(
                f"  {roi.from_model} → {roi.to_model}: "
                f"ROI={roi.roi:.2f}, +{roi.performance_gain:.3f} perf, "
                f"+{roi.cost_increase_percent:.1f}% cost"
            )

    lines.extend([
        "",
        "Recommendations:",
    ])
    for rec in report.recommendations:
        lines.append(f"  → {rec}")

    return "\n".join(lines)
