"""Consensus checker for comparing multiple spectral metrics.

This module compares the three primary metrics (cosine similarity,
complexity contribution, and theoretical spreading) to verify they
agree on risk assessment for fluorophore pairs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class RiskLevel(Enum):
    """Risk levels for fluorophore pair combinations."""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other):
        if not isinstance(other, RiskLevel):
            return NotImplemented
        order = [
            RiskLevel.MINIMAL,
            RiskLevel.LOW,
            RiskLevel.MODERATE,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]
        return order.index(self) < order.index(other)

    def __le__(self, other):
        return self == other or self < other


@dataclass
class ConsensusResult:
    """Result of consensus check between metrics for a fluorophore pair."""

    fluorophore_a: str
    fluorophore_b: str
    cosine_similarity: float
    complexity_contribution: float
    theoretical_spread: float
    similarity_risk: RiskLevel
    spread_risk: RiskLevel
    consensus_risk: RiskLevel
    metrics_agree: bool
    notes: str


def _similarity_to_risk(similarity: float, thresholds: dict) -> RiskLevel:
    """Convert cosine similarity to risk level."""
    if similarity >= thresholds.get("cosine_critical", 0.98):
        return RiskLevel.CRITICAL
    elif similarity >= thresholds.get("cosine_high", 0.95):
        return RiskLevel.HIGH
    elif similarity >= thresholds.get("cosine_moderate", 0.90):
        return RiskLevel.MODERATE
    elif similarity >= thresholds.get("cosine_low", 0.80):
        return RiskLevel.LOW
    else:
        return RiskLevel.MINIMAL


def _spread_to_risk(spread: float, thresholds: dict) -> RiskLevel:
    """Convert spread value to risk level."""
    if spread >= thresholds.get("spread_critical", 20.0):
        return RiskLevel.CRITICAL
    elif spread >= thresholds.get("spread_high", 10.0):
        return RiskLevel.HIGH
    elif spread >= thresholds.get("spread_moderate", 5.0):
        return RiskLevel.MODERATE
    elif spread >= thresholds.get("spread_low", 2.0):
        return RiskLevel.LOW
    else:
        return RiskLevel.MINIMAL


def check_consensus(
    name_a: str,
    name_b: str,
    cosine_sim: float,
    complexity_contrib: float,
    spread_value: float,
    thresholds: Optional[dict] = None,
) -> ConsensusResult:
    """Check if all three metrics agree on risk assessment.

    This function compares the risk levels assigned by different metrics
    to determine if they have consensus on the severity of interference
    between two fluorophores.

    Args:
        name_a: First fluorophore name.
        name_b: Second fluorophore name.
        cosine_sim: Cosine similarity value (0-1).
        complexity_contrib: Contribution to complexity index.
        spread_value: Theoretical spreading value.
        thresholds: Custom thresholds dict (optional). Keys:
            - cosine_critical: Default 0.98
            - cosine_high: Default 0.95
            - cosine_moderate: Default 0.90
            - cosine_low: Default 0.80
            - spread_critical: Default 20.0
            - spread_high: Default 10.0
            - spread_moderate: Default 5.0
            - spread_low: Default 2.0

    Returns:
        ConsensusResult with agreement status and risk assessment.

    Example:
        >>> result = check_consensus(
        ...     "FITC", "BB515",
        ...     cosine_sim=0.98,
        ...     complexity_contrib=0.08,
        ...     spread_value=15.0
        ... )
        >>> result.consensus_risk
        <RiskLevel.CRITICAL: 'critical'>
        >>> result.metrics_agree
        False  # similarity=CRITICAL, spread=HIGH
    """
    if thresholds is None:
        thresholds = {}

    # Get risk from each metric
    similarity_risk = _similarity_to_risk(cosine_sim, thresholds)
    spread_risk = _spread_to_risk(spread_value, thresholds)

    # Check agreement
    risks = [similarity_risk, spread_risk]
    metrics_agree = len(set(risks)) == 1

    # Consensus risk is the highest of individual assessments
    consensus_risk = max(risks)

    # Generate notes
    notes = []
    if not metrics_agree:
        notes.append(f"Metrics disagree: similarity={similarity_risk.value}, spread={spread_risk.value}")
    if cosine_sim >= 0.95:
        notes.append("Very high spectral similarity - likely unmixing issues")
    if spread_value >= 15.0:
        notes.append("High theoretical spread - may impact dim populations")
    if complexity_contrib >= 0.05:
        notes.append("Significant complexity contribution")

    return ConsensusResult(
        fluorophore_a=name_a,
        fluorophore_b=name_b,
        cosine_similarity=round(cosine_sim, 4),
        complexity_contribution=round(complexity_contrib, 4),
        theoretical_spread=round(spread_value, 2),
        similarity_risk=similarity_risk,
        spread_risk=spread_risk,
        consensus_risk=consensus_risk,
        metrics_agree=metrics_agree,
        notes="; ".join(notes) if notes else "OK",
    )


def validate_panel_consensus(
    panel_name: str,
    similarity_matrix: np.ndarray,
    spreading_matrix: np.ndarray,
    fluorophore_names: list[str],
    complexity_threshold: float = 0.90,
) -> dict:
    """Run consensus check on entire panel.

    Args:
        panel_name: Name of the panel being validated.
        similarity_matrix: NxN cosine similarity matrix.
        spreading_matrix: NxN theoretical spreading matrix.
        fluorophore_names: Ordered list of fluorophore names.
        complexity_threshold: Threshold for complexity calculation.

    Returns:
        Summary dict with keys:
            - panel_name: Name of panel
            - total_pairs: Total number of pairs checked
            - metrics_agree: Number of pairs where metrics agree
            - agreement_rate: Fraction of agreeing pairs (0-1)
            - by_risk_level: Count of pairs at each risk level
            - high_risk_pairs: List of HIGH/CRITICAL consensus pairs
            - critical_pairs: Number of CRITICAL pairs
            - all_results: List of all ConsensusResult objects
    """
    n = len(fluorophore_names)
    results = []

    for i in range(n):
        for j in range(i + 1, n):
            # Calculate complexity contribution for this pair
            sim = similarity_matrix[i, j]
            contrib = max(0, sim - complexity_threshold)

            # Average of spread in both directions
            spread = (spreading_matrix[i, j] + spreading_matrix[j, i]) / 2

            result = check_consensus(
                name_a=fluorophore_names[i],
                name_b=fluorophore_names[j],
                cosine_sim=sim,
                complexity_contrib=contrib,
                spread_value=spread,
            )
            results.append(result)

    # Calculate statistics
    total_pairs = len(results)
    agreeing = sum(1 for r in results if r.metrics_agree)

    # Count by risk level
    by_risk = {level: 0 for level in RiskLevel}
    for r in results:
        by_risk[r.consensus_risk] += 1

    # Get high-risk pairs
    high_risk_pairs = [
        {
            "fluor_a": r.fluorophore_a,
            "fluor_b": r.fluorophore_b,
            "risk": r.consensus_risk.value,
            "similarity": r.cosine_similarity,
            "notes": r.notes,
        }
        for r in results
        if r.consensus_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    ]

    return {
        "panel_name": panel_name,
        "total_pairs": total_pairs,
        "metrics_agree": agreeing,
        "agreement_rate": round(agreeing / total_pairs, 3) if total_pairs > 0 else 0,
        "by_risk_level": {k.value: v for k, v in by_risk.items()},
        "high_risk_pairs": high_risk_pairs,
        "critical_pairs": by_risk[RiskLevel.CRITICAL],
        "all_results": results,
    }


def summarize_consensus(validation_result: dict) -> str:
    """Generate human-readable summary of consensus validation.

    Args:
        validation_result: Output from validate_panel_consensus().

    Returns:
        Formatted summary string.
    """
    lines = [
        f"Panel: {validation_result['panel_name']}",
        f"Total pairs analyzed: {validation_result['total_pairs']}",
        f"Metrics agreement rate: {validation_result['agreement_rate']:.1%}",
        "",
        "Risk distribution:",
    ]

    for level in RiskLevel:
        count = validation_result["by_risk_level"].get(level.value, 0)
        pct = count / validation_result["total_pairs"] * 100 if validation_result["total_pairs"] > 0 else 0
        lines.append(f"  {level.value.capitalize():10} {count:4} ({pct:.1f}%)")

    if validation_result["high_risk_pairs"]:
        lines.append("")
        lines.append(f"High/Critical risk pairs ({len(validation_result['high_risk_pairs'])}):")
        for pair in validation_result["high_risk_pairs"][:10]:  # Show top 10
            lines.append(
                f"  {pair['fluor_a']:15} - {pair['fluor_b']:15} "
                f"[{pair['risk'].upper()}] SI={pair['similarity']:.3f}"
            )
        if len(validation_result["high_risk_pairs"]) > 10:
            lines.append(f"  ... and {len(validation_result['high_risk_pairs']) - 10} more")

    return "\n".join(lines)
