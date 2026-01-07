"""Spectral metric calculations for flow cytometry panel design."""

from flow_panel_optimizer.spectral.similarity import (
    cosine_similarity,
    build_similarity_matrix,
    find_high_similarity_pairs,
)
from flow_panel_optimizer.spectral.complexity import (
    complexity_index,
    complexity_index_v2,
    pair_complexity_contribution,
)
from flow_panel_optimizer.spectral.spreading import (
    theoretical_spreading,
    build_spreading_matrix,
    estimate_spillover_from_similarity,
)

__all__ = [
    "cosine_similarity",
    "build_similarity_matrix",
    "find_high_similarity_pairs",
    "complexity_index",
    "complexity_index_v2",
    "pair_complexity_contribution",
    "theoretical_spreading",
    "build_spreading_matrix",
    "estimate_spillover_from_similarity",
]
