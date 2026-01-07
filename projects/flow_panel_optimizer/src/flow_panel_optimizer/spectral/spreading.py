"""Spillover Spreading Matrix (SSM) calculation.

The SSM quantifies the spread (increase in standard deviation) in one
detector caused by spillover from another fluorophore's signal.

Reference:
    Nguyen R, Perfetto S, Mahnke YD, Chattopadhyay P, Roederer M.
    "Quantifying spillover spreading for comparing instrument performance
    and aiding in multicolor panel design." Cytometry A. 2013;83(3):306-315.

Note: Actual SSM calculation requires single-stained controls and
instrument data. This module provides THEORETICAL estimates based on
spectral similarity and brightness metrics.
"""

import numpy as np


def estimate_spillover_from_similarity(
    similarity: float,
    power: float = 2.0,
) -> float:
    """Estimate spillover coefficient from spectral similarity.

    This is a rough approximation since actual spillover depends on
    instrument configuration, detector sensitivity, and filter sets.

    Args:
        similarity: Cosine similarity between fluorophores (0-1).
        power: Exponent for non-linear relationship. Higher values
            make the relationship more aggressive at high similarities.

    Returns:
        Estimated spillover as a fraction (0-1).
    """
    # Spillover is roughly proportional to similarity squared
    # This is a simplification - real spillover depends on
    # filter overlap, not just spectral shape
    return similarity ** power


def theoretical_spreading(
    similarity: float,
    spillover_coefficient: float,
    stain_index_primary: float,
    stain_index_secondary: float | None = None,
) -> float:
    """Calculate theoretical spillover spreading between two fluorophores.

    Based on Nguyen et al. (2013), spread is related to:
    1. Spillover coefficient (how much light bleeds into other detectors)
    2. Signal intensity (brighter signals cause more spread)
    3. Detector characteristics (instrument-specific)

    The actual SSM formula from the paper:
        ΔσC = √(σ²_C(pos) - σ²_C(neg))

    This theoretical version estimates spread without actual controls.

    Args:
        similarity: Cosine similarity between fluorophores (0-1).
        spillover_coefficient: Estimated spillover fraction (0-1).
        stain_index_primary: Brightness metric of primary fluorophore.
        stain_index_secondary: Optional brightness of receiving channel.

    Returns:
        Theoretical spread value (arbitrary units, higher = more spread).
    """
    if stain_index_secondary is None:
        stain_index_secondary = stain_index_primary

    # SSM is proportional to sqrt of spillover (Poisson statistics)
    # and scales with signal intensity
    spread = np.sqrt(spillover_coefficient * stain_index_primary)

    # Additional modulation based on receiving channel sensitivity
    # Brighter primary signals cause more spread in dimmer channels
    if stain_index_secondary > 0:
        relative_brightness = stain_index_primary / stain_index_secondary
        spread *= np.sqrt(relative_brightness)

    return spread


def build_spreading_matrix(
    similarity_matrix: np.ndarray,
    stain_indices: np.ndarray | list[float] | None = None,
    spillover_estimates: np.ndarray | None = None,
    default_stain_index: float = 100.0,
) -> np.ndarray:
    """Build theoretical NxN spreading matrix.

    IMPORTANT: This is an APPROXIMATION. Real SSM requires:
    1. Single-stained controls
    2. Actual instrument acquisition
    3. Compensation matrix calculation

    Args:
        similarity_matrix: NxN cosine similarity matrix.
        stain_indices: 1D array of stain index per fluorophore.
            If None, uses default value for all.
        spillover_estimates: Optional NxN spillover coefficient matrix.
            If None, estimates from similarity.
        default_stain_index: Default stain index if not provided.

    Returns:
        NxN spreading matrix (theoretical estimates).

    Example:
        >>> sim_matrix = np.array([
        ...     [1.0, 0.9, 0.3],
        ...     [0.9, 1.0, 0.4],
        ...     [0.3, 0.4, 1.0],
        ... ])
        >>> stain_indices = [200, 150, 100]
        >>> ssm = build_spreading_matrix(sim_matrix, stain_indices)
        >>> ssm.shape
        (3, 3)
    """
    n = similarity_matrix.shape[0]

    # Handle stain indices
    if stain_indices is None:
        stain_indices = np.full(n, default_stain_index)
    else:
        stain_indices = np.asarray(stain_indices, dtype=np.float64)

    if len(stain_indices) != n:
        raise ValueError(
            f"stain_indices length ({len(stain_indices)}) must match "
            f"matrix size ({n})"
        )

    # Estimate spillover from similarity if not provided
    if spillover_estimates is None:
        spillover_estimates = estimate_spillover_from_similarity(similarity_matrix)

    ssm = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                ssm[i, j] = 0.0  # No self-spreading
            else:
                ssm[i, j] = theoretical_spreading(
                    similarity=similarity_matrix[i, j],
                    spillover_coefficient=spillover_estimates[i, j],
                    stain_index_primary=stain_indices[i],
                    stain_index_secondary=stain_indices[j],
                )

    return ssm


def spreading_risk_level(spread_value: float) -> str:
    """Classify spread value into risk categories.

    These thresholds are approximate and may need calibration
    against real instrument data.

    Args:
        spread_value: Theoretical spread value.

    Returns:
        Risk level string.
    """
    if spread_value >= 20.0:
        return "critical"
    elif spread_value >= 10.0:
        return "high"
    elif spread_value >= 5.0:
        return "moderate"
    elif spread_value >= 2.0:
        return "low"
    else:
        return "minimal"


def find_high_spreading_pairs(
    spreading_matrix: np.ndarray,
    fluorophore_names: list[str],
    threshold: float = 10.0,
) -> list[tuple[str, str, float]]:
    """Find fluorophore pairs with high spreading.

    Args:
        spreading_matrix: NxN theoretical spreading matrix.
        fluorophore_names: Ordered list of fluorophore names.
        threshold: Minimum spread value to include.

    Returns:
        List of (fluor_a, fluor_b, spread) tuples, sorted by
        spread descending.
    """
    n = len(fluorophore_names)
    pairs = []

    for i in range(n):
        for j in range(n):
            if i != j:
                spread = spreading_matrix[i, j]
                if spread >= threshold:
                    pairs.append((
                        fluorophore_names[i],
                        fluorophore_names[j],
                        round(spread, 2),
                    ))

    # Sort by spread descending
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Remove duplicates (since spreading can be asymmetric, keep both directions)
    return pairs


def total_panel_spreading(spreading_matrix: np.ndarray) -> float:
    """Calculate total spreading for the panel.

    Args:
        spreading_matrix: NxN theoretical spreading matrix.

    Returns:
        Sum of all non-diagonal spreading values.
    """
    n = spreading_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.sum(spreading_matrix[mask]))


def max_spreading_per_fluorophore(
    spreading_matrix: np.ndarray,
    fluorophore_names: list[str],
) -> dict[str, float]:
    """Find maximum spreading impact for each fluorophore.

    For each fluorophore, find the maximum spread it causes
    in any other channel.

    Args:
        spreading_matrix: NxN theoretical spreading matrix.
        fluorophore_names: Ordered list of fluorophore names.

    Returns:
        Dict mapping fluorophore name -> max spread value.
    """
    result = {}
    n = len(fluorophore_names)

    for i in range(n):
        # Maximum spread this fluorophore causes
        max_spread = 0.0
        for j in range(n):
            if i != j and spreading_matrix[i, j] > max_spread:
                max_spread = spreading_matrix[i, j]
        result[fluorophore_names[i]] = round(max_spread, 2)

    return result
