"""Complexity Index calculation for spectral flow cytometry panels.

The Complexity Index measures how difficult spectral unmixing will be for a panel.
It is based on the condition number of the spectral reference matrix, which
reflects the eigenvalue spread - higher values indicate more challenging unmixing.

References:
    - EasyPanel: https://flow-cytometry.net/rules-for-spectral-panel-design/
    - OMIP-069: Park LM et al. Cytometry A. 2020;97(10):1044-1051.
    - Nguyen et al. Cytometry A. 2013;83(3):306-315.
"""

import numpy as np
from typing import Optional


def complexity_index(
    spectral_matrix: np.ndarray,
    similarity_matrix: Optional[np.ndarray] = None,
) -> float:
    """Calculate complexity index using condition number of spectral reference matrix.

    The complexity index is defined as the condition number of the reference
    matrix used for spectral unmixing. Higher values indicate higher overall
    spread and more challenging unmixing.

    Args:
        spectral_matrix: NxM matrix where N is number of fluorophores and M is
            number of wavelength channels. Each row is a fluorophore's emission
            spectrum. If this is a square similarity matrix, it will be treated
            as such for backward compatibility.
        similarity_matrix: Optional NxN similarity matrix. If provided along with
            spectral_matrix, will use spectral_matrix for condition number.

    Returns:
        Complexity index (condition number). Lower is better.
        - < 10: Excellent unmixing expected
        - 10-30: Good unmixing
        - 30-50: Acceptable, may have some spread
        - > 50: Challenging, significant spread expected

    Example:
        >>> # Well-separated spectra (low complexity)
        >>> spectra = np.array([
        ...     [1.0, 0.1, 0.0, 0.0],  # Blue fluorophore
        ...     [0.0, 1.0, 0.1, 0.0],  # Green fluorophore
        ...     [0.0, 0.0, 1.0, 0.1],  # Red fluorophore
        ... ])
        >>> complexity_index(spectra)  # Low value

        >>> # Overlapping spectra (high complexity)
        >>> spectra_bad = np.array([
        ...     [1.0, 0.9, 0.5, 0.1],
        ...     [0.9, 1.0, 0.8, 0.3],
        ...     [0.5, 0.8, 1.0, 0.7],
        ... ])
        >>> complexity_index(spectra_bad)  # Higher value
    """
    if spectral_matrix.size == 0:
        return 0.0

    n = spectral_matrix.shape[0]
    if n < 2:
        return 0.0

    # Check if this is a square similarity matrix (backward compatibility)
    # A similarity matrix has 1s on diagonal and values 0-1 elsewhere
    is_similarity_matrix = (
        spectral_matrix.shape[0] == spectral_matrix.shape[1] and
        np.allclose(np.diag(spectral_matrix), 1.0) and
        np.all(spectral_matrix >= 0) and
        np.all(spectral_matrix <= 1)
    )

    if is_similarity_matrix:
        # For backward compatibility: use total similarity score
        # This is the sum of ALL pairwise similarities (EasyPanel method)
        return total_similarity_score(spectral_matrix)

    # Calculate condition number of the spectral reference matrix
    # This measures how sensitive the unmixing solution is to noise
    try:
        cond = np.linalg.cond(spectral_matrix)
        # Cap at reasonable maximum
        return min(round(cond, 2), 1000.0)
    except np.linalg.LinAlgError:
        return float('inf')


def total_similarity_score(similarity_matrix: np.ndarray) -> float:
    """Calculate total similarity score (sum of ALL pairwise similarities).

    This is the EasyPanel optimization target - lower total similarity
    means better spectral separation across the entire panel.

    Args:
        similarity_matrix: NxN cosine similarity matrix.

    Returns:
        Sum of all unique pairwise similarity values.
    """
    n = similarity_matrix.shape[0]
    if n < 2:
        return 0.0

    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_values = similarity_matrix[mask]

    return round(float(np.sum(upper_values)), 4)


def complexity_from_spectra(
    spectra: dict[str, np.ndarray],
    wavelengths: Optional[np.ndarray] = None,
) -> float:
    """Calculate complexity index from fluorophore emission spectra.

    Builds the spectral reference matrix and computes its condition number.

    Args:
        spectra: Dict mapping fluorophore name -> emission intensity array.
        wavelengths: Optional wavelength array (for alignment).

    Returns:
        Complexity index (condition number of spectral matrix).
    """
    if len(spectra) < 2:
        return 0.0

    # Stack spectra into matrix (N fluorophores x M wavelengths)
    spectral_matrix = np.array(list(spectra.values()))

    # Normalize each spectrum
    norms = np.linalg.norm(spectral_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    spectral_matrix = spectral_matrix / norms

    return complexity_index(spectral_matrix)


def complexity_index_v2(
    similarity_matrix: np.ndarray,
    critical_threshold: float = 0.90,
    autofluorescence_impact: float = 0.0,
) -> float:
    """Enhanced complexity calculation with critical pair penalties.

    Combines:
    1. Total similarity score (sum of ALL pairs) - EasyPanel method
    2. Extra penalty for critical (>0.90) similarity pairs
    3. Optional autofluorescence impact factor

    Args:
        similarity_matrix: NxN cosine similarity matrix.
        critical_threshold: Similarity above which pairs get extra penalty.
        autofluorescence_impact: Additional penalty for autofluorescence.

    Returns:
        Enhanced complexity index.
    """
    n = similarity_matrix.shape[0]

    if n < 2:
        return autofluorescence_impact

    # Base: total similarity score (sum of ALL pairs)
    base_score = total_similarity_score(similarity_matrix)

    # Extra penalty for critical pairs (similarity > threshold)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_values = similarity_matrix[mask]
    critical_pairs = upper_values[upper_values > critical_threshold]
    critical_penalty = len(critical_pairs) * 5.0  # 5 points per critical pair

    total = base_score + critical_penalty + autofluorescence_impact

    return round(total, 2)


def pair_complexity_contribution(
    similarity: float,
    critical_threshold: float = 0.90,
) -> float:
    """Calculate how much a single pair contributes to total similarity score.

    In the EasyPanel model, ALL pairs contribute to total similarity.
    Pairs above critical threshold get additional penalty.

    Args:
        similarity: Cosine similarity between two fluorophores.
        critical_threshold: Similarity above which pairs get extra penalty.

    Returns:
        Contribution value (similarity + penalty if above threshold).
    """
    contribution = similarity  # ALL pairs contribute

    # Extra penalty for critical pairs
    if similarity > critical_threshold:
        contribution += 5.0  # Penalty for critical overlap

    return contribution


def identify_complexity_drivers(
    similarity_matrix: np.ndarray,
    fluorophore_names: list[str],
    critical_threshold: float = 0.90,
    top_n: int = 10,
) -> list[dict]:
    """Identify the pairs contributing most to total similarity score.

    All pairs contribute to the total similarity score. This function
    ranks them by contribution (similarity value + any critical penalty).

    Args:
        similarity_matrix: NxN cosine similarity matrix.
        fluorophore_names: Ordered list of fluorophore names.
        critical_threshold: Similarity above which pairs are flagged as critical.
        top_n: Number of top contributors to return.

    Returns:
        List of dicts with keys:
            - fluor_a: First fluorophore name
            - fluor_b: Second fluorophore name
            - similarity: Cosine similarity value
            - contribution: Contribution to total similarity score
            - is_critical: Whether similarity exceeds critical threshold
    """
    n = len(fluorophore_names)
    contributors = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            contrib = pair_complexity_contribution(sim, critical_threshold)
            is_critical = sim > critical_threshold

            contributors.append({
                "fluor_a": fluorophore_names[i],
                "fluor_b": fluorophore_names[j],
                "similarity": round(sim, 4),
                "contribution": round(contrib, 4),
                "is_critical": is_critical,
            })

    # Sort by contribution descending
    contributors.sort(key=lambda x: x["contribution"], reverse=True)

    return contributors[:top_n]


def estimate_panel_quality(complexity: float, panel_size: int) -> str:
    """Estimate panel quality based on total similarity score and panel size.

    The total similarity score scales with the number of pairs: n*(n-1)/2.
    Quality is assessed by comparing actual score to theoretical maximum.

    For EasyPanel-style scoring:
    - A panel with N fluorophores has N*(N-1)/2 pairs
    - Average similarity of 0.5 is typical for random selection
    - Well-designed panels achieve average similarity < 0.3

    Args:
        complexity: Total similarity score (sum of all pairwise similarities).
        panel_size: Number of fluorophores in panel.

    Returns:
        Quality rating: 'excellent', 'good', 'acceptable', 'poor'.
    """
    if panel_size < 2:
        return "excellent"

    # Number of unique pairs
    num_pairs = panel_size * (panel_size - 1) / 2

    # Average similarity per pair
    avg_similarity = complexity / num_pairs if num_pairs > 0 else 0

    # Quality thresholds based on average pairwise similarity
    # Lower average = better spectral separation
    if avg_similarity < 0.25:
        return "excellent"
    elif avg_similarity < 0.35:
        return "good"
    elif avg_similarity < 0.50:
        return "acceptable"
    else:
        return "poor"


def count_critical_pairs(
    similarity_matrix: np.ndarray,
    threshold: float = 0.90,
) -> int:
    """Count the number of pairs with similarity above threshold.

    Args:
        similarity_matrix: NxN cosine similarity matrix.
        threshold: Similarity threshold (default 0.90).

    Returns:
        Number of pairs exceeding threshold.
    """
    n = similarity_matrix.shape[0]
    if n < 2:
        return 0

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_values = similarity_matrix[mask]

    return int(np.sum(upper_values > threshold))
