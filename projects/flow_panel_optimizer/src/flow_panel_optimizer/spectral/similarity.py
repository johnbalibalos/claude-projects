"""Cosine similarity calculations for emission spectra.

Reference:
    FlowJo Cosine Similarity Matrix documentation
    https://docs.flowjo.com/flowjo/experiment-based-platforms/plat-comp-overview/cosine-similarity-matrix/
"""

from typing import Optional
import numpy as np
from scipy.spatial.distance import cosine

from flow_panel_optimizer.models.spectrum import Spectrum


def cosine_similarity(
    spectrum_a: np.ndarray | Spectrum,
    spectrum_b: np.ndarray | Spectrum,
    align_wavelengths: bool = True,
) -> float:
    """Calculate cosine similarity between two emission spectra.

    Cosine similarity measures the angle between two vectors, providing
    a value from 0 (orthogonal/completely different) to 1 (identical).

    For flow cytometry, high similarity (>0.90) indicates fluorophores
    that will have significant spectral overlap and spillover.

    Args:
        spectrum_a: Emission intensity values or Spectrum object.
        spectrum_b: Emission intensity values or Spectrum object.
        align_wavelengths: If True and inputs are Spectrum objects,
            interpolate to common wavelength grid.

    Returns:
        Similarity score from 0.0 (orthogonal) to 1.0 (identical).

    Raises:
        ValueError: If arrays have different lengths (when not aligning).

    Example:
        >>> import numpy as np
        >>> spec_a = np.array([0.1, 0.5, 1.0, 0.7, 0.2])
        >>> spec_b = np.array([0.1, 0.5, 1.0, 0.7, 0.2])
        >>> cosine_similarity(spec_a, spec_b)
        1.0

        >>> spec_c = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        >>> cosine_similarity(spec_a, spec_c)  # Different spectrum
        0.18...
    """
    # Handle Spectrum objects
    if isinstance(spectrum_a, Spectrum) and isinstance(spectrum_b, Spectrum):
        if align_wavelengths:
            # Create common wavelength grid
            min_wl = max(spectrum_a.wavelengths.min(), spectrum_b.wavelengths.min())
            max_wl = min(spectrum_a.wavelengths.max(), spectrum_b.wavelengths.max())
            common_wl = np.linspace(min_wl, max_wl, 100)

            spectrum_a = spectrum_a.interpolate(common_wl)
            spectrum_b = spectrum_b.interpolate(common_wl)

        arr_a = spectrum_a.intensities
        arr_b = spectrum_b.intensities
    elif isinstance(spectrum_a, Spectrum):
        arr_a = spectrum_a.intensities
        arr_b = np.asarray(spectrum_b)
    elif isinstance(spectrum_b, Spectrum):
        arr_a = np.asarray(spectrum_a)
        arr_b = spectrum_b.intensities
    else:
        arr_a = np.asarray(spectrum_a)
        arr_b = np.asarray(spectrum_b)

    if len(arr_a) != len(arr_b):
        raise ValueError(
            f"Spectrum arrays must have same length. Got {len(arr_a)} and {len(arr_b)}"
        )

    # Handle zero vectors
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Cosine similarity = 1 - cosine distance
    # scipy.spatial.distance.cosine computes the cosine distance
    return 1.0 - cosine(arr_a, arr_b)


def build_similarity_matrix(
    spectra: dict[str, np.ndarray | Spectrum],
) -> tuple[list[str], np.ndarray]:
    """Build NxN cosine similarity matrix for all fluorophores.

    Args:
        spectra: Dict mapping fluorophore name -> emission spectrum.
            Values can be numpy arrays or Spectrum objects.

    Returns:
        Tuple of (fluorophore_names, similarity_matrix):
            - fluorophore_names: Ordered list of fluorophore names.
            - similarity_matrix: NxN numpy array where [i,j] is the
              cosine similarity between fluorophores i and j.

    Example:
        >>> spectra = {
        ...     'FITC': np.array([0.2, 0.8, 1.0, 0.5, 0.1]),
        ...     'PE': np.array([0.1, 0.3, 0.6, 1.0, 0.4]),
        ...     'APC': np.array([0.0, 0.1, 0.3, 0.8, 1.0]),
        ... }
        >>> names, matrix = build_similarity_matrix(spectra)
        >>> names
        ['FITC', 'PE', 'APC']
        >>> matrix.shape
        (3, 3)
    """
    names = list(spectra.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif i < j:
                sim = cosine_similarity(spectra[names[i]], spectra[names[j]])
                matrix[i, j] = sim
                matrix[j, i] = sim  # Symmetric

    return names, matrix


def find_high_similarity_pairs(
    similarity_matrix: np.ndarray,
    fluorophore_names: list[str],
    threshold: float = 0.90,
) -> list[tuple[str, str, float]]:
    """Find fluorophore pairs with similarity above threshold.

    Args:
        similarity_matrix: NxN cosine similarity matrix.
        fluorophore_names: Ordered list of fluorophore names.
        threshold: Minimum similarity to include (default 0.90).

    Returns:
        List of (fluor_a, fluor_b, similarity) tuples, sorted by
        similarity descending.

    Example:
        >>> names = ['FITC', 'BB515', 'PE']
        >>> matrix = np.array([
        ...     [1.0, 0.98, 0.45],
        ...     [0.98, 1.0, 0.42],
        ...     [0.45, 0.42, 1.0],
        ... ])
        >>> find_high_similarity_pairs(matrix, names, threshold=0.90)
        [('FITC', 'BB515', 0.98)]
    """
    n = len(fluorophore_names)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            if sim >= threshold:
                pairs.append((fluorophore_names[i], fluorophore_names[j], sim))

    # Sort by similarity descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def similarity_risk_level(similarity: float) -> str:
    """Classify similarity value into risk categories.

    Based on common flow cytometry guidelines:
    - >= 0.98: Critical - essentially identical spectra
    - >= 0.95: High - significant unmixing issues expected
    - >= 0.90: Moderate - may cause problems with co-expressed markers
    - >= 0.80: Low - generally acceptable
    - < 0.80: Minimal - safe to use together

    Args:
        similarity: Cosine similarity value (0-1).

    Returns:
        Risk level string.
    """
    if similarity >= 0.98:
        return "critical"
    elif similarity >= 0.95:
        return "high"
    elif similarity >= 0.90:
        return "moderate"
    elif similarity >= 0.80:
        return "low"
    else:
        return "minimal"


def interpolate_to_common_grid(
    spectra: dict[str, Spectrum],
    wavelength_range: Optional[tuple[float, float]] = None,
    num_points: int = 100,
) -> dict[str, np.ndarray]:
    """Interpolate all spectra to a common wavelength grid.

    Args:
        spectra: Dict mapping name -> Spectrum object.
        wavelength_range: Optional (min, max) wavelength range.
            If None, uses intersection of all spectrum ranges.
        num_points: Number of wavelength points to interpolate to.

    Returns:
        Dict mapping name -> interpolated intensity array.
    """
    if not spectra:
        return {}

    # Find common wavelength range
    if wavelength_range is None:
        min_wl = max(s.wavelengths.min() for s in spectra.values())
        max_wl = min(s.wavelengths.max() for s in spectra.values())
    else:
        min_wl, max_wl = wavelength_range

    common_wl = np.linspace(min_wl, max_wl, num_points)

    result = {}
    for name, spectrum in spectra.items():
        interpolated = spectrum.interpolate(common_wl)
        result[name] = interpolated.intensities

    return result
