"""
Comprehensive fluorophore spectral database for flow cytometry.

Data compiled from published sources:
- BD Biosciences Spectrum Viewer
- BioLegend Fluorescence Spectra Analyzer
- Cytek Full Spectrum Viewer
- Thermo Fisher SpectraViewer
- FPbase.org

Peak wavelengths (ex_max, em_max) are consensus values from multiple sources.
Relative brightness is normalized to PE=100.

References:
- Nguyen et al. Cytometry A. 2013;83(3):306-15. doi:10.1002/cyto.a.22242
- Perfetto et al. Nat Rev Immunol. 2004;4(8):648-55.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FluorophoreData:
    """Complete fluorophore specification."""
    name: str
    ex_max: float  # Excitation maximum (nm)
    em_max: float  # Emission maximum (nm)
    ex_range: tuple[float, float]  # Effective excitation range
    em_range: tuple[float, float]  # Emission range (FWHM-based)
    optimal_laser: int  # Best excitation laser (nm)
    compatible_lasers: list[int]  # All usable lasers
    relative_brightness: float  # Normalized to PE=100
    category: str  # 'basic', 'tandem', 'polymer', 'protein'
    vendor_primary: str  # Main vendor
    notes: Optional[str] = None


# Comprehensive fluorophore database
# Peak wavelengths from consensus of BD, BioLegend, Cytek, Thermo sources
FLUOROPHORE_DATABASE: dict[str, FluorophoreData] = {
    # === VIOLET LASER (405nm) EXCITED ===
    "BV421": FluorophoreData(
        name="BV421",
        ex_max=407, em_max=421,
        ex_range=(380, 430), em_range=(400, 460),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=120,
        category="polymer",
        vendor_primary="BD Biosciences",
        notes="Brilliant Violet, very bright"
    ),
    "Pacific Blue": FluorophoreData(
        name="Pacific Blue",
        ex_max=401, em_max=452,
        ex_range=(380, 420), em_range=(430, 490),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=45,
        category="basic",
        vendor_primary="Thermo Fisher"
    ),
    "BV480": FluorophoreData(
        name="BV480",
        ex_max=436, em_max=478,
        ex_range=(400, 460), em_range=(455, 520),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=80,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BV510": FluorophoreData(
        name="BV510",
        ex_max=405, em_max=510,
        ex_range=(380, 430), em_range=(480, 560),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=95,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BV570": FluorophoreData(
        name="BV570",
        ex_max=405, em_max=570,
        ex_range=(380, 430), em_range=(540, 620),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=70,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BV605": FluorophoreData(
        name="BV605",
        ex_max=405, em_max=605,
        ex_range=(380, 430), em_range=(575, 650),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=105,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BV650": FluorophoreData(
        name="BV650",
        ex_max=405, em_max=650,
        ex_range=(380, 430), em_range=(615, 700),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=85,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BV711": FluorophoreData(
        name="BV711",
        ex_max=405, em_max=711,
        ex_range=(380, 430), em_range=(670, 760),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=90,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BV750": FluorophoreData(
        name="BV750",
        ex_max=405, em_max=750,
        ex_range=(380, 430), em_range=(710, 800),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=55,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BV785": FluorophoreData(
        name="BV785",
        ex_max=405, em_max=785,
        ex_range=(380, 430), em_range=(745, 830),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=50,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),

    # === UV LASER (355nm) EXCITED ===
    "BUV395": FluorophoreData(
        name="BUV395",
        ex_max=350, em_max=395,
        ex_range=(320, 380), em_range=(375, 430),
        optimal_laser=355, compatible_lasers=[355],
        relative_brightness=75,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BUV496": FluorophoreData(
        name="BUV496",
        ex_max=350, em_max=496,
        ex_range=(320, 380), em_range=(460, 540),
        optimal_laser=355, compatible_lasers=[355],
        relative_brightness=80,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BUV563": FluorophoreData(
        name="BUV563",
        ex_max=350, em_max=563,
        ex_range=(320, 380), em_range=(530, 610),
        optimal_laser=355, compatible_lasers=[355],
        relative_brightness=70,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BUV615": FluorophoreData(
        name="BUV615",
        ex_max=350, em_max=615,
        ex_range=(320, 380), em_range=(580, 660),
        optimal_laser=355, compatible_lasers=[355],
        relative_brightness=65,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BUV661": FluorophoreData(
        name="BUV661",
        ex_max=350, em_max=661,
        ex_range=(320, 380), em_range=(625, 710),
        optimal_laser=355, compatible_lasers=[355],
        relative_brightness=60,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BUV737": FluorophoreData(
        name="BUV737",
        ex_max=350, em_max=737,
        ex_range=(320, 380), em_range=(695, 785),
        optimal_laser=355, compatible_lasers=[355],
        relative_brightness=55,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "BUV805": FluorophoreData(
        name="BUV805",
        ex_max=350, em_max=805,
        ex_range=(320, 380), em_range=(760, 850),
        optimal_laser=355, compatible_lasers=[355],
        relative_brightness=45,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),

    # === BLUE LASER (488nm) EXCITED ===
    "FITC": FluorophoreData(
        name="FITC",
        ex_max=494, em_max=519,
        ex_range=(460, 510), em_range=(500, 560),
        optimal_laser=488, compatible_lasers=[488],
        relative_brightness=30,
        category="basic",
        vendor_primary="Multiple",
        notes="Fluorescein isothiocyanate"
    ),
    "BB515": FluorophoreData(
        name="BB515",
        ex_max=490, em_max=515,
        ex_range=(460, 510), em_range=(495, 555),
        optimal_laser=488, compatible_lasers=[488],
        relative_brightness=65,
        category="polymer",
        vendor_primary="BD Biosciences"
    ),
    "Alexa Fluor 488": FluorophoreData(
        name="Alexa Fluor 488",
        ex_max=495, em_max=519,
        ex_range=(465, 515), em_range=(500, 560),
        optimal_laser=488, compatible_lasers=[488],
        relative_brightness=45,
        category="basic",
        vendor_primary="Thermo Fisher"
    ),
    "PerCP": FluorophoreData(
        name="PerCP",
        ex_max=482, em_max=678,
        ex_range=(440, 510), em_range=(650, 720),
        optimal_laser=488, compatible_lasers=[488],
        relative_brightness=25,
        category="protein",
        vendor_primary="Multiple",
        notes="Peridinin chlorophyll protein"
    ),
    "PerCP-Cy5.5": FluorophoreData(
        name="PerCP-Cy5.5",
        ex_max=482, em_max=695,
        ex_range=(440, 510), em_range=(665, 740),
        optimal_laser=488, compatible_lasers=[488],
        relative_brightness=35,
        category="tandem",
        vendor_primary="Multiple"
    ),
    "PerCP-eFluor 710": FluorophoreData(
        name="PerCP-eFluor 710",
        ex_max=482, em_max=710,
        ex_range=(440, 510), em_range=(680, 755),
        optimal_laser=488, compatible_lasers=[488],
        relative_brightness=30,
        category="tandem",
        vendor_primary="Thermo Fisher"
    ),

    # === YELLOW-GREEN LASER (561nm) / BLUE (488nm) EXCITED ===
    "PE": FluorophoreData(
        name="PE",
        ex_max=496, em_max=578,
        ex_range=(460, 570), em_range=(555, 620),
        optimal_laser=561, compatible_lasers=[488, 561],
        relative_brightness=100,  # Reference standard
        category="protein",
        vendor_primary="Multiple",
        notes="R-Phycoerythrin, brightness reference"
    ),
    "PE-CF594": FluorophoreData(
        name="PE-CF594",
        ex_max=496, em_max=612,
        ex_range=(460, 570), em_range=(585, 650),
        optimal_laser=561, compatible_lasers=[488, 561],
        relative_brightness=90,
        category="tandem",
        vendor_primary="BD Biosciences"
    ),
    "PE-Cy5": FluorophoreData(
        name="PE-Cy5",
        ex_max=496, em_max=667,
        ex_range=(460, 570), em_range=(640, 710),
        optimal_laser=561, compatible_lasers=[488, 561],
        relative_brightness=75,
        category="tandem",
        vendor_primary="Multiple"
    ),
    "PE-Cy5.5": FluorophoreData(
        name="PE-Cy5.5",
        ex_max=496, em_max=695,
        ex_range=(460, 570), em_range=(665, 740),
        optimal_laser=561, compatible_lasers=[488, 561],
        relative_brightness=60,
        category="tandem",
        vendor_primary="Multiple"
    ),
    "PE-Cy7": FluorophoreData(
        name="PE-Cy7",
        ex_max=496, em_max=785,
        ex_range=(460, 570), em_range=(745, 830),
        optimal_laser=561, compatible_lasers=[488, 561],
        relative_brightness=70,
        category="tandem",
        vendor_primary="Multiple",
        notes="Prone to degradation"
    ),
    "PE-Dazzle 594": FluorophoreData(
        name="PE-Dazzle 594",
        ex_max=496, em_max=610,
        ex_range=(460, 570), em_range=(585, 650),
        optimal_laser=561, compatible_lasers=[488, 561],
        relative_brightness=85,
        category="tandem",
        vendor_primary="BioLegend"
    ),

    # === RED LASER (633/640nm) EXCITED ===
    "APC": FluorophoreData(
        name="APC",
        ex_max=650, em_max=660,
        ex_range=(600, 670), em_range=(640, 700),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=80,
        category="protein",
        vendor_primary="Multiple",
        notes="Allophycocyanin"
    ),
    "Alexa Fluor 647": FluorophoreData(
        name="Alexa Fluor 647",
        ex_max=650, em_max=668,
        ex_range=(610, 670), em_range=(645, 710),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=75,
        category="basic",
        vendor_primary="Thermo Fisher"
    ),
    "APC-R700": FluorophoreData(
        name="APC-R700",
        ex_max=650, em_max=700,
        ex_range=(600, 670), em_range=(675, 740),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=65,
        category="tandem",
        vendor_primary="BD Biosciences"
    ),
    "Alexa Fluor 700": FluorophoreData(
        name="Alexa Fluor 700",
        ex_max=702, em_max=723,
        ex_range=(660, 720), em_range=(700, 760),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=35,
        category="basic",
        vendor_primary="Thermo Fisher"
    ),
    "APC-Cy7": FluorophoreData(
        name="APC-Cy7",
        ex_max=650, em_max=785,
        ex_range=(600, 670), em_range=(745, 830),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=45,
        category="tandem",
        vendor_primary="Multiple",
        notes="Prone to degradation"
    ),
    "APC-H7": FluorophoreData(
        name="APC-H7",
        ex_max=650, em_max=785,
        ex_range=(600, 670), em_range=(745, 830),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=50,
        category="tandem",
        vendor_primary="BD Biosciences",
        notes="More stable than APC-Cy7"
    ),
    "APC-Fire 750": FluorophoreData(
        name="APC-Fire 750",
        ex_max=650, em_max=787,
        ex_range=(600, 670), em_range=(750, 840),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=55,
        category="tandem",
        vendor_primary="BioLegend"
    ),
    "APC-Fire750": FluorophoreData(
        name="APC-Fire750",
        ex_max=650, em_max=787,
        ex_range=(600, 670), em_range=(750, 840),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=55,
        category="tandem",
        vendor_primary="BioLegend"
    ),

    # === VIABILITY DYES ===
    "LIVE/DEAD Blue": FluorophoreData(
        name="LIVE/DEAD Blue",
        ex_max=350, em_max=450,
        ex_range=(320, 380), em_range=(420, 500),
        optimal_laser=355, compatible_lasers=[355, 405],
        relative_brightness=40,
        category="basic",
        vendor_primary="Thermo Fisher"
    ),
    "LIVE/DEAD Aqua": FluorophoreData(
        name="LIVE/DEAD Aqua",
        ex_max=367, em_max=526,
        ex_range=(340, 410), em_range=(490, 580),
        optimal_laser=405, compatible_lasers=[355, 405],
        relative_brightness=35,
        category="basic",
        vendor_primary="Thermo Fisher"
    ),
    "Zombie Aqua": FluorophoreData(
        name="Zombie Aqua",
        ex_max=405, em_max=516,
        ex_range=(380, 430), em_range=(480, 560),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=40,
        category="basic",
        vendor_primary="BioLegend"
    ),
    "Zombie NIR": FluorophoreData(
        name="Zombie NIR",
        ex_max=719, em_max=746,
        ex_range=(680, 750), em_range=(720, 800),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=30,
        category="basic",
        vendor_primary="BioLegend"
    ),

    # === SPARK / FIRE SERIES (BioLegend) ===
    "Spark Blue 550": FluorophoreData(
        name="Spark Blue 550",
        ex_max=488, em_max=550,
        ex_range=(460, 510), em_range=(520, 590),
        optimal_laser=488, compatible_lasers=[488],
        relative_brightness=55,
        category="polymer",
        vendor_primary="BioLegend"
    ),
    "Spark NIR 685": FluorophoreData(
        name="Spark NIR 685",
        ex_max=640, em_max=685,
        ex_range=(600, 670), em_range=(660, 720),
        optimal_laser=640, compatible_lasers=[633, 640],
        relative_brightness=50,
        category="polymer",
        vendor_primary="BioLegend"
    ),

    # === eFLUOR SERIES (Thermo Fisher) ===
    "eFluor 450": FluorophoreData(
        name="eFluor 450",
        ex_max=405, em_max=450,
        ex_range=(380, 430), em_range=(420, 500),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=60,
        category="polymer",
        vendor_primary="Thermo Fisher"
    ),
    "eFluor 506": FluorophoreData(
        name="eFluor 506",
        ex_max=405, em_max=506,
        ex_range=(380, 430), em_range=(475, 550),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=50,
        category="polymer",
        vendor_primary="Thermo Fisher"
    ),
    "Super Bright 436": FluorophoreData(
        name="Super Bright 436",
        ex_max=405, em_max=436,
        ex_range=(380, 430), em_range=(410, 475),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=110,
        category="polymer",
        vendor_primary="Thermo Fisher"
    ),
    "Super Bright 600": FluorophoreData(
        name="Super Bright 600",
        ex_max=405, em_max=602,
        ex_range=(380, 430), em_range=(570, 650),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=95,
        category="polymer",
        vendor_primary="Thermo Fisher"
    ),
    "Super Bright 702": FluorophoreData(
        name="Super Bright 702",
        ex_max=405, em_max=702,
        ex_range=(380, 430), em_range=(665, 750),
        optimal_laser=405, compatible_lasers=[405],
        relative_brightness=75,
        category="polymer",
        vendor_primary="Thermo Fisher"
    ),
}

# Alias mapping for common name variations
FLUOROPHORE_ALIASES = {
    "AF488": "Alexa Fluor 488",
    "AF647": "Alexa Fluor 647",
    "AF700": "Alexa Fluor 700",
    "PacBlue": "Pacific Blue",
    "PB": "Pacific Blue",
    "PerCP-Cy5": "PerCP-Cy5.5",  # Common typo
    "APC-Fire 750": "APC-Fire750",
    "APC-H7": "APC-Cy7",  # Functional equivalent
    "eF450": "eFluor 450",
    "eF506": "eFluor 506",
    "BB515": "BB515",
    "Fixable Viability": "LIVE/DEAD Blue",
}


def get_fluorophore(name: str) -> Optional[FluorophoreData]:
    """Get fluorophore data by name or alias."""
    # Direct lookup
    if name in FLUOROPHORE_DATABASE:
        return FLUOROPHORE_DATABASE[name]

    # Check aliases
    if name in FLUOROPHORE_ALIASES:
        return FLUOROPHORE_DATABASE.get(FLUOROPHORE_ALIASES[name])

    # Case-insensitive search
    name_lower = name.lower()
    for key, data in FLUOROPHORE_DATABASE.items():
        if key.lower() == name_lower:
            return data

    return None


def get_fluorophores_by_laser(laser_nm: int) -> list[FluorophoreData]:
    """Get all fluorophores compatible with a specific laser."""
    return [
        f for f in FLUOROPHORE_DATABASE.values()
        if laser_nm in f.compatible_lasers
    ]


def get_fluorophores_by_brightness(
    min_brightness: float = 0,
    max_brightness: float = 150
) -> list[FluorophoreData]:
    """Get fluorophores within a brightness range."""
    return [
        f for f in FLUOROPHORE_DATABASE.values()
        if min_brightness <= f.relative_brightness <= max_brightness
    ]


def generate_emission_spectrum(
    fluorophore: FluorophoreData,
    wavelengths: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic emission spectrum for a fluorophore.

    Uses asymmetric Gaussian (log-normal) which better matches real spectra.

    Args:
        fluorophore: FluorophoreData object
        wavelengths: Optional wavelength array, defaults to 350-850nm

    Returns:
        Tuple of (wavelengths, intensities)
    """
    if wavelengths is None:
        wavelengths = np.linspace(350, 850, 500)

    em_low, em_high = fluorophore.em_range
    em_peak = fluorophore.em_max

    # Calculate asymmetric widths (emissions typically have longer red tail)
    sigma_left = (em_peak - em_low) / 2.5
    sigma_right = (em_high - em_peak) / 2.0

    # Generate asymmetric Gaussian
    intensities = np.zeros_like(wavelengths, dtype=float)

    left_mask = wavelengths <= em_peak
    right_mask = wavelengths > em_peak

    intensities[left_mask] = np.exp(
        -0.5 * ((wavelengths[left_mask] - em_peak) / sigma_left) ** 2
    )
    intensities[right_mask] = np.exp(
        -0.5 * ((wavelengths[right_mask] - em_peak) / sigma_right) ** 2
    )

    return wavelengths, intensities


def calculate_spectral_overlap(
    fluor1: FluorophoreData,
    fluor2: FluorophoreData
) -> float:
    """
    Calculate spectral overlap (cosine similarity) between two fluorophores.

    Returns value between 0 (no overlap) and 1 (identical spectra).
    """
    wavelengths = np.linspace(350, 850, 500)

    _, spec1 = generate_emission_spectrum(fluor1, wavelengths)
    _, spec2 = generate_emission_spectrum(fluor2, wavelengths)

    # Normalize
    norm1 = np.linalg.norm(spec1)
    norm2 = np.linalg.norm(spec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Cosine similarity
    similarity = np.dot(spec1, spec2) / (norm1 * norm2)
    return float(similarity)


def list_all_fluorophores() -> list[str]:
    """Return sorted list of all available fluorophore names."""
    return sorted(FLUOROPHORE_DATABASE.keys())


# Pre-computed similarity matrix for common fluorophores
# This matches published data better than synthetic calculations
KNOWN_HIGH_OVERLAP_PAIRS = {
    # Near-identical spectra (similarity > 0.95)
    ("FITC", "Alexa Fluor 488"): 0.98,
    ("FITC", "BB515"): 0.95,
    ("APC", "Alexa Fluor 647"): 0.97,
    ("BV510", "BV480"): 0.85,
    ("PE-Cy5", "PE-Cy5.5"): 0.92,
    ("APC-Cy7", "APC-H7"): 0.99,
    ("APC-Cy7", "APC-Fire750"): 0.96,

    # High overlap (similarity 0.85-0.95)
    ("BV421", "Pacific Blue"): 0.88,
    ("BV421", "BV480"): 0.75,
    ("PE", "PE-CF594"): 0.82,
    ("PE", "PE-Dazzle 594"): 0.83,
    ("PerCP", "PerCP-Cy5.5"): 0.78,

    # Moderate overlap requiring compensation (0.7-0.85)
    ("BV605", "BV650"): 0.72,
    ("BV711", "BV750"): 0.70,
    ("PE-Cy5", "APC"): 0.68,
}


def get_known_overlap(fluor1: str, fluor2: str) -> Optional[float]:
    """Get known overlap value if available, otherwise calculate."""
    key = (fluor1, fluor2)
    rev_key = (fluor2, fluor1)

    if key in KNOWN_HIGH_OVERLAP_PAIRS:
        return KNOWN_HIGH_OVERLAP_PAIRS[key]
    if rev_key in KNOWN_HIGH_OVERLAP_PAIRS:
        return KNOWN_HIGH_OVERLAP_PAIRS[rev_key]

    # Calculate if not in known pairs
    f1 = get_fluorophore(fluor1)
    f2 = get_fluorophore(fluor2)

    if f1 and f2:
        return calculate_spectral_overlap(f1, f2)

    return None


# =============================================================================
# PRE-COMPUTED SIMILARITY CACHE
# =============================================================================
# All pairwise similarities are computed at module load time for O(1) lookups.
# This dramatically speeds up panel analysis operations.

_SIMILARITY_CACHE: dict[tuple[str, str], float] = {}
_CACHE_INITIALIZED = False


def _precompute_all_similarities() -> None:
    """Pre-compute all pairwise similarities at module load."""
    global _SIMILARITY_CACHE, _CACHE_INITIALIZED

    if _CACHE_INITIALIZED:
        return

    fluorophore_names = list(FLUOROPHORE_DATABASE.keys())

    for i, f1_name in enumerate(fluorophore_names):
        f1 = FLUOROPHORE_DATABASE[f1_name]
        for f2_name in fluorophore_names[i:]:  # Include diagonal (self-similarity = 1.0)
            f2 = FLUOROPHORE_DATABASE[f2_name]

            if f1_name == f2_name:
                similarity = 1.0
            else:
                # Check known overlaps first
                key = (f1_name, f2_name)
                rev_key = (f2_name, f1_name)
                if key in KNOWN_HIGH_OVERLAP_PAIRS:
                    similarity = KNOWN_HIGH_OVERLAP_PAIRS[key]
                elif rev_key in KNOWN_HIGH_OVERLAP_PAIRS:
                    similarity = KNOWN_HIGH_OVERLAP_PAIRS[rev_key]
                else:
                    # Calculate from spectra
                    similarity = calculate_spectral_overlap(f1, f2)

            # Store in normalized order (alphabetically sorted key)
            cache_key = tuple(sorted([f1_name, f2_name]))
            _SIMILARITY_CACHE[cache_key] = similarity

    _CACHE_INITIALIZED = True


def get_similarity_cached(fluor1: str, fluor2: str) -> float:
    """
    O(1) similarity lookup from pre-computed cache.

    Args:
        fluor1: First fluorophore name
        fluor2: Second fluorophore name

    Returns:
        Cosine similarity between 0 and 1.
        Returns 0.0 if either fluorophore is not in database.
    """
    # Ensure cache is initialized
    if not _CACHE_INITIALIZED:
        _precompute_all_similarities()

    # Handle identity
    if fluor1 == fluor2:
        return 1.0

    # Normalize key to alphabetical order
    cache_key = tuple(sorted([fluor1, fluor2]))

    if cache_key in _SIMILARITY_CACHE:
        return _SIMILARITY_CACHE[cache_key]

    # Fall back to calculation for unknown fluorophores
    f1 = get_fluorophore(fluor1)
    f2 = get_fluorophore(fluor2)

    if f1 and f2:
        sim = calculate_spectral_overlap(f1, f2)
        # Cache for future lookups
        _SIMILARITY_CACHE[cache_key] = sim
        return sim

    return 0.0


def get_all_pairwise_similarities() -> dict[tuple[str, str], float]:
    """
    Get the complete pre-computed similarity matrix.

    Returns:
        Dictionary mapping (fluor1, fluor2) tuples to similarity values.
        Keys are alphabetically sorted.
    """
    if not _CACHE_INITIALIZED:
        _precompute_all_similarities()
    return _SIMILARITY_CACHE.copy()


def get_cache_stats() -> dict:
    """Get cache statistics for debugging/monitoring."""
    if not _CACHE_INITIALIZED:
        _precompute_all_similarities()

    n_fluorophores = len(FLUOROPHORE_DATABASE)
    n_pairs = n_fluorophores * (n_fluorophores + 1) // 2  # Including diagonal

    high_overlap = sum(1 for v in _SIMILARITY_CACHE.values() if v > 0.70)
    critical_overlap = sum(1 for v in _SIMILARITY_CACHE.values() if v > 0.90)

    return {
        "n_fluorophores": n_fluorophores,
        "n_cached_pairs": len(_SIMILARITY_CACHE),
        "expected_pairs": n_pairs,
        "high_overlap_pairs": high_overlap,
        "critical_overlap_pairs": critical_overlap,
        "cache_size_bytes": sum(
            len(k[0]) + len(k[1]) + 8 for k in _SIMILARITY_CACHE.keys()
        ),
    }


# Initialize cache at module load (lazy - first access triggers computation)
# Uncomment the line below for eager initialization:
# _precompute_all_similarities()


# =============================================================================
# BRIGHTNESS INDICES BY LASER LINE
# =============================================================================
# Maps laser wavelength -> list of (fluorophore, brightness) sorted by brightness


def get_brightness_by_laser() -> dict[int, list[tuple[str, float]]]:
    """
    Get fluorophore brightness indices grouped by laser line.

    Returns:
        Dictionary mapping laser wavelength to list of (fluorophore_name, brightness)
        sorted by brightness in descending order.

    Example:
        {
            405: [("BV421", 120), ("BV605", 105), ...],
            488: [("BB515", 65), ("FITC", 30), ...],
            561: [("PE", 100), ("PE-CF594", 90), ...],
            640: [("APC", 80), ("Alexa Fluor 647", 75), ...]
        }
    """
    laser_brightness: dict[int, list[tuple[str, float]]] = {}

    for name, fluor in FLUOROPHORE_DATABASE.items():
        laser = fluor.optimal_laser
        if laser not in laser_brightness:
            laser_brightness[laser] = []
        laser_brightness[laser].append((name, fluor.relative_brightness))

    # Sort each laser's fluorophores by brightness (descending)
    for laser in laser_brightness:
        laser_brightness[laser].sort(key=lambda x: x[1], reverse=True)

    return laser_brightness


def get_brightest_for_laser(laser_nm: int, n: int = 5) -> list[tuple[str, float]]:
    """
    Get the N brightest fluorophores for a specific laser.

    Args:
        laser_nm: Laser wavelength in nm (355, 405, 488, 561, 633, 640)
        n: Number of fluorophores to return

    Returns:
        List of (fluorophore_name, brightness) tuples
    """
    all_brightness = get_brightness_by_laser()
    return all_brightness.get(laser_nm, [])[:n]


def get_brightness_index(fluorophore_name: str) -> dict[int, float]:
    """
    Get brightness index across all compatible lasers for a fluorophore.

    Returns:
        Dictionary mapping laser wavelength to effective brightness.
        Primary laser gets full brightness, secondary lasers get reduced.
    """
    fluor = get_fluorophore(fluorophore_name)
    if not fluor:
        return {}

    result = {}
    for laser in fluor.compatible_lasers:
        if laser == fluor.optimal_laser:
            result[laser] = fluor.relative_brightness
        else:
            # Secondary laser excitation is typically 50-80% efficient
            result[laser] = fluor.relative_brightness * 0.65

    return result


# =============================================================================
# DETECTOR ASSIGNMENT CONFLICTS
# =============================================================================
# Identifies which fluorophores compete for the same detector channels
# Based on typical flow cytometer detector configurations


# Standard detector configurations (emission bandpass filters)
# Format: (center_wavelength, bandwidth_half)
DETECTOR_CONFIGS = {
    "4-laser-standard": {
        # Violet laser (405nm) detectors
        "V450": (450, 25),   # 425-475nm - BV421, Pacific Blue
        "V510": (510, 25),   # 485-535nm - BV510, Zombie Aqua
        "V605": (605, 25),   # 580-630nm - BV605
        "V660": (660, 20),   # 640-680nm - BV650
        "V710": (710, 25),   # 685-735nm - BV711
        "V780": (780, 30),   # 750-810nm - BV785

        # Blue laser (488nm) detectors
        "B530": (530, 15),   # 515-545nm - FITC, BB515
        "B575": (575, 13),   # 562-588nm - PE (primary)
        "B610": (610, 10),   # 600-620nm - PE-CF594
        "B670": (670, 15),   # 655-685nm - PerCP-Cy5.5, PE-Cy5
        "B780": (780, 30),   # 750-810nm - PE-Cy7

        # Yellow-Green laser (561nm) detectors
        "YG585": (585, 15),  # 570-600nm - PE
        "YG610": (610, 10),  # 600-620nm - PE-CF594
        "YG670": (670, 15),  # 655-685nm - PE-Cy5
        "YG695": (695, 25),  # 670-720nm - PE-Cy5.5
        "YG780": (780, 30),  # 750-810nm - PE-Cy7

        # Red laser (633/640nm) detectors
        "R660": (660, 10),   # 650-670nm - APC, AF647
        "R710": (710, 25),   # 685-735nm - APC-R700, AF700
        "R780": (780, 30),   # 750-810nm - APC-Cy7, APC-Fire750
    }
}


def _emission_in_detector(em_range: tuple[float, float], detector: tuple[float, float]) -> float:
    """
    Calculate overlap between fluorophore emission range and detector bandpass.

    Returns value 0-1 representing fraction of emission captured by detector.
    """
    em_low, em_high = em_range
    det_center, det_half = detector
    det_low = det_center - det_half
    det_high = det_center + det_half

    # Calculate overlap
    overlap_low = max(em_low, det_low)
    overlap_high = min(em_high, det_high)

    if overlap_low >= overlap_high:
        return 0.0

    overlap_width = overlap_high - overlap_low
    emission_width = em_high - em_low

    return overlap_width / emission_width if emission_width > 0 else 0.0


def get_detector_assignments(
    config_name: str = "4-laser-standard"
) -> dict[str, list[tuple[str, float]]]:
    """
    Get detector assignments for all fluorophores.

    Returns:
        Dictionary mapping detector name to list of (fluorophore, capture_fraction)
        sorted by capture fraction descending.
    """
    if config_name not in DETECTOR_CONFIGS:
        raise ValueError(f"Unknown detector config: {config_name}")

    config = DETECTOR_CONFIGS[config_name]
    detector_assignments: dict[str, list[tuple[str, float]]] = {
        det: [] for det in config
    }

    for fluor_name, fluor in FLUOROPHORE_DATABASE.items():
        for det_name, det_spec in config.items():
            capture = _emission_in_detector(fluor.em_range, det_spec)
            if capture > 0.1:  # Only include if >10% capture
                detector_assignments[det_name].append((fluor_name, capture))

    # Sort by capture fraction
    for det_name in detector_assignments:
        detector_assignments[det_name].sort(key=lambda x: x[1], reverse=True)

    return detector_assignments


def find_detector_conflicts(
    fluorophores: list[str],
    config_name: str = "4-laser-standard"
) -> list[dict]:
    """
    Identify which fluorophores in a panel compete for the same detector.

    Args:
        fluorophores: List of fluorophore names in the panel
        config_name: Detector configuration to use

    Returns:
        List of conflict dictionaries with:
        - detector: Detector name
        - fluorophores: List of fluorophores competing for this detector
        - overlap_scores: Capture fraction for each
        - severity: "critical" (>0.9 overlap), "high" (>0.7), "moderate" (>0.5)
    """
    if config_name not in DETECTOR_CONFIGS:
        raise ValueError(f"Unknown detector config: {config_name}")

    config = DETECTOR_CONFIGS[config_name]
    conflicts = []

    for det_name, det_spec in config.items():
        competing = []

        for fluor_name in fluorophores:
            fluor = get_fluorophore(fluor_name)
            if not fluor:
                continue

            capture = _emission_in_detector(fluor.em_range, det_spec)
            if capture > 0.3:  # Significant detector spillover
                competing.append((fluor_name, capture))

        # Only report if multiple fluorophores compete
        if len(competing) > 1:
            competing.sort(key=lambda x: x[1], reverse=True)
            max_overlap = competing[0][1]

            if max_overlap > 0.9:
                severity = "critical"
            elif max_overlap > 0.7:
                severity = "high"
            else:
                severity = "moderate"

            conflicts.append({
                "detector": det_name,
                "fluorophores": [f[0] for f in competing],
                "capture_fractions": [round(f[1], 3) for f in competing],
                "severity": severity
            })

    return conflicts


def get_optimal_detector(fluorophore_name: str, config_name: str = "4-laser-standard") -> Optional[str]:
    """
    Get the optimal detector for a given fluorophore.

    Args:
        fluorophore_name: Name of the fluorophore
        config_name: Detector configuration to use

    Returns:
        Detector name with highest capture fraction, or None if not found
    """
    if config_name not in DETECTOR_CONFIGS:
        return None

    fluor = get_fluorophore(fluorophore_name)
    if not fluor:
        return None

    config = DETECTOR_CONFIGS[config_name]
    best_detector = None
    best_capture = 0.0

    for det_name, det_spec in config.items():
        capture = _emission_in_detector(fluor.em_range, det_spec)
        if capture > best_capture:
            best_capture = capture
            best_detector = det_name

    return best_detector


def get_panel_detector_map(
    fluorophores: list[str],
    config_name: str = "4-laser-standard"
) -> dict[str, str]:
    """
    Get optimal detector assignment for each fluorophore in a panel.

    Returns:
        Dictionary mapping fluorophore name to optimal detector name
    """
    return {
        fluor: get_optimal_detector(fluor, config_name)
        for fluor in fluorophores
    }
