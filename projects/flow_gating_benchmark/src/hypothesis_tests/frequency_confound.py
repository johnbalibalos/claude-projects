"""
Frequency Confound Hypothesis Tests.

Tests whether model failures are due to token frequency in pre-training corpus
vs. genuine reasoning deficits.

Hypotheses:
- H_A (Frequency View): Performance correlates with term frequency in corpus
- H_B (Reasoning View): Performance correlates with logical complexity

Tests:
1. PubMed Correlation: Plot Log(Citation Count) vs Model F1
2. Alien Cell Injection: Replace population names with nonsense tokens
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_HTTPX = False

from curation.schemas import (
    GateNode,
    GatingHierarchy,
    MarkerExpression,
    Panel,
    TestCase,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PUBMED FREQUENCY LOOKUP
# =============================================================================


@dataclass
class PopulationFrequency:
    """Frequency data for a cell population term."""

    term: str
    pubmed_count: int
    log_count: float
    query_used: str
    fetch_time: float


class PubMedFrequencyLookup:
    """
    Lookup cell population term frequency in PubMed.

    Uses NCBI ESearch API to get citation counts as a proxy for
    how often a term appears in biomedical literature (and thus
    likely in pre-training data).
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: str | None = None,
        rate_limit_delay: float = 0.35,  # NCBI allows 3 requests/sec without key
    ):
        """
        Initialize PubMed lookup.

        Args:
            api_key: NCBI API key (optional, increases rate limit)
            cache_path: Path to cache file for persistence
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key
        self.cache_path = cache_path
        self.rate_limit_delay = rate_limit_delay
        self._cache: dict[str, PopulationFrequency] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached results from disk."""
        if self.cache_path:
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                    for term, freq_data in data.items():
                        self._cache[term] = PopulationFrequency(**freq_data)
                logger.info(f"Loaded {len(self._cache)} cached frequencies")
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        if self.cache_path:
            data = {
                term: {
                    "term": freq.term,
                    "pubmed_count": freq.pubmed_count,
                    "log_count": freq.log_count,
                    "query_used": freq.query_used,
                    "fetch_time": freq.fetch_time,
                }
                for term, freq in self._cache.items()
            }
            with open(self.cache_path, "w") as f:
                json.dump(data, f, indent=2)

    def _build_query(self, population_name: str) -> str:
        """
        Build PubMed query for a cell population.

        Handles common variations:
        - "CD4+ T cells" -> "CD4+ T cells" OR "CD4-positive T cells"
        - "Tregs" -> "Tregs" OR "regulatory T cells"
        """
        # Normalize the term
        term = population_name.strip()

        # Build query with variations
        queries = [f'"{term}"[Title/Abstract]']

        # Add variations for +/- notation
        if "+" in term:
            variation = term.replace("+", "-positive ")
            queries.append(f'"{variation}"[Title/Abstract]')
        if "-" in term and not term.endswith("-"):
            # Avoid matching "CD4-" as "CD4-negative"
            if re.search(r"[A-Z0-9]+-$", term):
                variation = term[:-1] + "-negative"
                queries.append(f'"{variation}"[Title/Abstract]')

        # Common abbreviation expansions
        abbreviations = {
            "Treg": "regulatory T cell",
            "Tfh": "T follicular helper",
            "NK": "natural killer",
            "DC": "dendritic cell",
            "pDC": "plasmacytoid dendritic cell",
            "mDC": "myeloid dendritic cell",
            "TEMRA": "terminally differentiated effector memory",
        }
        for abbrev, full in abbreviations.items():
            if abbrev.lower() in term.lower():
                queries.append(f'"{full}"[Title/Abstract]')

        # Combine with OR
        return " OR ".join(queries)

    def lookup(self, population_name: str) -> PopulationFrequency:
        """
        Get PubMed citation count for a population term.

        Args:
            population_name: Cell population name (e.g., "CD4+ T cells")

        Returns:
            PopulationFrequency with count and metadata
        """
        # Check cache first
        cache_key = population_name.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build and execute query
        query = self._build_query(population_name)
        start_time = time.time()

        params = {
            "db": "pubmed",
            "term": query,
            "rettype": "count",
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            if HAS_HTTPX:
                with httpx.Client() as client:
                    response = client.get(self.ESEARCH_URL, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
            else:
                # Fallback to urllib
                query_string = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
                url = f"{self.ESEARCH_URL}?{query_string}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())

            count = int(data.get("esearchresult", {}).get("count", 0))
            fetch_time = time.time() - start_time

            result = PopulationFrequency(
                term=population_name,
                pubmed_count=count,
                log_count=_safe_log(count),
                query_used=query,
                fetch_time=fetch_time,
            )

            # Cache and save
            self._cache[cache_key] = result
            self._save_cache()

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return result

        except Exception as e:
            logger.warning(f"PubMed lookup failed for '{population_name}': {e}")
            return PopulationFrequency(
                term=population_name,
                pubmed_count=0,
                log_count=0.0,
                query_used=query,
                fetch_time=time.time() - start_time,
            )

    def lookup_batch(self, population_names: list[str]) -> dict[str, PopulationFrequency]:
        """Look up frequencies for multiple populations."""
        results = {}
        for name in population_names:
            results[name] = self.lookup(name)
        return results


def _safe_log(x: int | float) -> float:
    """Compute log(x) safely, returning 0 for x <= 0."""
    import math
    return math.log(x + 1) if x > 0 else 0.0


# =============================================================================
# FREQUENCY CORRELATION ANALYSIS
# =============================================================================


@dataclass
class CorrelationResult:
    """Result of frequency-performance correlation analysis."""

    r_squared: float
    pearson_r: float
    p_value: float
    n_samples: int
    interpretation: str
    data_points: list[tuple[str, float, float]]  # (name, log_freq, score)


class FrequencyCorrelation:
    """
    Analyze correlation between term frequency and model performance.

    If R^2 > 0.8, the frequency hypothesis is supported (it's just memory).
    If R^2 is low, the reasoning gap hypothesis stands.
    """

    def __init__(self, pubmed_lookup: PubMedFrequencyLookup | None = None):
        """
        Initialize correlation analyzer.

        Args:
            pubmed_lookup: PubMed lookup instance (created if not provided)
        """
        self.lookup = pubmed_lookup or PubMedFrequencyLookup()

    def analyze(
        self,
        population_scores: dict[str, float],
        frequency_threshold: float = 0.8,
    ) -> CorrelationResult:
        """
        Analyze correlation between population frequency and model scores.

        Args:
            population_scores: Dict of population_name -> F1 score
            frequency_threshold: R^2 threshold above which frequency explains performance

        Returns:
            CorrelationResult with statistical analysis
        """
        # Gather frequency data
        data_points: list[tuple[str, float, float]] = []

        for name, score in population_scores.items():
            freq = self.lookup.lookup(name)
            data_points.append((name, freq.log_count, score))

        if len(data_points) < 3:
            return CorrelationResult(
                r_squared=0.0,
                pearson_r=0.0,
                p_value=1.0,
                n_samples=len(data_points),
                interpretation="Insufficient data for correlation analysis",
                data_points=data_points,
            )

        # Compute correlation
        frequencies = [dp[1] for dp in data_points]
        scores = [dp[2] for dp in data_points]

        r, p_value = _pearson_correlation(frequencies, scores)
        r_squared = r ** 2

        # Interpret results
        if r_squared > frequency_threshold:
            interpretation = (
                f"FREQUENCY HYPOTHESIS SUPPORTED (R²={r_squared:.3f} > {frequency_threshold}): "
                "Performance is strongly correlated with term frequency. "
                "Model may be relying on memorization rather than reasoning."
            )
        elif r_squared > 0.5:
            interpretation = (
                f"MIXED EVIDENCE (R²={r_squared:.3f}): "
                "Moderate correlation suggests both frequency and reasoning play roles."
            )
        else:
            interpretation = (
                f"REASONING HYPOTHESIS SUPPORTED (R²={r_squared:.3f}): "
                "Weak correlation with frequency suggests failures are due to "
                "reasoning deficits, not lack of training data."
            )

        return CorrelationResult(
            r_squared=r_squared,
            pearson_r=r,
            p_value=p_value,
            n_samples=len(data_points),
            interpretation=interpretation,
            data_points=data_points,
        )


def _pearson_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return 0.0, 1.0

    r = numerator / (denom_x * denom_y)

    # Compute p-value using t-distribution approximation
    import math
    if abs(r) == 1.0:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
        # Two-tailed p-value (approximation)
        p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), n - 2))

    return r, p_value


def _t_cdf(t: float, df: int) -> float:
    """Approximate t-distribution CDF using normal approximation for large df."""
    import math
    # For df > 30, t-distribution approaches normal
    if df > 30:
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))
    # Simple approximation for smaller df
    return 0.5 * (1 + math.erf(t / math.sqrt(2)))


# =============================================================================
# ALIEN CELL INJECTION TEST
# =============================================================================


ALIEN_CELL_NAMES = [
    "Glorp Cells",
    "Blixon Population",
    "Zythroid Subset",
    "Flumox Lineage",
    "Quandrix Cells",
    "Vexil Population",
    "Droflex Subset",
    "Mortix Lineage",
    "Splynk Cells",
    "Throbin Population",
]


@dataclass
class AlienCellMapping:
    """Mapping between real and alien cell names."""

    original_name: str
    alien_name: str
    marker_logic: str
    depth: int


@dataclass
class AlienCellTestCase:
    """A test case with alien cell names injected."""

    original_test_case: TestCase
    modified_test_case: TestCase
    mappings: list[AlienCellMapping]
    mapping_context: str  # Context explaining the alien names


class AlienCellTest:
    """
    The "Alien Cell" injection test.

    Takes a real gating hierarchy and replaces population names with
    nonsense words while preserving the marker logic. If the model
    can still correctly identify populations based on markers alone,
    it demonstrates genuine reasoning. If it fails, it proves reliance
    on token associations.

    Example:
    - Original: "CD3+ CD4+ CD25+ -> T-Regulatory Cells"
    - Alien: "CD3+ CD4+ CD25+ -> Glorp Cells"

    Prediction: If model reasons from markers, it should identify
    "Glorp Cells" correctly. If it relies on the token "T-Reg",
    it will fail.
    """

    def __init__(
        self,
        alien_names: list[str] | None = None,
        preserve_critical_gates: bool = True,
    ):
        """
        Initialize Alien Cell test.

        Args:
            alien_names: List of nonsense names to use (default: ALIEN_CELL_NAMES)
            preserve_critical_gates: Whether to preserve QC gate names (Live, Singlets)
        """
        self.alien_names = alien_names or ALIEN_CELL_NAMES.copy()
        self.preserve_critical_gates = preserve_critical_gates
        self._name_index = 0

        # Gates that should NOT be renamed (QC gates)
        self.preserved_patterns = [
            r"^All\s*Events$",
            r"^Time$",
            r"^Singlet",
            r"^Live",
            r"^Dead",
            r"^Debris",
            r"^Doublet",
        ]

    def _should_preserve(self, gate_name: str) -> bool:
        """Check if a gate name should be preserved."""
        if not self.preserve_critical_gates:
            return False
        for pattern in self.preserved_patterns:
            if re.match(pattern, gate_name, re.IGNORECASE):
                return True
        return False

    def _get_next_alien_name(self) -> str:
        """Get the next alien name from the pool."""
        if self._name_index >= len(self.alien_names):
            # Generate more names if needed
            suffix = self._name_index // len(ALIEN_CELL_NAMES)
            base_name = ALIEN_CELL_NAMES[self._name_index % len(ALIEN_CELL_NAMES)]
            name = f"{base_name} Type-{suffix + 1}"
        else:
            name = self.alien_names[self._name_index]
        self._name_index += 1
        return name

    def create_alien_test_case(
        self,
        test_case: TestCase,
        include_mapping_context: bool = True,
    ) -> AlienCellTestCase:
        """
        Create an alien cell version of a test case.

        Args:
            test_case: Original test case
            include_mapping_context: Whether to include a context explaining mappings

        Returns:
            AlienCellTestCase with modified hierarchy and mappings
        """
        self._name_index = 0  # Reset for consistent naming
        mappings: list[AlienCellMapping] = []

        def transform_gate(gate: GateNode, depth: int = 0) -> GateNode:
            """Recursively transform gate names."""
            if self._should_preserve(gate.name):
                new_name = gate.name
            else:
                alien_name = self._get_next_alien_name()
                marker_logic_str = (
                    gate.marker_logic_str
                    if gate.marker_logic
                    else " ".join(gate.markers)
                )
                mappings.append(AlienCellMapping(
                    original_name=gate.name,
                    alien_name=alien_name,
                    marker_logic=marker_logic_str,
                    depth=depth,
                ))
                new_name = alien_name

            # Transform children
            new_children = [
                transform_gate(child, depth + 1)
                for child in gate.children
            ]

            return GateNode(
                name=new_name,
                markers=gate.markers,
                marker_logic=gate.marker_logic,
                gate_type=gate.gate_type,
                children=new_children,
                is_critical=gate.is_critical,
                notes=gate.notes,
            )

        # Transform the hierarchy
        new_root = transform_gate(test_case.gating_hierarchy.root)
        new_hierarchy = GatingHierarchy(root=new_root)

        # Build mapping context for the prompt
        mapping_context = ""
        if include_mapping_context:
            mapping_context = self._build_mapping_context(mappings, test_case.panel)

        # Create modified test case
        modified_test_case = TestCase(
            test_case_id=f"{test_case.test_case_id}_alien",
            source_type=test_case.source_type,
            omip_id=test_case.omip_id,
            doi=test_case.doi,
            flowrepository_id=test_case.flowrepository_id,
            has_wsp=test_case.has_wsp,
            wsp_validated=test_case.wsp_validated,
            context=test_case.context,
            panel=test_case.panel,
            gating_hierarchy=new_hierarchy,
            validation=test_case.validation,
            metadata=test_case.metadata,
        )

        return AlienCellTestCase(
            original_test_case=test_case,
            modified_test_case=modified_test_case,
            mappings=mappings,
            mapping_context=mapping_context,
        )

    def _build_mapping_context(
        self,
        mappings: list[AlienCellMapping],
        panel: Panel,
    ) -> str:
        """
        Build a context string that explains the alien cell mappings.

        This is included in the prompt to give the model the marker logic
        for each alien population.
        """
        lines = [
            "## Cell Population Definitions",
            "",
            "The following populations are defined by their marker phenotypes:",
            "",
        ]

        for mapping in mappings:
            if mapping.marker_logic:
                lines.append(f"- **{mapping.alien_name}**: {mapping.marker_logic}")
            else:
                lines.append(f"- **{mapping.alien_name}**: (defined by gating position)")

        lines.extend([
            "",
            "Use these definitions to construct the gating hierarchy.",
            "",
        ])

        return "\n".join(lines)


@dataclass
class AlienCellResult:
    """Result of an alien cell test run."""

    test_case_id: str
    original_f1: float
    alien_f1: float
    delta_f1: float
    mappings: list[AlienCellMapping]

    # Per-population results
    population_results: dict[str, dict[str, Any]]

    # Interpretation
    reasoning_score: float  # 0-1, higher = more reasoning, less memorization
    interpretation: str


class AlienCellAnalyzer:
    """
    Analyze results from alien cell tests.

    Compares performance on original vs. alien versions to quantify
    the model's reliance on token associations vs. marker logic.
    """

    def analyze(
        self,
        original_result: Any,  # ScoringResult
        alien_result: Any,  # ScoringResult
        alien_test_case: AlienCellTestCase,
    ) -> AlienCellResult:
        """
        Analyze the difference between original and alien test results.

        Args:
            original_result: Result from original test case
            alien_result: Result from alien cell test case
            alien_test_case: The alien test case with mappings

        Returns:
            AlienCellResult with analysis
        """
        original_f1 = original_result.hierarchy_f1
        alien_f1 = alien_result.hierarchy_f1
        delta_f1 = original_f1 - alien_f1

        # Calculate reasoning score
        # If alien_f1 == original_f1, reasoning_score = 1.0 (pure reasoning)
        # If alien_f1 == 0 and original_f1 > 0, reasoning_score = 0.0 (pure memorization)
        if original_f1 > 0:
            reasoning_score = alien_f1 / original_f1
        else:
            reasoning_score = 1.0 if alien_f1 == 0 else 0.0

        # Generate interpretation
        if delta_f1 < 0.05:
            interpretation = (
                f"REASONING SUPPORTED (delta={delta_f1:.3f}): "
                "Model performs equally well with alien names, suggesting "
                "it reasons from marker logic rather than population name tokens."
            )
        elif delta_f1 < 0.20:
            interpretation = (
                f"MIXED EVIDENCE (delta={delta_f1:.3f}): "
                "Moderate performance drop with alien names suggests "
                "partial reliance on both reasoning and token associations."
            )
        else:
            interpretation = (
                f"MEMORIZATION INDICATED (delta={delta_f1:.3f}): "
                "Large performance drop with alien names suggests "
                "model relies heavily on population name tokens rather than "
                "marker logic for predictions."
            )

        return AlienCellResult(
            test_case_id=alien_test_case.original_test_case.test_case_id,
            original_f1=original_f1,
            alien_f1=alien_f1,
            delta_f1=delta_f1,
            mappings=alien_test_case.mappings,
            population_results={},  # TODO: Add per-population analysis
            reasoning_score=reasoning_score,
            interpretation=interpretation,
        )
