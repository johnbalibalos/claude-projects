"""
Test set diversity analysis module.

Analyzes the diversity and coverage of evaluation test sets:
- Feature distribution analysis
- Coverage metrics
- Bias detection
- Stratification recommendations
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike


# =============================================================================
# DIVERSITY METRICS
# =============================================================================


@dataclass
class DistributionAnalysis:
    """Analysis of a single feature distribution."""

    feature_name: str
    n_unique: int
    n_samples: int
    distribution: dict[Any, int]  # value -> count
    entropy: float  # Normalized Shannon entropy (0-1)
    is_balanced: bool
    imbalance_ratio: float  # max_count / min_count
    most_common: list[tuple[Any, int]]
    least_common: list[tuple[Any, int]]


def analyze_distribution(
    values: Sequence[Any],
    feature_name: str = "feature",
    top_k: int = 5,
) -> DistributionAnalysis:
    """
    Analyze the distribution of a categorical feature.

    Args:
        values: List of feature values
        feature_name: Name of the feature
        top_k: Number of most/least common values to report

    Returns:
        DistributionAnalysis with distribution metrics
    """
    counter = Counter(values)
    n_unique = len(counter)
    n_samples = len(values)

    if n_unique == 0:
        return DistributionAnalysis(
            feature_name=feature_name,
            n_unique=0,
            n_samples=n_samples,
            distribution={},
            entropy=0.0,
            is_balanced=True,
            imbalance_ratio=1.0,
            most_common=[],
            least_common=[],
        )

    # Compute entropy
    probs = np.array(list(counter.values())) / n_samples
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(n_unique) if n_unique > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Imbalance ratio
    counts = list(counter.values())
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')

    # Balanced if imbalance ratio < 3 and entropy > 0.8
    is_balanced = imbalance_ratio < 3 and normalized_entropy > 0.8

    most_common = counter.most_common(top_k)
    least_common = counter.most_common()[:-top_k-1:-1] if n_unique > top_k else []

    return DistributionAnalysis(
        feature_name=feature_name,
        n_unique=n_unique,
        n_samples=n_samples,
        distribution=dict(counter),
        entropy=float(normalized_entropy),
        is_balanced=is_balanced,
        imbalance_ratio=float(imbalance_ratio),
        most_common=most_common,
        least_common=least_common,
    )


@dataclass
class DiversityReport:
    """Complete diversity analysis of a test set."""

    n_samples: int
    n_features_analyzed: int
    feature_analyses: dict[str, DistributionAnalysis]
    overall_diversity_score: float  # 0-1, higher is more diverse
    coverage_gaps: list[str]  # Features with poor coverage
    bias_warnings: list[str]  # Potential bias issues
    recommendations: list[str]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "TEST SET DIVERSITY REPORT",
            "=" * 60,
            f"Total samples: {self.n_samples}",
            f"Features analyzed: {self.n_features_analyzed}",
            f"Overall diversity score: {self.overall_diversity_score:.2f}/1.00",
            "",
            "Feature Distributions:",
            "",
        ]

        for name, analysis in self.feature_analyses.items():
            balance_str = "✓" if analysis.is_balanced else "✗"
            lines.append(f"  {name}:")
            lines.append(f"    Unique values: {analysis.n_unique}")
            lines.append(f"    Entropy: {analysis.entropy:.2f}")
            lines.append(f"    Balanced: {balance_str}")
            lines.append(f"    Most common: {analysis.most_common[:3]}")
            lines.append("")

        if self.coverage_gaps:
            lines.append("Coverage Gaps:")
            for gap in self.coverage_gaps:
                lines.append(f"  - {gap}")
            lines.append("")

        if self.bias_warnings:
            lines.append("Bias Warnings:")
            for warning in self.bias_warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")

        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  → {rec}")

        return "\n".join(lines)


def analyze_test_set_diversity(
    test_cases: list[dict[str, Any]],
    feature_fields: list[str] | None = None,
    custom_extractors: dict[str, Callable[[dict], Any]] | None = None,
) -> DiversityReport:
    """
    Analyze diversity of a test set across multiple dimensions.

    Args:
        test_cases: List of test case dictionaries
        feature_fields: Fields to analyze (default: auto-detect)
        custom_extractors: Custom functions to extract features

    Returns:
        DiversityReport with comprehensive analysis

    Example:
        >>> test_cases = [
        ...     {"id": 1, "complexity": "simple", "domain": "biology"},
        ...     {"id": 2, "complexity": "complex", "domain": "chemistry"},
        ... ]
        >>> report = analyze_test_set_diversity(test_cases, ["complexity", "domain"])
        >>> print(report.summary())
    """
    if not test_cases:
        return DiversityReport(
            n_samples=0,
            n_features_analyzed=0,
            feature_analyses={},
            overall_diversity_score=0.0,
            coverage_gaps=[],
            bias_warnings=[],
            recommendations=["Add test cases to the test set"],
        )

    n_samples = len(test_cases)

    # Auto-detect features if not specified
    if feature_fields is None:
        # Find fields that have limited unique values (likely categorical)
        candidate_fields = set()
        for tc in test_cases[:100]:  # Sample first 100
            candidate_fields.update(tc.keys())

        feature_fields = []
        for field in candidate_fields:
            values = [tc.get(field) for tc in test_cases if field in tc]
            # Skip if too many unique values or all same
            n_unique = len(set(str(v) for v in values))
            if 2 <= n_unique <= n_samples * 0.3:
                feature_fields.append(field)

    # Add custom extractors
    extractors: dict[str, Callable] = {}
    if custom_extractors:
        extractors.update(custom_extractors)

    for field in feature_fields:
        if field not in extractors:
            extractors[field] = lambda tc, f=field: tc.get(f)

    # Analyze each feature
    feature_analyses = {}
    for name, extractor in extractors.items():
        values = [extractor(tc) for tc in test_cases]
        values = [v for v in values if v is not None]
        if values:
            feature_analyses[name] = analyze_distribution(values, name)

    # Compute overall diversity score
    if feature_analyses:
        entropies = [a.entropy for a in feature_analyses.values()]
        balance_scores = [1.0 if a.is_balanced else 0.5 for a in feature_analyses.values()]
        overall_diversity = np.mean(entropies) * 0.7 + np.mean(balance_scores) * 0.3
    else:
        overall_diversity = 0.0

    # Identify coverage gaps
    coverage_gaps = []
    for name, analysis in feature_analyses.items():
        if analysis.n_unique <= 2:
            coverage_gaps.append(f"{name}: Only {analysis.n_unique} unique values")
        elif analysis.entropy < 0.5:
            coverage_gaps.append(f"{name}: Low entropy ({analysis.entropy:.2f}), dominated by few values")

    # Identify bias warnings
    bias_warnings = []
    for name, analysis in feature_analyses.items():
        if analysis.imbalance_ratio > 5:
            most = analysis.most_common[0] if analysis.most_common else ("unknown", 0)
            bias_warnings.append(
                f"{name}: Highly imbalanced ({analysis.imbalance_ratio:.1f}x), "
                f"'{most[0]}' dominates with {most[1]} samples"
            )

    # Generate recommendations
    recommendations = []
    if overall_diversity < 0.5:
        recommendations.append("Overall diversity is low. Consider adding more varied test cases.")

    for name, analysis in feature_analyses.items():
        if not analysis.is_balanced:
            least = analysis.least_common[0] if analysis.least_common else None
            if least:
                recommendations.append(
                    f"Add more test cases with {name}='{least[0]}' "
                    f"(currently only {least[1]} samples)"
                )

    if n_samples < 30:
        recommendations.append(
            f"Test set has only {n_samples} samples. "
            "Consider expanding to at least 30 for statistical reliability."
        )

    return DiversityReport(
        n_samples=n_samples,
        n_features_analyzed=len(feature_analyses),
        feature_analyses=feature_analyses,
        overall_diversity_score=float(overall_diversity),
        coverage_gaps=coverage_gaps,
        bias_warnings=bias_warnings,
        recommendations=recommendations,
    )


# =============================================================================
# COVERAGE ANALYSIS
# =============================================================================


@dataclass
class CoverageCell:
    """Coverage information for a combination of feature values."""

    feature_values: dict[str, Any]
    count: int
    expected_count: float
    coverage_ratio: float  # count / expected


@dataclass
class CrossFeatureCoverage:
    """Cross-feature coverage analysis."""

    features: list[str]
    total_combinations: int
    covered_combinations: int
    coverage_rate: float
    cells: list[CoverageCell]
    missing_combinations: list[dict[str, Any]]


def analyze_cross_feature_coverage(
    test_cases: list[dict[str, Any]],
    feature_fields: list[str],
    max_combinations: int = 1000,
) -> CrossFeatureCoverage:
    """
    Analyze coverage across combinations of features.

    Args:
        test_cases: List of test case dictionaries
        feature_fields: Fields to analyze together
        max_combinations: Maximum combinations to report

    Returns:
        CrossFeatureCoverage with combination analysis
    """
    n_samples = len(test_cases)

    # Get unique values for each feature
    feature_values: dict[str, set] = defaultdict(set)
    for tc in test_cases:
        for field in feature_fields:
            if field in tc:
                feature_values[field].add(tc[field])

    # Convert to lists for iteration
    feature_value_lists = {
        f: list(values) for f, values in feature_values.items()
    }

    # Count actual combinations
    combination_counts: Counter[tuple] = Counter()
    for tc in test_cases:
        combo = tuple(tc.get(f) for f in feature_fields)
        combination_counts[combo] += 1

    # Generate all possible combinations
    from itertools import product
    all_combos = list(product(*[feature_value_lists.get(f, [None]) for f in feature_fields]))
    total_combinations = len(all_combos)

    # Expected count per combination (uniform)
    expected_per_combo = n_samples / total_combinations if total_combinations > 0 else 0

    # Build cells
    cells = []
    missing = []

    for combo in all_combos[:max_combinations]:
        count = combination_counts.get(combo, 0)
        feature_dict = dict(zip(feature_fields, combo))

        if count > 0:
            cells.append(CoverageCell(
                feature_values=feature_dict,
                count=count,
                expected_count=expected_per_combo,
                coverage_ratio=count / expected_per_combo if expected_per_combo > 0 else 0,
            ))
        else:
            missing.append(feature_dict)

    covered = sum(1 for c in combination_counts.values() if c > 0)
    coverage_rate = covered / total_combinations if total_combinations > 0 else 0

    return CrossFeatureCoverage(
        features=feature_fields,
        total_combinations=total_combinations,
        covered_combinations=covered,
        coverage_rate=float(coverage_rate),
        cells=cells,
        missing_combinations=missing[:50],  # Limit reported missing
    )


# =============================================================================
# STRATIFICATION
# =============================================================================


@dataclass
class StratificationResult:
    """Result of stratification analysis."""

    stratification_field: str
    n_strata: int
    strata_counts: dict[Any, int]
    is_stratified: bool
    stratification_quality: float  # 0-1
    suggested_sample_sizes: dict[Any, int]


def analyze_stratification(
    test_cases: list[dict[str, Any]],
    stratification_field: str,
    target_distribution: dict[Any, float] | None = None,
) -> StratificationResult:
    """
    Analyze how well test set is stratified by a given field.

    Args:
        test_cases: List of test case dictionaries
        stratification_field: Field to stratify by
        target_distribution: Target proportions (default: uniform)

    Returns:
        StratificationResult with stratification quality metrics
    """
    values = [tc.get(stratification_field) for tc in test_cases]
    values = [v for v in values if v is not None]

    if not values:
        return StratificationResult(
            stratification_field=stratification_field,
            n_strata=0,
            strata_counts={},
            is_stratified=False,
            stratification_quality=0.0,
            suggested_sample_sizes={},
        )

    counter = Counter(values)
    n_strata = len(counter)
    n_total = len(values)

    # Target distribution (uniform if not specified)
    if target_distribution is None:
        target_distribution = {k: 1.0 / n_strata for k in counter.keys()}

    # Normalize target distribution
    total_target = sum(target_distribution.values())
    target_distribution = {k: v / total_target for k, v in target_distribution.items()}

    # Compute actual proportions
    actual_distribution = {k: v / n_total for k, v in counter.items()}

    # Stratification quality: 1 - mean absolute deviation from target
    deviations = []
    for k in set(target_distribution.keys()) | set(actual_distribution.keys()):
        target = target_distribution.get(k, 0)
        actual = actual_distribution.get(k, 0)
        deviations.append(abs(target - actual))

    mean_deviation = np.mean(deviations) if deviations else 0
    quality = 1 - min(1, mean_deviation * 2)  # Scale deviation

    # Is stratified if quality > 0.8
    is_stratified = quality > 0.8

    # Suggested sample sizes to achieve target distribution
    suggested = {}
    for k, target_prop in target_distribution.items():
        current = counter.get(k, 0)
        target_count = int(n_total * target_prop)
        if current < target_count:
            suggested[k] = target_count - current

    return StratificationResult(
        stratification_field=stratification_field,
        n_strata=n_strata,
        strata_counts=dict(counter),
        is_stratified=is_stratified,
        stratification_quality=float(quality),
        suggested_sample_sizes=suggested,
    )


# =============================================================================
# CONTENT DIVERSITY
# =============================================================================


@dataclass
class ContentDiversityResult:
    """Content diversity analysis result."""

    n_samples: int
    avg_length: float
    length_std: float
    vocabulary_size: int
    type_token_ratio: float  # Vocabulary richness
    avg_unique_words_per_sample: float
    duplicate_rate: float  # Fraction of near-duplicate samples


def analyze_content_diversity(
    texts: Sequence[str],
    similarity_threshold: float = 0.9,
) -> ContentDiversityResult:
    """
    Analyze lexical diversity of text content.

    Args:
        texts: List of text samples
        similarity_threshold: Threshold for near-duplicate detection

    Returns:
        ContentDiversityResult with diversity metrics
    """
    if not texts:
        return ContentDiversityResult(
            n_samples=0,
            avg_length=0,
            length_std=0,
            vocabulary_size=0,
            type_token_ratio=0,
            avg_unique_words_per_sample=0,
            duplicate_rate=0,
        )

    # Basic length stats
    lengths = [len(t) for t in texts]
    avg_length = float(np.mean(lengths))
    length_std = float(np.std(lengths))

    # Vocabulary analysis
    all_words = []
    unique_per_sample = []

    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
        unique_per_sample.append(len(set(words)))

    vocabulary_size = len(set(all_words))
    type_token_ratio = vocabulary_size / len(all_words) if all_words else 0
    avg_unique = float(np.mean(unique_per_sample)) if unique_per_sample else 0

    # Near-duplicate detection (simplified)
    duplicates = 0
    seen_hashes = set()

    for text in texts:
        # Simple fingerprint based on word set
        words = frozenset(text.lower().split())
        word_hash = hash(words)

        if word_hash in seen_hashes:
            duplicates += 1
        seen_hashes.add(word_hash)

    duplicate_rate = duplicates / len(texts) if texts else 0

    return ContentDiversityResult(
        n_samples=len(texts),
        avg_length=avg_length,
        length_std=length_std,
        vocabulary_size=vocabulary_size,
        type_token_ratio=float(type_token_ratio),
        avg_unique_words_per_sample=avg_unique,
        duplicate_rate=float(duplicate_rate),
    )


# =============================================================================
# DIFFICULTY DISTRIBUTION
# =============================================================================


@dataclass
class DifficultyAnalysis:
    """Analysis of test case difficulty distribution."""

    difficulty_scores: list[float]
    mean_difficulty: float
    std_difficulty: float
    easy_fraction: float  # score < 0.3
    medium_fraction: float  # 0.3 <= score < 0.7
    hard_fraction: float  # score >= 0.7
    is_well_distributed: bool


def analyze_difficulty_distribution(
    difficulty_scores: Sequence[float],
    target_easy: float = 0.3,
    target_medium: float = 0.4,
    target_hard: float = 0.3,
) -> DifficultyAnalysis:
    """
    Analyze distribution of test case difficulties.

    Args:
        difficulty_scores: Difficulty scores (0-1, higher = harder)
        target_easy: Target fraction of easy cases
        target_medium: Target fraction of medium cases
        target_hard: Target fraction of hard cases

    Returns:
        DifficultyAnalysis with distribution metrics
    """
    scores = np.array(difficulty_scores)
    n = len(scores)

    if n == 0:
        return DifficultyAnalysis(
            difficulty_scores=[],
            mean_difficulty=0,
            std_difficulty=0,
            easy_fraction=0,
            medium_fraction=0,
            hard_fraction=0,
            is_well_distributed=False,
        )

    mean_diff = float(np.mean(scores))
    std_diff = float(np.std(scores))

    easy = np.sum(scores < 0.3) / n
    medium = np.sum((scores >= 0.3) & (scores < 0.7)) / n
    hard = np.sum(scores >= 0.7) / n

    # Check if distribution is close to target
    deviation = (
        abs(easy - target_easy) +
        abs(medium - target_medium) +
        abs(hard - target_hard)
    )
    is_well_distributed = deviation < 0.3

    return DifficultyAnalysis(
        difficulty_scores=list(scores),
        mean_difficulty=mean_diff,
        std_difficulty=std_diff,
        easy_fraction=float(easy),
        medium_fraction=float(medium),
        hard_fraction=float(hard),
        is_well_distributed=is_well_distributed,
    )


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================


@dataclass
class ComprehensiveDiversityAnalysis:
    """Complete diversity analysis combining all metrics."""

    feature_diversity: DiversityReport
    content_diversity: ContentDiversityResult | None
    difficulty_analysis: DifficultyAnalysis | None
    cross_coverage: CrossFeatureCoverage | None
    overall_score: float
    executive_summary: str


def comprehensive_diversity_analysis(
    test_cases: list[dict[str, Any]],
    feature_fields: list[str] | None = None,
    content_field: str | None = None,
    difficulty_field: str | None = None,
    cross_coverage_fields: list[str] | None = None,
) -> ComprehensiveDiversityAnalysis:
    """
    Run comprehensive diversity analysis on test set.

    Args:
        test_cases: List of test case dictionaries
        feature_fields: Categorical fields to analyze
        content_field: Text field for content diversity
        difficulty_field: Field containing difficulty scores
        cross_coverage_fields: Fields for cross-coverage analysis

    Returns:
        ComprehensiveDiversityAnalysis with all metrics
    """
    # Feature diversity
    feature_diversity = analyze_test_set_diversity(test_cases, feature_fields)

    # Content diversity
    content_diversity = None
    if content_field:
        texts = [tc.get(content_field, "") for tc in test_cases]
        texts = [t for t in texts if t]
        if texts:
            content_diversity = analyze_content_diversity(texts)

    # Difficulty analysis
    difficulty_analysis = None
    if difficulty_field:
        scores = [tc.get(difficulty_field) for tc in test_cases]
        scores = [s for s in scores if s is not None]
        if scores:
            difficulty_analysis = analyze_difficulty_distribution(scores)

    # Cross-coverage
    cross_coverage = None
    if cross_coverage_fields and len(cross_coverage_fields) >= 2:
        cross_coverage = analyze_cross_feature_coverage(test_cases, cross_coverage_fields)

    # Compute overall score
    scores = [feature_diversity.overall_diversity_score]

    if content_diversity:
        # Content diversity score based on type-token ratio and duplicate rate
        content_score = content_diversity.type_token_ratio * (1 - content_diversity.duplicate_rate)
        scores.append(min(1.0, content_score * 2))

    if difficulty_analysis:
        scores.append(1.0 if difficulty_analysis.is_well_distributed else 0.5)

    if cross_coverage:
        scores.append(cross_coverage.coverage_rate)

    overall_score = float(np.mean(scores))

    # Generate executive summary
    summary_parts = [f"Test set contains {len(test_cases)} samples."]

    if overall_score >= 0.8:
        summary_parts.append("Overall diversity is GOOD.")
    elif overall_score >= 0.6:
        summary_parts.append("Overall diversity is MODERATE.")
    else:
        summary_parts.append("Overall diversity is LOW - improvements recommended.")

    if feature_diversity.bias_warnings:
        summary_parts.append(f"Found {len(feature_diversity.bias_warnings)} potential bias issues.")

    if content_diversity and content_diversity.duplicate_rate > 0.1:
        summary_parts.append(f"Warning: {content_diversity.duplicate_rate:.1%} near-duplicate content detected.")

    executive_summary = " ".join(summary_parts)

    return ComprehensiveDiversityAnalysis(
        feature_diversity=feature_diversity,
        content_diversity=content_diversity,
        difficulty_analysis=difficulty_analysis,
        cross_coverage=cross_coverage,
        overall_score=overall_score,
        executive_summary=executive_summary,
    )
