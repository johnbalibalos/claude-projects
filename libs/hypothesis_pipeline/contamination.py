"""
Contamination and memorization detection for LLM evaluation.

Provides tools to detect whether models have memorized benchmark data:
- Completion tests (can model complete truncated inputs?)
- N-gram overlap analysis
- Verbatim reproduction detection
- Paraphrase robustness testing
- Dynamic test case generation for contamination-resistant evaluation
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np


# =============================================================================
# PROTOCOLS
# =============================================================================


class ModelClient(Protocol):
    """Protocol for model clients used in contamination testing."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion from the model."""
        ...


# =============================================================================
# MEMORIZATION DETECTION
# =============================================================================


@dataclass
class MemorizationResult:
    """Result of memorization test for a single test case."""

    test_case_id: str
    completion_overlap: float  # 0-1, how much of truncated portion was reproduced
    exact_match: bool  # Whether completion matches exactly
    high_confidence_memorization: bool  # Strong evidence of memorization
    partial_text_provided: str
    model_completion: str
    expected_completion: str
    n_gram_matches: dict[int, float]  # N-gram overlap scores

    def __str__(self) -> str:
        status = "MEMORIZED" if self.high_confidence_memorization else "likely novel"
        return f"[{self.test_case_id}] {status}: overlap={self.completion_overlap:.2%}"


@dataclass
class ContaminationReport:
    """Aggregate contamination report for a test set."""

    n_test_cases: int
    n_memorized: int
    memorization_rate: float
    high_risk_cases: list[str]  # IDs of likely contaminated cases
    results: list[MemorizationResult]
    recommendations: list[str]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CONTAMINATION DETECTION REPORT",
            "=" * 60,
            f"Test cases analyzed: {self.n_test_cases}",
            f"Likely memorized: {self.n_memorized} ({self.memorization_rate:.1%})",
            "",
        ]

        if self.high_risk_cases:
            lines.append("HIGH RISK CASES (consider removing):")
            for case_id in self.high_risk_cases[:10]:
                lines.append(f"  - {case_id}")
            if len(self.high_risk_cases) > 10:
                lines.append(f"  ... and {len(self.high_risk_cases) - 10} more")
            lines.append("")

        if self.recommendations:
            lines.append("RECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class MemorizationDetector:
    """
    Detects whether a model has memorized test cases from training data.

    Uses multiple signals:
    1. Completion test: Can model complete truncated ground truth?
    2. N-gram overlap: Does model output contain rare n-grams from ground truth?
    3. Verbatim detection: Are there exact substring matches?
    """

    def __init__(
        self,
        model_client: ModelClient,
        truncation_ratio: float = 0.5,
        overlap_threshold: float = 0.7,
        n_gram_sizes: Sequence[int] = (3, 4, 5),
    ):
        """
        Initialize detector.

        Args:
            model_client: Client for model inference
            truncation_ratio: Portion of ground truth to provide (0.5 = first half)
            overlap_threshold: Overlap score above which indicates memorization
            n_gram_sizes: N-gram sizes to analyze
        """
        self.client = model_client
        self.truncation_ratio = truncation_ratio
        self.overlap_threshold = overlap_threshold
        self.n_gram_sizes = list(n_gram_sizes)

    def check_memorization(
        self,
        test_case_id: str,
        ground_truth: str,
        prompt_template: str = "Complete the following text:\n\n{partial}\n\nContinuation:",
    ) -> MemorizationResult:
        """
        Check if a model has memorized a specific test case.

        Args:
            test_case_id: Identifier for the test case
            ground_truth: The complete ground truth text
            prompt_template: Template for completion prompt

        Returns:
            MemorizationResult with detection signals
        """
        # Truncate ground truth
        truncation_point = int(len(ground_truth) * self.truncation_ratio)
        partial_text = ground_truth[:truncation_point]
        expected_completion = ground_truth[truncation_point:]

        # Get model completion
        prompt = prompt_template.format(partial=partial_text)
        model_completion = self.client.generate(prompt, max_tokens=len(expected_completion) + 100)

        # Compute overlap metrics
        completion_overlap = self._compute_text_overlap(model_completion, expected_completion)
        exact_match = expected_completion.strip() in model_completion
        n_gram_matches = self._compute_ngram_overlap(model_completion, expected_completion)

        # Determine if likely memorized
        high_confidence = (
            completion_overlap >= self.overlap_threshold
            or exact_match
            or any(score >= 0.5 for score in n_gram_matches.values())
        )

        return MemorizationResult(
            test_case_id=test_case_id,
            completion_overlap=completion_overlap,
            exact_match=exact_match,
            high_confidence_memorization=high_confidence,
            partial_text_provided=partial_text,
            model_completion=model_completion,
            expected_completion=expected_completion,
            n_gram_matches=n_gram_matches,
        )

    def _compute_text_overlap(self, prediction: str, reference: str) -> float:
        """Compute character-level overlap between prediction and reference."""
        if not reference:
            return 0.0

        # Normalize texts
        pred_normalized = prediction.lower().strip()
        ref_normalized = reference.lower().strip()

        # Find longest common subsequence ratio
        lcs_length = self._lcs_length(pred_normalized, ref_normalized)
        return lcs_length / len(ref_normalized)

    def _lcs_length(self, s1: str, s2: str) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(s1), len(s2)

        # Optimize for memory
        if m < n:
            s1, s2 = s2, s1
            m, n = n, m

        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, prev

        return prev[n]

    def _compute_ngram_overlap(
        self,
        prediction: str,
        reference: str,
    ) -> dict[int, float]:
        """Compute n-gram overlap for multiple n values."""
        results = {}

        for n in self.n_gram_sizes:
            pred_ngrams = self._get_ngrams(prediction, n)
            ref_ngrams = self._get_ngrams(reference, n)

            if not ref_ngrams:
                results[n] = 0.0
                continue

            # Compute Jaccard similarity
            intersection = len(pred_ngrams & ref_ngrams)
            union = len(pred_ngrams | ref_ngrams)
            results[n] = intersection / union if union > 0 else 0.0

        return results

    def _get_ngrams(self, text: str, n: int) -> set[tuple[str, ...]]:
        """Extract n-grams from text."""
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

    def analyze_test_set(
        self,
        test_cases: list[dict[str, Any]],
        id_field: str = "id",
        ground_truth_field: str = "ground_truth",
        verbose: bool = True,
    ) -> ContaminationReport:
        """
        Analyze an entire test set for contamination.

        Args:
            test_cases: List of test case dictionaries
            id_field: Field name for test case ID
            ground_truth_field: Field name for ground truth
            verbose: Whether to print progress

        Returns:
            ContaminationReport with aggregate results
        """
        results = []
        high_risk_cases = []

        for i, tc in enumerate(test_cases):
            if verbose:
                print(f"Checking [{i+1}/{len(test_cases)}] {tc[id_field]}...", end=" ")

            result = self.check_memorization(
                test_case_id=tc[id_field],
                ground_truth=str(tc[ground_truth_field]),
            )
            results.append(result)

            if result.high_confidence_memorization:
                high_risk_cases.append(result.test_case_id)

            if verbose:
                print("RISK" if result.high_confidence_memorization else "ok")

        # Generate recommendations
        recommendations = self._generate_recommendations(results, high_risk_cases)

        memorization_rate = len(high_risk_cases) / len(test_cases) if test_cases else 0

        return ContaminationReport(
            n_test_cases=len(test_cases),
            n_memorized=len(high_risk_cases),
            memorization_rate=memorization_rate,
            high_risk_cases=high_risk_cases,
            results=results,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        results: list[MemorizationResult],
        high_risk_cases: list[str],
    ) -> list[str]:
        """Generate recommendations based on contamination analysis."""
        recommendations = []

        memorization_rate = len(high_risk_cases) / len(results) if results else 0

        if memorization_rate > 0.3:
            recommendations.append(
                "HIGH CONTAMINATION: >30% of test cases show memorization signals. "
                "Consider using a completely different benchmark or generating synthetic variants."
            )
        elif memorization_rate > 0.1:
            recommendations.append(
                "MODERATE CONTAMINATION: 10-30% contamination detected. "
                "Remove flagged cases and consider paraphrasing remaining test cases."
            )
        elif memorization_rate > 0:
            recommendations.append(
                "LOW CONTAMINATION: <10% contamination detected. "
                "Remove flagged cases for cleaner evaluation."
            )
        else:
            recommendations.append(
                "No clear memorization signals detected. "
                "Test set appears suitable for evaluation."
            )

        if high_risk_cases:
            recommendations.append(
                f"Consider removing {len(high_risk_cases)} flagged test cases "
                "or replacing with paraphrased variants."
            )

        recommendations.append(
            "For robust evaluation, consider generating dynamic test variants "
            "that preserve semantic content but change surface form."
        )

        return recommendations


# =============================================================================
# DYNAMIC TEST CASE GENERATION
# =============================================================================


@dataclass
class TestCaseVariant:
    """A variant of a test case for contamination-resistant evaluation."""

    original_id: str
    variant_id: str
    variant_type: str
    original_content: str
    variant_content: str
    transformation_description: str


class DynamicTestGenerator:
    """
    Generate contamination-resistant test case variants.

    Creates modified versions of test cases that preserve semantic content
    but change surface form to avoid memorization-based shortcuts.
    """

    def __init__(
        self,
        model_client: ModelClient | None = None,
        seed: int | None = None,
    ):
        """
        Initialize generator.

        Args:
            model_client: Optional client for LLM-based transformations
            seed: Random seed for reproducibility
        """
        self.client = model_client
        self.rng = np.random.default_rng(seed)

    def generate_variants(
        self,
        test_case_id: str,
        content: str,
        n_variants: int = 3,
        variant_types: Sequence[str] | None = None,
    ) -> list[TestCaseVariant]:
        """
        Generate multiple variants of a test case.

        Args:
            test_case_id: Original test case ID
            content: Original test case content
            n_variants: Number of variants to generate
            variant_types: Types of variants to generate

        Returns:
            List of test case variants
        """
        if variant_types is None:
            variant_types = ["shuffle", "synonym", "reorder", "noise"]

        variants = []
        types_to_use = self.rng.choice(
            variant_types,
            size=min(n_variants, len(variant_types)),
            replace=False,
        )

        for i, vtype in enumerate(types_to_use):
            variant = self._generate_variant(test_case_id, content, vtype, i)
            if variant:
                variants.append(variant)

        return variants

    def _generate_variant(
        self,
        test_case_id: str,
        content: str,
        variant_type: str,
        index: int,
    ) -> TestCaseVariant | None:
        """Generate a single variant of specified type."""
        variant_id = f"{test_case_id}_v{index}_{variant_type}"

        if variant_type == "shuffle":
            variant_content = self._shuffle_sentences(content)
            description = "Sentences shuffled while preserving overall meaning"

        elif variant_type == "synonym":
            variant_content = self._apply_synonyms(content)
            description = "Key terms replaced with synonyms"

        elif variant_type == "reorder":
            variant_content = self._reorder_list_items(content)
            description = "List items reordered"

        elif variant_type == "noise":
            variant_content = self._add_formatting_noise(content)
            description = "Minor formatting variations added"

        elif variant_type == "paraphrase" and self.client:
            variant_content = self._llm_paraphrase(content)
            description = "LLM-generated paraphrase"

        else:
            return None

        return TestCaseVariant(
            original_id=test_case_id,
            variant_id=variant_id,
            variant_type=variant_type,
            original_content=content,
            variant_content=variant_content,
            transformation_description=description,
        )

    def _shuffle_sentences(self, text: str) -> str:
        """Shuffle sentences in text."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            # Keep first and last, shuffle middle
            middle = sentences[1:-1]
            self.rng.shuffle(middle)
            sentences = [sentences[0]] + list(middle) + [sentences[-1]]
        return " ".join(sentences)

    def _apply_synonyms(self, text: str) -> str:
        """Replace some words with simple variations."""
        # Simple word-level variations (no external dictionary needed)
        replacements = {
            "large": "big",
            "small": "tiny",
            "important": "significant",
            "show": "demonstrate",
            "use": "utilize",
            "make": "create",
            "good": "effective",
            "bad": "poor",
            "many": "numerous",
            "few": "several",
            "high": "elevated",
            "low": "reduced",
            "increase": "rise",
            "decrease": "decline",
        }

        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?;:")
            if word_lower in replacements and self.rng.random() > 0.5:
                # Preserve case and punctuation
                replacement = replacements[word_lower]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve trailing punctuation
                trailing = ""
                for char in reversed(word):
                    if char in ".,!?;:":
                        trailing = char + trailing
                    else:
                        break
                words[i] = replacement + trailing

        return " ".join(words)

    def _reorder_list_items(self, text: str) -> str:
        """Reorder items in lists."""
        # Find bullet points or numbered lists
        lines = text.split("\n")
        list_groups = []
        current_group = []
        current_start = -1

        for i, line in enumerate(lines):
            is_list_item = bool(re.match(r'^\s*[-*•]\s+', line) or re.match(r'^\s*\d+[.)]\s+', line))
            if is_list_item:
                if current_start == -1:
                    current_start = i
                current_group.append((i, line))
            else:
                if current_group:
                    list_groups.append((current_start, current_group.copy()))
                    current_group = []
                    current_start = -1

        if current_group:
            list_groups.append((current_start, current_group.copy()))

        # Shuffle each list group
        for start_idx, group in list_groups:
            items = [item for _, item in group]
            self.rng.shuffle(items)
            for i, item in enumerate(items):
                lines[start_idx + i] = item

        return "\n".join(lines)

    def _add_formatting_noise(self, text: str) -> str:
        """Add minor formatting variations."""
        # Random capitalization variations
        words = text.split()
        for i in range(len(words)):
            if self.rng.random() > 0.95 and words[i].islower():
                words[i] = words[i].capitalize()

        text = " ".join(words)

        # Vary whitespace slightly
        text = re.sub(r'  +', ' ', text)

        return text

    def _llm_paraphrase(self, text: str) -> str:
        """Use LLM to generate paraphrase."""
        if not self.client:
            return text

        prompt = (
            "Paraphrase the following text while preserving its exact meaning. "
            "Use different words and sentence structures but keep all information intact.\n\n"
            f"Original:\n{text}\n\n"
            "Paraphrased version:"
        )

        try:
            return self.client.generate(prompt, max_tokens=len(text) * 2)
        except Exception:
            return text


# =============================================================================
# VERBATIM DETECTION
# =============================================================================


@dataclass
class VerbatimMatch:
    """A verbatim match found in model output."""

    matched_text: str
    source_location: str
    match_length: int
    context: str


def detect_verbatim_reproduction(
    model_output: str,
    reference_corpus: list[str],
    min_match_length: int = 50,
    case_sensitive: bool = False,
) -> list[VerbatimMatch]:
    """
    Detect verbatim text reproduction from a reference corpus.

    Args:
        model_output: The model's output text
        reference_corpus: List of reference documents to check against
        min_match_length: Minimum characters for a match to count
        case_sensitive: Whether matching is case-sensitive

    Returns:
        List of verbatim matches found
    """
    matches = []

    output_check = model_output if case_sensitive else model_output.lower()

    for doc_idx, doc in enumerate(reference_corpus):
        doc_check = doc if case_sensitive else doc.lower()

        # Find all common substrings of sufficient length
        common_substrings = _find_common_substrings(output_check, doc_check, min_match_length)

        for substring in common_substrings:
            # Get context around match in original output
            start_idx = output_check.find(substring)
            context_start = max(0, start_idx - 50)
            context_end = min(len(model_output), start_idx + len(substring) + 50)
            context = model_output[context_start:context_end]

            matches.append(VerbatimMatch(
                matched_text=substring,
                source_location=f"document_{doc_idx}",
                match_length=len(substring),
                context=f"...{context}...",
            ))

    return matches


def _find_common_substrings(s1: str, s2: str, min_length: int) -> list[str]:
    """Find all common substrings of at least min_length characters."""
    # Use suffix array approach for efficiency
    common = []
    len1, len2 = len(s1), len(s2)

    # Build set of all substrings from s2 of min_length
    s2_substrings = set()
    for i in range(len2 - min_length + 1):
        s2_substrings.add(s2[i:i + min_length])

    # Check which appear in s1
    i = 0
    while i < len1 - min_length + 1:
        if s1[i:i + min_length] in s2_substrings:
            # Extend match as far as possible
            match_end = i + min_length
            while match_end < len1 and s1[i:match_end + 1] in s2:
                match_end += 1

            common.append(s1[i:match_end])
            i = match_end
        else:
            i += 1

    return common


# =============================================================================
# TEST SET FINGERPRINTING
# =============================================================================


@dataclass
class TestSetFingerprint:
    """Fingerprint of a test set for tracking contamination over time."""

    fingerprint_id: str
    n_test_cases: int
    content_hash: str
    n_gram_signature: dict[str, int]  # Most common n-grams
    created_at: str
    version: str


def create_test_set_fingerprint(
    test_cases: list[dict[str, Any]],
    content_field: str = "ground_truth",
    version: str = "1.0",
) -> TestSetFingerprint:
    """
    Create a fingerprint of a test set for contamination tracking.

    This fingerprint can be used to:
    1. Track if test sets are being leaked/published
    2. Detect if models have been trained on specific versions
    3. Version test sets for reproducibility

    Args:
        test_cases: List of test case dictionaries
        content_field: Field containing content to fingerprint
        version: Version string for this test set

    Returns:
        TestSetFingerprint object
    """
    from datetime import datetime

    # Combine all content
    all_content = " ".join(str(tc.get(content_field, "")) for tc in test_cases)

    # Create content hash
    content_hash = hashlib.sha256(all_content.encode()).hexdigest()[:16]

    # Extract n-gram signature (most distinctive n-grams)
    words = all_content.lower().split()
    ngram_counts: Counter[tuple[str, ...]] = Counter()
    for n in [3, 4, 5]:
        for i in range(len(words) - n + 1):
            ngram_counts[tuple(words[i:i+n])] += 1

    # Keep top 100 most common
    signature = {
        " ".join(ngram): count
        for ngram, count in ngram_counts.most_common(100)
    }

    fingerprint_id = f"fp_{content_hash}_{version}"

    return TestSetFingerprint(
        fingerprint_id=fingerprint_id,
        n_test_cases=len(test_cases),
        content_hash=content_hash,
        n_gram_signature=signature,
        created_at=datetime.now().isoformat(),
        version=version,
    )


def check_fingerprint_in_output(
    model_output: str,
    fingerprint: TestSetFingerprint,
    threshold: float = 0.1,
) -> tuple[bool, float]:
    """
    Check if model output contains fingerprint n-grams from test set.

    High overlap suggests the model may have seen the test set.

    Args:
        model_output: Model's output text
        fingerprint: Test set fingerprint to check against
        threshold: Minimum overlap ratio to flag as potential contamination

    Returns:
        Tuple of (is_contaminated, overlap_score)
    """
    # Extract n-grams from output
    words = model_output.lower().split()
    output_ngrams = set()
    for n in [3, 4, 5]:
        for i in range(len(words) - n + 1):
            output_ngrams.add(" ".join(words[i:i+n]))

    # Check overlap with fingerprint
    fingerprint_ngrams = set(fingerprint.n_gram_signature.keys())
    overlap = len(output_ngrams & fingerprint_ngrams)
    overlap_ratio = overlap / len(fingerprint_ngrams) if fingerprint_ngrams else 0

    return overlap_ratio >= threshold, overlap_ratio
