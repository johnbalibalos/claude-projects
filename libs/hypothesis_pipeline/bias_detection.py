"""
Bias detection and mitigation for LLM-as-Judge evaluation.

Implements bias detection based on the CALM framework from:
"Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge"
https://arxiv.org/abs/2410.02736

Supported biases:
1. Position Bias - preference based on response order
2. Verbosity Bias - favoring longer responses
3. Self-Enhancement Bias - favoring own model's outputs
4. Authority Bias - influenced by citations/credentials
5. Sentiment Bias - preference for positive/negative tone
6. Fallacy Oversight Bias - ignoring logical errors
7. Bandwagon Effect Bias - influenced by popularity claims
8. Beauty Bias - influenced by formatting/presentation
9. Familiarity Bias - favoring lower perplexity text
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BiasReport:
    """Comprehensive bias analysis report for an evaluation."""

    # Position bias
    position_bias_detected: bool = False
    position_bias_score_delta: float = 0.0

    # Verbosity bias
    verbosity_ratio: float = 1.0
    verbosity_warning: str | None = None

    # Self-enhancement bias
    self_enhancement_risk: bool = False
    judge_model_family: str | None = None
    response_model_family: str | None = None

    # Authority bias
    authority_markers_found: int = 0
    authority_markers_stripped: bool = False

    # Sentiment bias
    response_sentiment_score: float = 0.0
    sentiment_bias_correlation: float | None = None

    # Bandwagon bias
    bandwagon_markers_found: int = 0

    # Beauty/formatting bias
    formatting_score: float = 0.0

    # Overall
    total_bias_flags: int = 0
    bias_warnings: list[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        """Add a bias warning and increment flag count."""
        self.bias_warnings.append(warning)
        self.total_bias_flags += 1

    def summary(self) -> str:
        """Return human-readable summary."""
        if not self.bias_warnings:
            return "No significant biases detected."

        lines = [f"Detected {self.total_bias_flags} potential bias(es):"]
        for w in self.bias_warnings:
            lines.append(f"  - {w}")
        return "\n".join(lines)


@dataclass
class SelfEnhancementAudit:
    """Audit result for self-enhancement bias detection."""

    judge_model: str
    response_source_model: str | None
    judge_family: str | None
    response_family: str | None
    is_same_family: bool
    bias_warning: str | None


# =============================================================================
# MODEL FAMILY DETECTION
# =============================================================================

MODEL_FAMILIES: dict[str, list[str]] = {
    "openai": [
        "gpt-3.5",
        "gpt-4",
        "gpt-4o",
        "gpt-4-turbo",
        "o1",
        "o3",
        "davinci",
        "text-davinci",
    ],
    "anthropic": [
        "claude-2",
        "claude-3",
        "claude-instant",
        "claude-opus",
        "claude-sonnet",
        "claude-haiku",
    ],
    "google": [
        "gemini",
        "gemini-pro",
        "gemini-ultra",
        "gemini-flash",
        "palm",
        "bard",
    ],
    "meta": [
        "llama",
        "llama-2",
        "llama-3",
        "code-llama",
    ],
    "mistral": [
        "mistral",
        "mixtral",
        "mistral-large",
        "mistral-medium",
    ],
    "cohere": [
        "command",
        "command-r",
        "command-light",
    ],
}


def detect_model_family(model_name: str) -> str | None:
    """Detect the model family from a model name string."""
    if not model_name:
        return None

    model_lower = model_name.lower()

    for family, patterns in MODEL_FAMILIES.items():
        for pattern in patterns:
            if pattern in model_lower:
                return family

    return None


def detect_self_enhancement_risk(
    judge_model: str,
    response_source: str | None,
) -> SelfEnhancementAudit:
    """
    Detect potential self-enhancement bias risk.

    Self-enhancement bias occurs when an LLM judge systematically
    assigns higher scores to outputs from the same model family.

    Args:
        judge_model: Name/identifier of the judge model
        response_source: Name/identifier of the model that generated the response

    Returns:
        SelfEnhancementAudit with risk assessment
    """
    judge_family = detect_model_family(judge_model)
    source_family = detect_model_family(response_source) if response_source else None

    is_same = (
        judge_family is not None
        and source_family is not None
        and judge_family == source_family
    )

    warning = None
    if is_same:
        warning = (
            f"Self-enhancement risk: judge ({judge_model}) and response "
            f"({response_source}) are from the same model family ({judge_family})"
        )

    return SelfEnhancementAudit(
        judge_model=judge_model,
        response_source_model=response_source,
        judge_family=judge_family,
        response_family=source_family,
        is_same_family=is_same,
        bias_warning=warning,
    )


# =============================================================================
# AUTHORITY BIAS
# =============================================================================

AUTHORITY_PATTERNS: list[tuple[str, str]] = [
    # Academic credentials
    (r"\b(Dr\.|Professor|Prof\.|PhD|Ph\.D\.)\s+\w+", "[EXPERT]"),
    # Prestigious institutions
    (
        r"\b(Harvard|Stanford|MIT|Oxford|Cambridge|Yale|Princeton|Berkeley)\b",
        "[INSTITUTION]",
    ),
    # Scientific publications
    (r"\b(Nature|Science|NEJM|Lancet|Cell|PNAS)\b", "[JOURNAL]"),
    # Appeals to authority
    (r"(according to (leading )?experts?)", "[AUTHORITY_CLAIM]"),
    (r"(studies (have )?show(n|s)?)", "[STUDY_CLAIM]"),
    (r"(research (has )?(proven|demonstrated|shown))", "[RESEARCH_CLAIM]"),
    (r"(peer[- ]reviewed)", "[PEER_REVIEW]"),
    (r"(Nobel (Prize )?(laureate|winner))", "[NOBEL]"),
    # Citation patterns
    (r"\[\d+\]", ""),  # Remove citation numbers
    (r"\(\w+\s+et\s+al\.,?\s*\d{4}\)", "[CITATION]"),  # (Author et al., 2024)
    (r"\(\w+,?\s*\d{4}\)", "[CITATION]"),  # (Author, 2024)
]

URL_PATTERN = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+", re.IGNORECASE)


def strip_authority_markers(
    text: str,
    strip_urls: bool = True,
    replace_with_placeholder: bool = True,
) -> tuple[str, int]:
    """
    Remove authority-biasing elements from text.

    Args:
        text: Input text to process
        strip_urls: Whether to strip/replace URLs
        replace_with_placeholder: If True, replace with placeholders; if False, remove

    Returns:
        Tuple of (processed_text, count_of_markers_found)
    """
    markers_found = 0
    result = text

    # Strip URLs
    if strip_urls:
        url_matches = URL_PATTERN.findall(result)
        markers_found += len(url_matches)
        replacement = "[URL]" if replace_with_placeholder else ""
        result = URL_PATTERN.sub(replacement, result)

    # Apply authority patterns
    for pattern, replacement in AUTHORITY_PATTERNS:
        if not replace_with_placeholder and replacement.startswith("["):
            replacement = ""

        matches = re.findall(pattern, result, flags=re.IGNORECASE)
        markers_found += len(matches)
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Clean up multiple spaces
    result = re.sub(r"\s+", " ", result).strip()

    return result, markers_found


def detect_authority_markers(text: str) -> dict[str, Any]:
    """
    Detect authority markers without stripping them.

    Returns:
        Dict with marker counts by category
    """
    results = {
        "urls": len(URL_PATTERN.findall(text)),
        "credentials": 0,
        "institutions": 0,
        "journals": 0,
        "citations": 0,
        "authority_claims": 0,
        "total": 0,
    }

    # Credentials
    results["credentials"] = len(
        re.findall(r"\b(Dr\.|Professor|Prof\.|PhD|Ph\.D\.)\s+\w+", text, re.IGNORECASE)
    )

    # Institutions
    results["institutions"] = len(
        re.findall(
            r"\b(Harvard|Stanford|MIT|Oxford|Cambridge|Yale|Princeton|Berkeley)\b",
            text,
            re.IGNORECASE,
        )
    )

    # Journals
    results["journals"] = len(
        re.findall(r"\b(Nature|Science|NEJM|Lancet|Cell|PNAS)\b", text, re.IGNORECASE)
    )

    # Citations
    results["citations"] = len(re.findall(r"\[\d+\]|\(\w+.*?\d{4}\)", text))

    # Authority claims
    results["authority_claims"] = len(
        re.findall(
            r"(according to|studies show|research (has )?(proven|demonstrated))",
            text,
            re.IGNORECASE,
        )
    )

    results["total"] = sum(v for k, v in results.items() if k != "total")
    return results


# =============================================================================
# VERBOSITY BIAS
# =============================================================================


def compute_verbosity_metrics(
    response: str,
    reference: str | None = None,
    word_count_threshold: int = 500,
) -> dict[str, Any]:
    """
    Compute verbosity metrics for bias detection.

    Args:
        response: The response to analyze
        reference: Optional reference/ground truth for comparison
        word_count_threshold: Absolute threshold for "long" response

    Returns:
        Dict with verbosity metrics
    """

    def count_words(text: str) -> int:
        return len(text.split())

    def count_sentences(text: str) -> int:
        return len(re.findall(r"[.!?]+", text)) or 1

    response_words = count_words(response)
    response_sentences = count_sentences(response)
    response_chars = len(response)

    metrics = {
        "response_word_count": response_words,
        "response_char_count": response_chars,
        "response_sentence_count": response_sentences,
        "avg_words_per_sentence": response_words / max(response_sentences, 1),
        "is_verbose": response_words > word_count_threshold,
    }

    if reference:
        ref_words = count_words(reference)
        ref_chars = len(reference)

        metrics["reference_word_count"] = ref_words
        metrics["reference_char_count"] = ref_chars
        metrics["word_ratio"] = response_words / max(ref_words, 1)
        metrics["char_ratio"] = response_chars / max(ref_chars, 1)

        # Flag if significantly longer than reference
        metrics["significantly_longer"] = metrics["word_ratio"] > 1.5
        metrics["extremely_longer"] = metrics["word_ratio"] > 2.5

    return metrics


def apply_verbosity_penalty(
    score: float,
    verbosity_metrics: dict[str, Any],
    penalty_factor: float = 0.1,
    max_penalty: float = 0.2,
) -> tuple[float, float]:
    """
    Apply penalty to score based on verbosity.

    Args:
        score: Original normalized score (0-1)
        verbosity_metrics: Output from compute_verbosity_metrics
        penalty_factor: How much to penalize per unit of excess length
        max_penalty: Maximum penalty to apply

    Returns:
        Tuple of (adjusted_score, penalty_applied)
    """
    penalty = 0.0

    word_ratio = verbosity_metrics.get("word_ratio", 1.0)

    # Only penalize if significantly longer than reference
    if word_ratio > 1.5:
        excess = word_ratio - 1.5
        penalty = min(excess * penalty_factor, max_penalty)

    adjusted_score = max(0.0, score - penalty)
    return adjusted_score, penalty


# =============================================================================
# SENTIMENT BIAS
# =============================================================================

POSITIVE_WORDS = frozenset(
    {
        "excellent",
        "great",
        "good",
        "best",
        "perfect",
        "wonderful",
        "amazing",
        "outstanding",
        "superb",
        "fantastic",
        "brilliant",
        "exceptional",
        "impressive",
        "remarkable",
        "superior",
        "ideal",
        "optimal",
        "effective",
        "successful",
        "beneficial",
        "advantageous",
        "favorable",
        "positive",
        "helpful",
        "useful",
    }
)

NEGATIVE_WORDS = frozenset(
    {
        "poor",
        "bad",
        "wrong",
        "incorrect",
        "failed",
        "terrible",
        "awful",
        "horrible",
        "worst",
        "inferior",
        "inadequate",
        "insufficient",
        "deficient",
        "flawed",
        "problematic",
        "disappointing",
        "unsatisfactory",
        "negative",
        "harmful",
        "detrimental",
        "unfavorable",
        "unsuccessful",
        "ineffective",
        "useless",
        "worthless",
    }
)


def compute_sentiment_score(text: str) -> float:
    """
    Compute simple sentiment score for text.

    Returns:
        Score from -1 (very negative) to +1 (very positive)
    """
    words = set(text.lower().split())

    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def analyze_sentiment_bias(
    responses: list[str],
    judge_scores: list[float],
) -> dict[str, Any]:
    """
    Analyze correlation between response sentiment and judge scores.

    Args:
        responses: List of response texts
        judge_scores: Corresponding judge scores

    Returns:
        Dict with sentiment bias analysis
    """
    if len(responses) < 3:
        return {
            "sentiment_bias_detected": False,
            "error": "Insufficient samples for analysis",
        }

    sentiment_scores = [compute_sentiment_score(r) for r in responses]

    # Compute correlation
    correlation = float(np.corrcoef(sentiment_scores, judge_scores)[0, 1])

    # Handle NaN (can occur with constant values)
    if np.isnan(correlation):
        correlation = 0.0

    return {
        "sentiment_scores": sentiment_scores,
        "correlation": correlation,
        "sentiment_bias_detected": abs(correlation) > 0.3,
        "bias_direction": "positive" if correlation > 0 else "negative",
        "bias_strength": (
            "strong" if abs(correlation) > 0.5 else "moderate" if abs(correlation) > 0.3 else "weak"
        ),
    }


# =============================================================================
# BANDWAGON BIAS
# =============================================================================

BANDWAGON_PATTERNS = [
    r"(\d+%?\s+of\s+(people|users|experts?|studies))",
    r"(most\s+(people|experts?|researchers?|scientists?))",
    r"(widely\s+(accepted|believed|recognized|known))",
    r"(consensus\s+(is|shows?|agrees?))",
    r"(everyone\s+(knows?|agrees?|believes?))",
    r"(popular\s+(opinion|belief|view))",
    r"(majority\s+(of|believes?|agrees?))",
    r"(commonly\s+(accepted|believed|known))",
]


def detect_bandwagon_markers(text: str) -> dict[str, Any]:
    """
    Detect bandwagon effect markers in text.

    Returns:
        Dict with bandwagon marker analysis
    """
    markers_found = []

    for pattern in BANDWAGON_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            markers_found.extend(matches)

    return {
        "bandwagon_markers_count": len(markers_found),
        "markers": markers_found[:10],  # Limit to first 10
        "has_bandwagon_bias_risk": len(markers_found) > 0,
    }


def strip_bandwagon_markers(text: str) -> tuple[str, int]:
    """
    Strip bandwagon markers from text.

    Returns:
        Tuple of (cleaned_text, markers_removed_count)
    """
    result = text
    total_removed = 0

    for pattern in BANDWAGON_PATTERNS:
        matches = re.findall(pattern, result, flags=re.IGNORECASE)
        total_removed += len(matches)
        result = re.sub(pattern, "[CLAIM]", result, flags=re.IGNORECASE)

    return result, total_removed


# =============================================================================
# BEAUTY/FORMATTING BIAS
# =============================================================================


def compute_formatting_score(text: str) -> dict[str, Any]:
    """
    Compute formatting/presentation quality score.

    Higher scores indicate more "beautiful" formatting that might
    bias judges regardless of content quality.

    Returns:
        Dict with formatting metrics
    """
    # Check for markdown formatting
    has_headers = bool(re.search(r"^#+\s", text, re.MULTILINE))
    has_bullets = bool(re.search(r"^[\-\*]\s", text, re.MULTILINE))
    has_numbered = bool(re.search(r"^\d+\.\s", text, re.MULTILINE))
    has_code_blocks = bool(re.search(r"```", text))
    has_inline_code = bool(re.search(r"`[^`]+`", text))
    has_bold = bool(re.search(r"\*\*[^*]+\*\*", text))
    has_emphasis = bool(re.search(r"_[^_]+_|\*[^*]+\*", text))

    # Count structural elements
    paragraph_count = len(re.findall(r"\n\n+", text)) + 1
    list_item_count = len(re.findall(r"^[\-\*\d]+\.?\s", text, re.MULTILINE))

    # Compute composite score (0-1)
    features = [
        has_headers,
        has_bullets,
        has_numbered,
        has_code_blocks,
        has_inline_code,
        has_bold,
        has_emphasis,
    ]
    formatting_score = sum(features) / len(features)

    return {
        "formatting_score": formatting_score,
        "has_headers": has_headers,
        "has_bullets": has_bullets,
        "has_numbered_lists": has_numbered,
        "has_code_blocks": has_code_blocks,
        "has_inline_code": has_inline_code,
        "has_bold": has_bold,
        "has_emphasis": has_emphasis,
        "paragraph_count": paragraph_count,
        "list_item_count": list_item_count,
        "well_formatted": formatting_score > 0.4,
    }


# =============================================================================
# COMPREHENSIVE BIAS ANALYSIS
# =============================================================================


def analyze_response_for_biases(
    response: str,
    reference: str | None = None,
    judge_model: str | None = None,
    response_source_model: str | None = None,
) -> BiasReport:
    """
    Perform comprehensive bias analysis on a response.

    Args:
        response: The response text to analyze
        reference: Optional reference/ground truth text
        judge_model: Name of the judge model
        response_source_model: Name of the model that generated the response

    Returns:
        BiasReport with all detected biases
    """
    report = BiasReport()

    # 1. Verbosity analysis
    verbosity = compute_verbosity_metrics(response, reference)
    report.verbosity_ratio = verbosity.get("word_ratio", 1.0)
    if verbosity.get("extremely_longer"):
        report.verbosity_warning = (
            f"Response is {report.verbosity_ratio:.1f}x longer than reference"
        )
        report.add_warning(f"Verbosity bias risk: {report.verbosity_warning}")
    elif verbosity.get("significantly_longer"):
        report.verbosity_warning = (
            f"Response is {report.verbosity_ratio:.1f}x longer than reference"
        )

    # 2. Self-enhancement analysis
    if judge_model and response_source_model:
        self_audit = detect_self_enhancement_risk(judge_model, response_source_model)
        report.self_enhancement_risk = self_audit.is_same_family
        report.judge_model_family = self_audit.judge_family
        report.response_model_family = self_audit.response_family
        if self_audit.bias_warning:
            report.add_warning(self_audit.bias_warning)

    # 3. Authority markers
    authority = detect_authority_markers(response)
    report.authority_markers_found = authority["total"]
    if authority["total"] > 3:
        report.add_warning(
            f"Authority bias risk: {authority['total']} authority markers found"
        )

    # 4. Sentiment analysis
    report.response_sentiment_score = compute_sentiment_score(response)
    if abs(report.response_sentiment_score) > 0.5:
        direction = "positive" if report.response_sentiment_score > 0 else "negative"
        report.add_warning(f"Strong {direction} sentiment may influence evaluation")

    # 5. Bandwagon markers
    bandwagon = detect_bandwagon_markers(response)
    report.bandwagon_markers_found = bandwagon["bandwagon_markers_count"]
    if bandwagon["has_bandwagon_bias_risk"]:
        report.add_warning(
            f"Bandwagon bias risk: {report.bandwagon_markers_found} popularity claims found"
        )

    # 6. Formatting/beauty
    formatting = compute_formatting_score(response)
    report.formatting_score = formatting["formatting_score"]

    return report


def prepare_response_for_evaluation(
    response: str,
    strip_authority: bool = True,
    strip_bandwagon: bool = True,
    normalize_formatting: bool = False,
) -> tuple[str, dict[str, Any]]:
    """
    Prepare a response for bias-reduced evaluation.

    Args:
        response: Original response text
        strip_authority: Remove authority markers
        strip_bandwagon: Remove bandwagon markers
        normalize_formatting: Remove markdown formatting

    Returns:
        Tuple of (prepared_text, modifications_made)
    """
    result = response
    modifications = {
        "original_length": len(response),
        "authority_markers_removed": 0,
        "bandwagon_markers_removed": 0,
        "formatting_normalized": False,
    }

    if strip_authority:
        result, count = strip_authority_markers(result)
        modifications["authority_markers_removed"] = count

    if strip_bandwagon:
        result, count = strip_bandwagon_markers(result)
        modifications["bandwagon_markers_removed"] = count

    if normalize_formatting:
        # Strip markdown formatting
        result = re.sub(r"#{1,6}\s*", "", result)  # Headers
        result = re.sub(r"\*\*([^*]+)\*\*", r"\1", result)  # Bold
        result = re.sub(r"_([^_]+)_", r"\1", result)  # Italic
        result = re.sub(r"`([^`]+)`", r"\1", result)  # Inline code
        result = re.sub(r"```[\s\S]*?```", "[CODE]", result)  # Code blocks
        modifications["formatting_normalized"] = True

    modifications["final_length"] = len(result)

    return result, modifications
