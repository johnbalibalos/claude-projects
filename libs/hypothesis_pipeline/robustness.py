"""
Robustness evaluation module for LLM evaluation.

Tests model stability under various perturbations:
- Typo injection
- Synonym replacement
- Sentence reordering
- Case variations
- Formatting changes
- Paraphrasing
- Consistency testing (same input → same output)
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Sequence

import numpy as np


# =============================================================================
# PROTOCOLS
# =============================================================================


class ModelClient(Protocol):
    """Protocol for model clients used in robustness testing."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion from the model."""
        ...


class Evaluator(Protocol):
    """Protocol for evaluating model responses."""

    def score(self, response: str, ground_truth: Any) -> float:
        """Score a response against ground truth."""
        ...


# =============================================================================
# PERTURBATION FUNCTIONS
# =============================================================================


def introduce_typos(
    text: str,
    rate: float = 0.02,
    seed: int | None = None,
) -> str:
    """
    Introduce random typos into text.

    Args:
        text: Original text
        rate: Probability of typo per character
        seed: Random seed

    Returns:
        Text with typos
    """
    rng = np.random.default_rng(seed)
    chars = list(text)

    for i in range(len(chars)):
        if chars[i].isalpha() and rng.random() < rate:
            typo_type = rng.choice(["swap", "delete", "duplicate", "replace"])

            if typo_type == "swap" and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif typo_type == "delete":
                chars[i] = ""
            elif typo_type == "duplicate":
                chars[i] = chars[i] * 2
            elif typo_type == "replace":
                # Replace with nearby key (simplified)
                nearby = {
                    'a': 'sqw', 'b': 'vgn', 'c': 'xvd', 'd': 'sfec',
                    'e': 'wrd', 'f': 'dgrc', 'g': 'fhtv', 'h': 'gjyn',
                    'i': 'uok', 'j': 'hkun', 'k': 'jlim', 'l': 'kop',
                    'm': 'njk', 'n': 'bmhj', 'o': 'iplk', 'p': 'ol',
                    'q': 'wa', 'r': 'etdf', 's': 'awdz', 't': 'ryfg',
                    'u': 'yij', 'v': 'cfgb', 'w': 'qeas', 'x': 'zsc',
                    'y': 'tugh', 'z': 'xas',
                }
                char_lower = chars[i].lower()
                if char_lower in nearby:
                    replacement = rng.choice(list(nearby[char_lower]))
                    if chars[i].isupper():
                        replacement = replacement.upper()
                    chars[i] = replacement

    return "".join(chars)


def replace_with_synonyms(
    text: str,
    replacement_rate: float = 0.1,
    seed: int | None = None,
) -> str:
    """
    Replace words with synonyms.

    Args:
        text: Original text
        replacement_rate: Probability of replacing each word
        seed: Random seed

    Returns:
        Text with synonym replacements
    """
    rng = np.random.default_rng(seed)

    # Simple synonym dictionary (can be expanded)
    synonyms = {
        "large": ["big", "huge", "massive", "substantial"],
        "small": ["tiny", "little", "minute", "compact"],
        "important": ["significant", "crucial", "essential", "vital"],
        "show": ["demonstrate", "display", "exhibit", "reveal"],
        "use": ["utilize", "employ", "apply", "leverage"],
        "make": ["create", "produce", "generate", "construct"],
        "good": ["effective", "excellent", "satisfactory", "positive"],
        "bad": ["poor", "negative", "adverse", "unfavorable"],
        "many": ["numerous", "several", "multiple", "various"],
        "few": ["several", "some", "a handful of", "limited"],
        "high": ["elevated", "increased", "substantial", "significant"],
        "low": ["reduced", "decreased", "minimal", "limited"],
        "increase": ["rise", "grow", "expand", "elevate"],
        "decrease": ["decline", "reduce", "diminish", "lower"],
        "fast": ["quick", "rapid", "swift", "speedy"],
        "slow": ["gradual", "unhurried", "leisurely", "measured"],
        "begin": ["start", "commence", "initiate", "launch"],
        "end": ["finish", "conclude", "terminate", "complete"],
        "help": ["assist", "aid", "support", "facilitate"],
        "difficult": ["challenging", "hard", "complex", "demanding"],
        "easy": ["simple", "straightforward", "uncomplicated", "effortless"],
        "new": ["novel", "fresh", "recent", "modern"],
        "old": ["ancient", "previous", "former", "dated"],
    }

    words = text.split()
    result = []

    for word in words:
        # Extract punctuation
        prefix = ""
        suffix = ""
        core = word

        while core and core[0] in string.punctuation:
            prefix += core[0]
            core = core[1:]
        while core and core[-1] in string.punctuation:
            suffix = core[-1] + suffix
            core = core[:-1]

        core_lower = core.lower()

        if core_lower in synonyms and rng.random() < replacement_rate:
            replacement = rng.choice(synonyms[core_lower])
            # Preserve case
            if core.isupper():
                replacement = replacement.upper()
            elif core[0].isupper():
                replacement = replacement.capitalize()
            result.append(prefix + replacement + suffix)
        else:
            result.append(word)

    return " ".join(result)


def reorder_sentences(
    text: str,
    keep_first: bool = True,
    keep_last: bool = True,
    seed: int | None = None,
) -> str:
    """
    Reorder sentences in text.

    Args:
        text: Original text
        keep_first: Keep first sentence in place
        keep_last: Keep last sentence in place
        seed: Random seed

    Returns:
        Text with reordered sentences
    """
    rng = np.random.default_rng(seed)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) <= 2:
        return text

    # Determine which sentences to shuffle
    if keep_first and keep_last:
        first = sentences[0]
        last = sentences[-1]
        middle = sentences[1:-1]
        rng.shuffle(middle)
        sentences = [first] + list(middle) + [last]
    elif keep_first:
        first = sentences[0]
        rest = sentences[1:]
        rng.shuffle(rest)
        sentences = [first] + list(rest)
    elif keep_last:
        last = sentences[-1]
        rest = sentences[:-1]
        rng.shuffle(rest)
        sentences = list(rest) + [last]
    else:
        rng.shuffle(sentences)

    return " ".join(sentences)


def random_case_change(
    text: str,
    rate: float = 0.05,
    seed: int | None = None,
) -> str:
    """
    Randomly change case of characters.

    Args:
        text: Original text
        rate: Probability of case change per character
        seed: Random seed

    Returns:
        Text with random case changes
    """
    rng = np.random.default_rng(seed)
    chars = list(text)

    for i in range(len(chars)):
        if chars[i].isalpha() and rng.random() < rate:
            if chars[i].isupper():
                chars[i] = chars[i].lower()
            else:
                chars[i] = chars[i].upper()

    return "".join(chars)


def add_whitespace_noise(
    text: str,
    rate: float = 0.03,
    seed: int | None = None,
) -> str:
    """
    Add random whitespace variations.

    Args:
        text: Original text
        rate: Probability of whitespace modification
        seed: Random seed

    Returns:
        Text with whitespace noise
    """
    rng = np.random.default_rng(seed)
    chars = list(text)
    result = []

    for char in chars:
        if char == " " and rng.random() < rate:
            # Either double space or remove
            if rng.random() < 0.5:
                result.append("  ")
            # else: skip (remove space)
        else:
            result.append(char)

    return "".join(result)


def add_punctuation_noise(
    text: str,
    rate: float = 0.02,
    seed: int | None = None,
) -> str:
    """
    Add minor punctuation variations.

    Args:
        text: Original text
        rate: Probability of punctuation modification
        seed: Random seed

    Returns:
        Text with punctuation noise
    """
    rng = np.random.default_rng(seed)

    # Period variations
    if rng.random() < rate:
        text = text.replace(". ", ".\n")

    # Comma variations
    chars = list(text)
    for i in range(len(chars)):
        if chars[i] == "," and rng.random() < rate:
            chars[i] = " -"  # Replace comma with dash

    return "".join(chars)


# =============================================================================
# PERTURBATION REGISTRY
# =============================================================================


PERTURBATIONS: dict[str, Callable[[str, float, int | None], str]] = {
    "typo": introduce_typos,
    "synonym": replace_with_synonyms,
    "reorder": lambda t, r, s: reorder_sentences(t, seed=s),
    "case": random_case_change,
    "whitespace": add_whitespace_noise,
    "punctuation": add_punctuation_noise,
}


def get_perturbation(name: str) -> Callable[[str, float, int | None], str]:
    """Get a perturbation function by name."""
    if name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {name}. Available: {list(PERTURBATIONS.keys())}")
    return PERTURBATIONS[name]


def apply_perturbation(
    text: str,
    perturbation: str,
    rate: float = 0.05,
    seed: int | None = None,
) -> str:
    """Apply a named perturbation to text."""
    fn = get_perturbation(perturbation)
    return fn(text, rate, seed)


# =============================================================================
# ROBUSTNESS EVALUATION
# =============================================================================


@dataclass
class PerturbationResult:
    """Result of a single perturbation test."""

    perturbation_type: str
    original_input: str
    perturbed_input: str
    original_response: str
    perturbed_response: str
    original_score: float
    perturbed_score: float
    score_delta: float
    response_similarity: float  # How similar are the responses


@dataclass
class RobustnessResult:
    """Aggregate robustness evaluation result."""

    n_test_cases: int
    perturbations_tested: list[str]
    results_by_perturbation: dict[str, list[PerturbationResult]]

    # Aggregate metrics
    mean_score_drop: dict[str, float]  # Average score drop per perturbation
    worst_case_drop: dict[str, float]  # Worst case score drop per perturbation
    consistency_rate: dict[str, float]  # Rate of consistent outputs per perturbation
    overall_robustness_score: float  # 0-1, higher is more robust

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ROBUSTNESS EVALUATION REPORT",
            "=" * 60,
            f"Test cases: {self.n_test_cases}",
            f"Perturbations tested: {', '.join(self.perturbations_tested)}",
            f"Overall robustness score: {self.overall_robustness_score:.3f}",
            "",
            "Per-Perturbation Results:",
            "| Perturbation | Mean Drop | Worst Drop | Consistency |",
            "|--------------|-----------|------------|-------------|",
        ]

        for p in self.perturbations_tested:
            lines.append(
                f"| {p:12} | {self.mean_score_drop.get(p, 0):9.3f} | "
                f"{self.worst_case_drop.get(p, 0):10.3f} | "
                f"{self.consistency_rate.get(p, 0):11.1%} |"
            )

        return "\n".join(lines)

    def interpretation(self) -> str:
        """Provide interpretation of robustness results."""
        if self.overall_robustness_score >= 0.9:
            quality = "excellent"
            advice = "Model is highly robust to input perturbations."
        elif self.overall_robustness_score >= 0.8:
            quality = "good"
            advice = "Model shows good robustness with minor sensitivity."
        elif self.overall_robustness_score >= 0.6:
            quality = "moderate"
            advice = "Model has some robustness issues. Consider data augmentation."
        else:
            quality = "poor"
            advice = "Model is highly sensitive to input perturbations."

        # Find most problematic perturbation
        worst_perturbation = max(self.mean_score_drop, key=self.mean_score_drop.get)
        worst_drop = self.mean_score_drop[worst_perturbation]

        return (
            f"Robustness quality: {quality}. {advice}\n"
            f"Most problematic perturbation: {worst_perturbation} "
            f"(avg drop: {worst_drop:.3f})"
        )


class RobustnessEvaluator:
    """
    Evaluates model robustness under various perturbations.

    Tests whether model outputs remain consistent and accurate
    when inputs are perturbed in semantically-preserving ways.
    """

    def __init__(
        self,
        model_client: ModelClient,
        evaluator: Evaluator,
        perturbations: Sequence[str] | None = None,
        perturbation_rate: float = 0.05,
        n_samples_per_perturbation: int = 1,
    ):
        """
        Initialize robustness evaluator.

        Args:
            model_client: Client for model inference
            evaluator: Evaluator for scoring responses
            perturbations: List of perturbation types to test
            perturbation_rate: Rate/intensity of perturbations
            n_samples_per_perturbation: Number of perturbed samples per test case
        """
        self.client = model_client
        self.evaluator = evaluator
        self.perturbations = list(perturbations or PERTURBATIONS.keys())
        self.perturbation_rate = perturbation_rate
        self.n_samples = n_samples_per_perturbation

    def evaluate(
        self,
        test_cases: list[dict[str, Any]],
        input_field: str = "input",
        ground_truth_field: str = "ground_truth",
        prompt_template: str = "{input}",
        verbose: bool = True,
    ) -> RobustnessResult:
        """
        Evaluate robustness on a test set.

        Args:
            test_cases: List of test case dictionaries
            input_field: Field name for input text
            ground_truth_field: Field name for ground truth
            prompt_template: Template for constructing prompts
            verbose: Whether to print progress

        Returns:
            RobustnessResult with aggregate metrics
        """
        results_by_perturbation: dict[str, list[PerturbationResult]] = {
            p: [] for p in self.perturbations
        }

        for i, tc in enumerate(test_cases):
            if verbose:
                print(f"[{i+1}/{len(test_cases)}] Testing robustness...")

            original_input = tc[input_field]
            ground_truth = tc[ground_truth_field]

            # Get original response
            original_prompt = prompt_template.format(input=original_input)
            original_response = self.client.generate(original_prompt)
            original_score = self.evaluator.score(original_response, ground_truth)

            # Test each perturbation
            for perturbation in self.perturbations:
                for sample_idx in range(self.n_samples):
                    seed = hash(f"{tc.get('id', i)}_{perturbation}_{sample_idx}") % (2**31)

                    perturbed_input = apply_perturbation(
                        original_input,
                        perturbation,
                        rate=self.perturbation_rate,
                        seed=seed,
                    )

                    perturbed_prompt = prompt_template.format(input=perturbed_input)
                    perturbed_response = self.client.generate(perturbed_prompt)
                    perturbed_score = self.evaluator.score(perturbed_response, ground_truth)

                    # Compute response similarity
                    response_similarity = self._compute_similarity(
                        original_response, perturbed_response
                    )

                    results_by_perturbation[perturbation].append(PerturbationResult(
                        perturbation_type=perturbation,
                        original_input=original_input,
                        perturbed_input=perturbed_input,
                        original_response=original_response,
                        perturbed_response=perturbed_response,
                        original_score=original_score,
                        perturbed_score=perturbed_score,
                        score_delta=original_score - perturbed_score,
                        response_similarity=response_similarity,
                    ))

        # Compute aggregate metrics
        mean_score_drop = {}
        worst_case_drop = {}
        consistency_rate = {}

        for p, results in results_by_perturbation.items():
            if results:
                drops = [r.score_delta for r in results]
                mean_score_drop[p] = float(np.mean(drops))
                worst_case_drop[p] = float(np.max(drops))
                # Consistency: similarity > 0.9 and score drop < 0.1
                consistent = sum(
                    1 for r in results
                    if r.response_similarity > 0.9 and r.score_delta < 0.1
                )
                consistency_rate[p] = consistent / len(results)

        # Overall robustness score
        if mean_score_drop:
            avg_drop = np.mean(list(mean_score_drop.values()))
            avg_consistency = np.mean(list(consistency_rate.values()))
            overall_robustness = (1 - avg_drop) * 0.5 + avg_consistency * 0.5
        else:
            overall_robustness = 1.0

        return RobustnessResult(
            n_test_cases=len(test_cases),
            perturbations_tested=self.perturbations,
            results_by_perturbation=results_by_perturbation,
            mean_score_drop=mean_score_drop,
            worst_case_drop=worst_case_drop,
            consistency_rate=consistency_rate,
            overall_robustness_score=float(overall_robustness),
        )

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# =============================================================================
# CONSISTENCY TESTING
# =============================================================================


@dataclass
class ConsistencyResult:
    """Result of consistency testing."""

    n_test_cases: int
    n_repetitions: int
    consistency_rate: float  # Fraction with identical outputs
    semantic_consistency_rate: float  # Fraction with semantically similar outputs
    score_variance: float  # Variance in scores across repetitions
    results: list[dict[str, Any]] = field(repr=False, default_factory=list)

    def summary(self) -> str:
        return (
            f"Consistency Analysis (n={self.n_test_cases}, reps={self.n_repetitions})\n"
            f"  Exact match rate: {self.consistency_rate:.1%}\n"
            f"  Semantic consistency: {self.semantic_consistency_rate:.1%}\n"
            f"  Score variance: {self.score_variance:.4f}"
        )


class ConsistencyTester:
    """
    Tests whether model produces consistent outputs for identical inputs.

    Important for evaluating:
    - Model determinism (temperature=0 should be deterministic)
    - Response stability
    - API reliability
    """

    def __init__(
        self,
        model_client: ModelClient,
        evaluator: Evaluator,
        n_repetitions: int = 3,
        similarity_threshold: float = 0.9,
    ):
        """
        Initialize consistency tester.

        Args:
            model_client: Client for model inference
            evaluator: Evaluator for scoring responses
            n_repetitions: Number of times to repeat each input
            similarity_threshold: Threshold for semantic similarity
        """
        self.client = model_client
        self.evaluator = evaluator
        self.n_repetitions = n_repetitions
        self.similarity_threshold = similarity_threshold

    def test_consistency(
        self,
        test_cases: list[dict[str, Any]],
        input_field: str = "input",
        ground_truth_field: str = "ground_truth",
        prompt_template: str = "{input}",
        verbose: bool = True,
    ) -> ConsistencyResult:
        """
        Test model consistency on a test set.

        Args:
            test_cases: List of test case dictionaries
            input_field: Field name for input text
            ground_truth_field: Field name for ground truth
            prompt_template: Template for constructing prompts
            verbose: Whether to print progress

        Returns:
            ConsistencyResult with consistency metrics
        """
        results = []
        exact_matches = 0
        semantic_matches = 0
        all_score_variances = []

        for i, tc in enumerate(test_cases):
            if verbose:
                print(f"[{i+1}/{len(test_cases)}] Testing consistency...")

            input_text = tc[input_field]
            ground_truth = tc[ground_truth_field]
            prompt = prompt_template.format(input=input_text)

            # Generate multiple responses
            responses = []
            scores = []
            for _ in range(self.n_repetitions):
                response = self.client.generate(prompt)
                score = self.evaluator.score(response, ground_truth)
                responses.append(response)
                scores.append(score)

            # Check exact match
            is_exact_match = len(set(responses)) == 1

            # Check semantic similarity
            is_semantic_match = all(
                self._compute_similarity(responses[0], r) >= self.similarity_threshold
                for r in responses[1:]
            )

            if is_exact_match:
                exact_matches += 1
            if is_semantic_match:
                semantic_matches += 1

            score_variance = float(np.var(scores))
            all_score_variances.append(score_variance)

            results.append({
                "test_case_id": tc.get("id", i),
                "responses": responses,
                "scores": scores,
                "is_exact_match": is_exact_match,
                "is_semantic_match": is_semantic_match,
                "score_variance": score_variance,
            })

        n = len(test_cases)

        return ConsistencyResult(
            n_test_cases=n,
            n_repetitions=self.n_repetitions,
            consistency_rate=exact_matches / n if n > 0 else 0.0,
            semantic_consistency_rate=semantic_matches / n if n > 0 else 0.0,
            score_variance=float(np.mean(all_score_variances)) if all_score_variances else 0.0,
            results=results,
        )

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# =============================================================================
# ADVERSARIAL ROBUSTNESS
# =============================================================================


@dataclass
class AdversarialResult:
    """Result of adversarial robustness testing."""

    attack_type: str
    success_rate: float  # Rate at which attack changed the output
    average_perturbation_size: float
    examples: list[dict[str, Any]]


def test_adversarial_robustness(
    model_client: ModelClient,
    evaluator: Evaluator,
    test_cases: list[dict[str, Any]],
    attack_type: Literal["character", "word", "sentence"] = "character",
    max_perturbations: int = 10,
    input_field: str = "input",
    ground_truth_field: str = "ground_truth",
) -> AdversarialResult:
    """
    Test adversarial robustness by finding minimal perturbations that change output.

    Args:
        model_client: Client for model inference
        evaluator: Evaluator for scoring
        test_cases: Test cases to evaluate
        attack_type: Type of perturbation to use
        max_perturbations: Maximum number of perturbation attempts
        input_field: Field name for input
        ground_truth_field: Field name for ground truth

    Returns:
        AdversarialResult with attack success metrics
    """
    successful_attacks = 0
    perturbation_sizes = []
    examples = []

    for tc in test_cases:
        original_input = tc[input_field]
        ground_truth = tc[ground_truth_field]

        # Get original prediction
        original_response = model_client.generate(original_input)
        original_score = evaluator.score(original_response, ground_truth)

        # Try increasing perturbation levels
        attack_succeeded = False
        perturbation_size = 0

        for level in range(1, max_perturbations + 1):
            if attack_type == "character":
                perturbed = introduce_typos(original_input, rate=0.01 * level, seed=level)
            elif attack_type == "word":
                perturbed = replace_with_synonyms(original_input, replacement_rate=0.05 * level, seed=level)
            else:  # sentence
                perturbed = reorder_sentences(original_input, seed=level)

            perturbed_response = model_client.generate(perturbed)
            perturbed_score = evaluator.score(perturbed_response, ground_truth)

            # Check if attack succeeded (score dropped significantly)
            if original_score - perturbed_score > 0.2:
                attack_succeeded = True
                perturbation_size = level
                examples.append({
                    "original": original_input,
                    "perturbed": perturbed,
                    "original_score": original_score,
                    "perturbed_score": perturbed_score,
                    "perturbation_level": level,
                })
                break

        if attack_succeeded:
            successful_attacks += 1
            perturbation_sizes.append(perturbation_size)

    n = len(test_cases)
    return AdversarialResult(
        attack_type=attack_type,
        success_rate=successful_attacks / n if n > 0 else 0.0,
        average_perturbation_size=float(np.mean(perturbation_sizes)) if perturbation_sizes else 0.0,
        examples=examples[:10],  # Keep top 10 examples
    )


# =============================================================================
# COMPREHENSIVE ROBUSTNESS ANALYSIS
# =============================================================================


@dataclass
class ComprehensiveRobustnessAnalysis:
    """Complete robustness analysis combining all tests."""

    perturbation_results: RobustnessResult
    consistency_results: ConsistencyResult
    adversarial_results: dict[str, AdversarialResult]
    overall_robustness_grade: str  # A, B, C, D, F
    recommendations: list[str]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "COMPREHENSIVE ROBUSTNESS ANALYSIS",
            "=" * 60,
            f"Overall Grade: {self.overall_robustness_grade}",
            "",
            "--- Perturbation Robustness ---",
            self.perturbation_results.summary(),
            "",
            "--- Consistency ---",
            self.consistency_results.summary(),
            "",
            "--- Adversarial Robustness ---",
        ]

        for attack_type, result in self.adversarial_results.items():
            lines.append(f"  {attack_type}: {result.success_rate:.1%} attack success rate")

        lines.append("")
        lines.append("Recommendations:")
        for rec in self.recommendations:
            lines.append(f"  - {rec}")

        return "\n".join(lines)


def comprehensive_robustness_analysis(
    model_client: ModelClient,
    evaluator: Evaluator,
    test_cases: list[dict[str, Any]],
    input_field: str = "input",
    ground_truth_field: str = "ground_truth",
    verbose: bool = True,
) -> ComprehensiveRobustnessAnalysis:
    """
    Run comprehensive robustness analysis.

    Args:
        model_client: Client for model inference
        evaluator: Evaluator for scoring
        test_cases: Test cases to evaluate
        input_field: Field name for input
        ground_truth_field: Field name for ground truth
        verbose: Whether to print progress

    Returns:
        ComprehensiveRobustnessAnalysis with all metrics
    """
    if verbose:
        print("Running perturbation robustness tests...")

    robustness_eval = RobustnessEvaluator(
        model_client, evaluator,
        perturbations=["typo", "synonym", "case"],
    )
    perturbation_results = robustness_eval.evaluate(
        test_cases, input_field, ground_truth_field, verbose=verbose
    )

    if verbose:
        print("\nRunning consistency tests...")

    consistency_tester = ConsistencyTester(model_client, evaluator, n_repetitions=2)
    consistency_results = consistency_tester.test_consistency(
        test_cases[:min(10, len(test_cases))],  # Limit for speed
        input_field, ground_truth_field, verbose=verbose
    )

    if verbose:
        print("\nRunning adversarial tests...")

    adversarial_results = {}
    for attack_type in ["character", "word"]:
        result = test_adversarial_robustness(
            model_client, evaluator,
            test_cases[:min(10, len(test_cases))],  # Limit for speed
            attack_type=attack_type,
            input_field=input_field,
            ground_truth_field=ground_truth_field,
        )
        adversarial_results[attack_type] = result

    # Compute overall grade
    scores = [
        perturbation_results.overall_robustness_score,
        consistency_results.consistency_rate,
        1 - np.mean([r.success_rate for r in adversarial_results.values()]),
    ]
    overall_score = np.mean(scores)

    if overall_score >= 0.9:
        grade = "A"
    elif overall_score >= 0.8:
        grade = "B"
    elif overall_score >= 0.7:
        grade = "C"
    elif overall_score >= 0.6:
        grade = "D"
    else:
        grade = "F"

    # Generate recommendations
    recommendations = []

    if perturbation_results.overall_robustness_score < 0.8:
        worst_perturb = max(
            perturbation_results.mean_score_drop,
            key=perturbation_results.mean_score_drop.get
        )
        recommendations.append(
            f"Improve robustness to {worst_perturb} perturbations through data augmentation"
        )

    if consistency_results.consistency_rate < 0.9:
        recommendations.append(
            "Increase output consistency by using lower temperature or ensemble methods"
        )

    if any(r.success_rate > 0.3 for r in adversarial_results.values()):
        recommendations.append(
            "Consider adversarial training to improve resistance to input perturbations"
        )

    return ComprehensiveRobustnessAnalysis(
        perturbation_results=perturbation_results,
        consistency_results=consistency_results,
        adversarial_results=adversarial_results,
        overall_robustness_grade=grade,
        recommendations=recommendations,
    )
