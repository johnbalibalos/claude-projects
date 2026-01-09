"""
Prompt sensitivity analysis module.

Analyzes how sensitive model performance is to prompt variations:
- Prompt perturbation testing
- Paraphrase stability
- Format sensitivity
- Instruction robustness
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike


# =============================================================================
# PROTOCOLS
# =============================================================================


class ModelClient(Protocol):
    """Protocol for model clients."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion from the model."""
        ...


class Evaluator(Protocol):
    """Protocol for evaluating responses."""

    def score(self, response: str, ground_truth: Any) -> float:
        """Score a response against ground truth."""
        ...


# =============================================================================
# PROMPT VARIATIONS
# =============================================================================


PROMPT_VARIATION_TEMPLATES = {
    # Politeness variations
    "polite": "Please {instruction}. Thank you.",
    "imperative": "{instruction}.",
    "request": "Could you {instruction}?",
    "command": "You must {instruction}.",

    # Framing variations
    "expert": "As an expert, {instruction}.",
    "student": "Help me understand: {instruction}",
    "assistant": "As a helpful assistant, {instruction}",
    "direct": "{instruction}",

    # Structure variations
    "numbered": "1. {instruction}\n2. Provide your answer.",
    "bullet": "• {instruction}\n• Give a clear response.",
    "paragraph": "{instruction} Make sure to be thorough in your response.",
    "minimal": "{instruction}",

    # Emphasis variations
    "caps_emphasis": "IMPORTANT: {instruction}",
    "bold_markers": "**{instruction}**",
    "quoted": '"{instruction}"',
    "plain": "{instruction}",
}


def get_prompt_variations(
    base_instruction: str,
    variation_types: list[str] | None = None,
) -> dict[str, str]:
    """
    Generate prompt variations from a base instruction.

    Args:
        base_instruction: The core instruction
        variation_types: Types of variations to generate (default: all)

    Returns:
        Dictionary mapping variation name to full prompt
    """
    if variation_types is None:
        variation_types = list(PROMPT_VARIATION_TEMPLATES.keys())

    variations = {}
    for vtype in variation_types:
        if vtype in PROMPT_VARIATION_TEMPLATES:
            template = PROMPT_VARIATION_TEMPLATES[vtype]
            variations[vtype] = template.format(instruction=base_instruction)

    return variations


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================


@dataclass
class VariationResult:
    """Result for a single prompt variation."""

    variation_name: str
    prompt: str
    response: str
    score: float
    response_length: int


@dataclass
class SensitivityResult:
    """Result of prompt sensitivity analysis."""

    n_variations: int
    n_test_cases: int
    variation_results: dict[str, list[VariationResult]]

    # Aggregate metrics
    mean_scores_by_variation: dict[str, float]
    std_scores_by_variation: dict[str, float]
    best_variation: str
    worst_variation: str

    # Sensitivity metrics
    score_variance: float  # Variance across variations
    score_range: float  # Max - min across variations
    sensitivity_index: float  # 0-1, higher = more sensitive
    is_prompt_sensitive: bool  # True if variance is high

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PROMPT SENSITIVITY ANALYSIS",
            "=" * 60,
            f"Variations tested: {self.n_variations}",
            f"Test cases: {self.n_test_cases}",
            f"Sensitivity index: {self.sensitivity_index:.3f}",
            f"Prompt sensitive: {'Yes' if self.is_prompt_sensitive else 'No'}",
            "",
            "Performance by Variation:",
            "| Variation | Mean Score | Std |",
            "|-----------|------------|-----|",
        ]

        for var in sorted(self.mean_scores_by_variation.keys()):
            mean = self.mean_scores_by_variation[var]
            std = self.std_scores_by_variation.get(var, 0)
            marker = "★" if var == self.best_variation else ("✗" if var == self.worst_variation else " ")
            lines.append(f"| {var:20} | {mean:.3f} | {std:.3f} | {marker}")

        lines.extend([
            "",
            f"Best variation: {self.best_variation}",
            f"Worst variation: {self.worst_variation}",
            f"Score range: {self.score_range:.3f}",
        ])

        return "\n".join(lines)


class PromptSensitivityAnalyzer:
    """
    Analyzes how sensitive model performance is to prompt variations.

    Tests the same content with different prompt phrasings to measure
    how robust the model is to superficial prompt changes.
    """

    def __init__(
        self,
        model_client: ModelClient,
        evaluator: Evaluator,
        variations: list[str] | None = None,
    ):
        """
        Initialize analyzer.

        Args:
            model_client: Client for model inference
            evaluator: Evaluator for scoring responses
            variations: List of variation types to test
        """
        self.client = model_client
        self.evaluator = evaluator
        self.variations = variations or list(PROMPT_VARIATION_TEMPLATES.keys())

    def analyze(
        self,
        test_cases: list[dict[str, Any]],
        instruction_field: str = "instruction",
        ground_truth_field: str = "ground_truth",
        context_field: str | None = None,
        verbose: bool = True,
    ) -> SensitivityResult:
        """
        Analyze prompt sensitivity on a test set.

        Args:
            test_cases: List of test case dictionaries
            instruction_field: Field containing base instruction
            ground_truth_field: Field containing ground truth
            context_field: Optional field with additional context
            verbose: Whether to print progress

        Returns:
            SensitivityResult with sensitivity metrics
        """
        variation_results: dict[str, list[VariationResult]] = {
            v: [] for v in self.variations
        }

        for i, tc in enumerate(test_cases):
            if verbose:
                print(f"[{i+1}/{len(test_cases)}] Testing sensitivity...")

            base_instruction = tc[instruction_field]
            ground_truth = tc[ground_truth_field]
            context = tc.get(context_field, "") if context_field else ""

            # Generate all variations
            prompts = get_prompt_variations(base_instruction, self.variations)

            for var_name, prompt in prompts.items():
                # Add context if available
                full_prompt = f"{context}\n\n{prompt}" if context else prompt

                # Get response
                response = self.client.generate(full_prompt)
                score = self.evaluator.score(response, ground_truth)

                variation_results[var_name].append(VariationResult(
                    variation_name=var_name,
                    prompt=prompt,
                    response=response,
                    score=score,
                    response_length=len(response),
                ))

        # Compute aggregate metrics
        mean_scores = {}
        std_scores = {}

        for var_name, results in variation_results.items():
            scores = [r.score for r in results]
            mean_scores[var_name] = float(np.mean(scores)) if scores else 0
            std_scores[var_name] = float(np.std(scores)) if scores else 0

        # Best and worst variations
        best_var = max(mean_scores, key=mean_scores.get)
        worst_var = min(mean_scores, key=mean_scores.get)

        # Sensitivity metrics
        all_means = list(mean_scores.values())
        score_variance = float(np.var(all_means)) if all_means else 0
        score_range = max(all_means) - min(all_means) if all_means else 0

        # Sensitivity index: normalize variance to 0-1
        sensitivity_index = min(1.0, score_variance * 10)  # Arbitrary scaling
        is_sensitive = sensitivity_index > 0.1 or score_range > 0.1

        return SensitivityResult(
            n_variations=len(self.variations),
            n_test_cases=len(test_cases),
            variation_results=variation_results,
            mean_scores_by_variation=mean_scores,
            std_scores_by_variation=std_scores,
            best_variation=best_var,
            worst_variation=worst_var,
            score_variance=score_variance,
            score_range=score_range,
            sensitivity_index=sensitivity_index,
            is_prompt_sensitive=is_sensitive,
        )


# =============================================================================
# INSTRUCTION FOLLOWING ANALYSIS
# =============================================================================


@dataclass
class InstructionFollowingResult:
    """Result of instruction following analysis."""

    instruction_type: str
    compliance_rate: float  # How often model follows the instruction
    examples: list[dict[str, Any]]


def analyze_instruction_following(
    model_client: ModelClient,
    test_prompts: list[str],
    instruction_types: dict[str, Callable[[str], bool]],
) -> dict[str, InstructionFollowingResult]:
    """
    Analyze how well model follows different types of instructions.

    Args:
        model_client: Client for model inference
        test_prompts: Base prompts to test
        instruction_types: Dict mapping instruction type to checker function

    Returns:
        Dictionary mapping instruction type to compliance result

    Example:
        >>> checkers = {
        ...     "json_format": lambda r: r.strip().startswith("{"),
        ...     "bullet_points": lambda r: "•" in r or "-" in r,
        ...     "word_limit": lambda r: len(r.split()) <= 50,
        ... }
        >>> results = analyze_instruction_following(client, prompts, checkers)
    """
    results = {}

    for inst_type, checker in instruction_types.items():
        compliant = 0
        examples = []

        for prompt in test_prompts:
            # Add instruction to prompt
            if inst_type == "json_format":
                full_prompt = f"{prompt}\n\nRespond in JSON format."
            elif inst_type == "bullet_points":
                full_prompt = f"{prompt}\n\nUse bullet points in your response."
            elif inst_type == "word_limit":
                full_prompt = f"{prompt}\n\nLimit your response to 50 words."
            else:
                full_prompt = prompt

            response = model_client.generate(full_prompt)
            is_compliant = checker(response)

            if is_compliant:
                compliant += 1

            if len(examples) < 3:
                examples.append({
                    "prompt": full_prompt,
                    "response": response[:200],
                    "compliant": is_compliant,
                })

        compliance_rate = compliant / len(test_prompts) if test_prompts else 0

        results[inst_type] = InstructionFollowingResult(
            instruction_type=inst_type,
            compliance_rate=compliance_rate,
            examples=examples,
        )

    return results


# =============================================================================
# FORMAT SENSITIVITY
# =============================================================================


@dataclass
class FormatSensitivityResult:
    """Result of format sensitivity analysis."""

    formats_tested: list[str]
    scores_by_format: dict[str, float]
    best_format: str
    worst_format: str
    format_matters: bool


def analyze_format_sensitivity(
    model_client: ModelClient,
    evaluator: Evaluator,
    test_cases: list[dict[str, Any]],
    content_field: str = "content",
    ground_truth_field: str = "ground_truth",
) -> FormatSensitivityResult:
    """
    Analyze sensitivity to input formatting.

    Tests same content with different formatting to see if format affects output.

    Args:
        model_client: Client for model inference
        evaluator: Evaluator for scoring
        test_cases: Test cases to use
        content_field: Field with input content
        ground_truth_field: Field with ground truth

    Returns:
        FormatSensitivityResult with format comparison
    """
    formats = {
        "plain": lambda x: x,
        "markdown": lambda x: f"# Input\n\n{x}\n\n# Task\n\nAnalyze the above.",
        "xml": lambda x: f"<input>{x}</input>\n<task>Analyze the input.</task>",
        "json_like": lambda x: f'{{"input": "{x[:100]}...", "task": "analyze"}}',
        "numbered": lambda x: f"1. Input:\n{x}\n\n2. Task: Analyze.",
    }

    scores_by_format: dict[str, list[float]] = {f: [] for f in formats}

    for tc in test_cases:
        content = tc[content_field]
        ground_truth = tc[ground_truth_field]

        for fmt_name, formatter in formats.items():
            formatted_content = formatter(content)
            response = model_client.generate(formatted_content)
            score = evaluator.score(response, ground_truth)
            scores_by_format[fmt_name].append(score)

    # Compute means
    mean_scores = {f: float(np.mean(s)) for f, s in scores_by_format.items()}

    best_format = max(mean_scores, key=mean_scores.get)
    worst_format = min(mean_scores, key=mean_scores.get)

    # Format matters if range > 0.1
    score_range = max(mean_scores.values()) - min(mean_scores.values())
    format_matters = score_range > 0.1

    return FormatSensitivityResult(
        formats_tested=list(formats.keys()),
        scores_by_format=mean_scores,
        best_format=best_format,
        worst_format=worst_format,
        format_matters=format_matters,
    )


# =============================================================================
# PROMPT OPTIMIZATION
# =============================================================================


@dataclass
class OptimizedPrompt:
    """Result of prompt optimization."""

    original_prompt: str
    optimized_prompt: str
    original_score: float
    optimized_score: float
    improvement: float
    optimization_steps: list[str]


def optimize_prompt_simple(
    model_client: ModelClient,
    evaluator: Evaluator,
    base_prompt: str,
    test_cases: list[dict[str, Any]],
    ground_truth_field: str = "ground_truth",
    max_iterations: int = 5,
) -> OptimizedPrompt:
    """
    Simple prompt optimization through variation testing.

    Tests different prompt styles and selects the best performing one.

    Args:
        model_client: Client for model inference
        evaluator: Evaluator for scoring
        base_prompt: Starting prompt template
        test_cases: Test cases for evaluation
        ground_truth_field: Field with ground truth
        max_iterations: Maximum optimization iterations

    Returns:
        OptimizedPrompt with best found prompt
    """
    # Extract instruction from base prompt (simplified)
    instruction = base_prompt

    # Test variations
    variations = get_prompt_variations(instruction)

    best_prompt = base_prompt
    best_score = 0
    optimization_steps = []

    # Evaluate original
    original_scores = []
    for tc in test_cases[:10]:  # Limit for speed
        response = model_client.generate(base_prompt)
        score = evaluator.score(response, tc[ground_truth_field])
        original_scores.append(score)
    original_mean = float(np.mean(original_scores))
    best_score = original_mean
    optimization_steps.append(f"Original score: {original_mean:.3f}")

    # Test each variation
    for var_name, prompt in variations.items():
        scores = []
        for tc in test_cases[:10]:
            response = model_client.generate(prompt)
            score = evaluator.score(response, tc[ground_truth_field])
            scores.append(score)

        mean_score = float(np.mean(scores))
        optimization_steps.append(f"{var_name}: {mean_score:.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_prompt = prompt

    return OptimizedPrompt(
        original_prompt=base_prompt,
        optimized_prompt=best_prompt,
        original_score=original_mean,
        optimized_score=best_score,
        improvement=best_score - original_mean,
        optimization_steps=optimization_steps,
    )


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================


@dataclass
class ComprehensivePromptAnalysis:
    """Complete prompt analysis results."""

    sensitivity_result: SensitivityResult
    format_sensitivity: FormatSensitivityResult | None
    recommendations: list[str]
    executive_summary: str


def comprehensive_prompt_analysis(
    model_client: ModelClient,
    evaluator: Evaluator,
    test_cases: list[dict[str, Any]],
    instruction_field: str = "instruction",
    content_field: str | None = None,
    ground_truth_field: str = "ground_truth",
    verbose: bool = True,
) -> ComprehensivePromptAnalysis:
    """
    Run comprehensive prompt analysis.

    Args:
        model_client: Client for model inference
        evaluator: Evaluator for scoring
        test_cases: Test cases to analyze
        instruction_field: Field with instructions
        content_field: Optional field with content for format testing
        ground_truth_field: Field with ground truth
        verbose: Whether to print progress

    Returns:
        ComprehensivePromptAnalysis with all results
    """
    # Sensitivity analysis
    analyzer = PromptSensitivityAnalyzer(model_client, evaluator)
    sensitivity = analyzer.analyze(
        test_cases,
        instruction_field=instruction_field,
        ground_truth_field=ground_truth_field,
        verbose=verbose,
    )

    # Format sensitivity (if content field provided)
    format_result = None
    if content_field:
        format_result = analyze_format_sensitivity(
            model_client, evaluator, test_cases,
            content_field=content_field,
            ground_truth_field=ground_truth_field,
        )

    # Generate recommendations
    recommendations = []

    if sensitivity.is_prompt_sensitive:
        recommendations.append(
            f"Model is sensitive to prompt phrasing. Use '{sensitivity.best_variation}' "
            f"style for best results (score: {sensitivity.mean_scores_by_variation[sensitivity.best_variation]:.3f})"
        )
        recommendations.append(
            f"Avoid '{sensitivity.worst_variation}' style "
            f"(score: {sensitivity.mean_scores_by_variation[sensitivity.worst_variation]:.3f})"
        )
    else:
        recommendations.append(
            "Model is relatively robust to prompt variations. "
            "Simple, direct prompts should work well."
        )

    if format_result and format_result.format_matters:
        recommendations.append(
            f"Input format matters. Use '{format_result.best_format}' format "
            f"(score: {format_result.scores_by_format[format_result.best_format]:.3f})"
        )

    # Executive summary
    summary_parts = []
    if sensitivity.is_prompt_sensitive:
        summary_parts.append(f"Model shows HIGH prompt sensitivity (index: {sensitivity.sensitivity_index:.2f}).")
    else:
        summary_parts.append(f"Model shows LOW prompt sensitivity (index: {sensitivity.sensitivity_index:.2f}).")

    summary_parts.append(
        f"Best performing prompt style: {sensitivity.best_variation} "
        f"({sensitivity.mean_scores_by_variation[sensitivity.best_variation]:.3f})."
    )

    if format_result:
        if format_result.format_matters:
            summary_parts.append(f"Input format significantly affects performance.")
        else:
            summary_parts.append("Input format has minimal effect on performance.")

    executive_summary = " ".join(summary_parts)

    return ComprehensivePromptAnalysis(
        sensitivity_result=sensitivity,
        format_sensitivity=format_result,
        recommendations=recommendations,
        executive_summary=executive_summary,
    )
