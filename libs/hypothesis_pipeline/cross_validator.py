"""
Cross-Validation Subagent for LLM Output Validation.

Uses a powerful model (Opus) to debate and validate LLM predictions against
ground truth, providing a 0-100 confidence score on prediction correctness.

Design:
- Self-debate pattern: Single model argues both perspectives then reconciles
- 0-100 confidence score: How strongly validator leans toward prediction vs ground truth
  - 0-30: Ground truth is clearly correct, prediction has significant issues
  - 31-50: Ground truth appears more correct, but prediction has some merit
  - 51-70: Both are reasonable, or prediction is acceptable alternative
  - 71-90: Prediction appears correct, or better than ground truth
  - 91-100: Prediction is clearly correct, ground truth may have issues

Future work:
- Escalation to multi-model debate when confidence is uncertain (40-60)
- Advocate-Judge pattern for high-stakes validation
- Iterative refinement for complex discrepancies
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


class ValidatorModel(Protocol):
    """Protocol for validator model clients."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Generate a validation response from the model."""
        ...


@dataclass
class Discrepancy:
    """A single point of disagreement between prediction and ground truth."""

    aspect: str  # e.g., "gate_name", "parent_relationship", "marker_used"
    prediction_value: str
    ground_truth_value: str
    severity: str  # "critical", "moderate", "minor"

    # Debate results
    leaning: int  # 0-100: 0=GT correct, 100=prediction correct
    reasoning: str


@dataclass
class CrossValidationResult:
    """
    Complete result of cross-validation debate.

    The confidence_score is the key output:
    - 0-100 scale indicating how much the validator agrees with the prediction
    - Lower scores mean ground truth is more correct
    - Higher scores mean prediction is more correct or acceptable
    """

    # Core assessment
    confidence_score: int  # 0-100: How much validator agrees with prediction

    # Discrepancy analysis
    discrepancies: list[Discrepancy] = field(default_factory=list)
    n_critical: int = 0
    n_moderate: int = 0
    n_minor: int = 0

    # Debate reasoning
    prediction_defense: str = ""  # Why prediction might be correct
    ground_truth_defense: str = ""  # Why ground truth is correct
    reconciliation: str = ""  # Final synthesis

    # Metadata
    validator_model: str = "unknown"
    raw_response: str = field(repr=False, default="")

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of the confidence score."""
        if self.confidence_score <= 30:
            return "ground_truth_clearly_correct"
        elif self.confidence_score <= 50:
            return "ground_truth_preferred"
        elif self.confidence_score <= 70:
            return "both_acceptable"
        elif self.confidence_score <= 90:
            return "prediction_preferred"
        else:
            return "prediction_clearly_correct"

    @property
    def should_trust_prediction(self) -> bool:
        """Whether the prediction is acceptable (score > 50)."""
        return self.confidence_score > 50

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Confidence Score: {self.confidence_score}/100 ({self.interpretation})",
            f"Discrepancies: {self.n_critical} critical, {self.n_moderate} moderate, {self.n_minor} minor",
            "",
            "Reconciliation:",
            self.reconciliation[:500] + "..." if len(self.reconciliation) > 500 else self.reconciliation,
        ]
        return "\n".join(lines)


class CrossValidator:
    """
    Cross-validation through self-debate.

    Uses a powerful model to:
    1. Identify all discrepancies between prediction and ground truth
    2. Argue FOR the prediction (devil's advocate)
    3. Argue FOR the ground truth (prosecution)
    4. Reconcile and produce a 0-100 confidence score
    """

    DEBATE_PROMPT = """You are an expert flow cytometry immunologist validating a gating strategy prediction.

Your task is to compare an LLM's predicted gating hierarchy against the ground truth and determine
how correct the prediction is on a 0-100 scale.

## Scoring Scale
- 0-30: Ground truth is clearly correct, prediction has significant errors
- 31-50: Ground truth appears more correct, but prediction has some valid elements
- 51-70: Both are reasonable approaches, or prediction is an acceptable alternative
- 71-90: Prediction appears correct, possibly better than ground truth
- 91-100: Prediction is clearly correct, ground truth may have issues

## Context
{context}

## Ground Truth Gating Hierarchy
```
{ground_truth}
```

## LLM Prediction
```
{prediction}
```

## Your Analysis

Perform a structured debate:

1. **Identify Discrepancies**: List EVERY difference between prediction and ground truth
   - Gate naming differences
   - Hierarchy/parent-child differences
   - Missing gates
   - Extra gates
   - Marker usage differences

2. **Argue FOR the Prediction** (Devil's Advocate):
   - What biological or technical reasons might justify the prediction's choices?
   - Are any "errors" actually valid alternative approaches?
   - Could the prediction be following a different but valid gating convention?

3. **Argue FOR the Ground Truth** (Prosecution):
   - Why is the ground truth the correct approach?
   - What critical errors does the prediction make?
   - What biological principles does the prediction violate?

4. **Reconcile**:
   - For each discrepancy, who is right? (0-100 sub-score)
   - What is the overall assessment?
   - Provide final 0-100 confidence score

## Output as JSON

```json
{{
  "discrepancies": [
    {{
      "aspect": "string describing what differs",
      "prediction_value": "what prediction has",
      "ground_truth_value": "what ground truth has",
      "severity": "critical|moderate|minor",
      "leaning": <0-100: 0=GT correct, 100=prediction correct>,
      "reasoning": "brief explanation"
    }}
  ],
  "prediction_defense": "Argument for why prediction might be correct...",
  "ground_truth_defense": "Argument for why ground truth is correct...",
  "reconciliation": "Final synthesis of both arguments...",
  "confidence_score": <0-100>
}}
```

Analyze carefully and provide your JSON response:"""

    def __init__(
        self,
        validator_model: ValidatorModel,
        model_name: str = "unknown",
    ):
        """
        Initialize cross-validator.

        Args:
            validator_model: Model client for validation (preferably Opus)
            model_name: Name of validator model for logging
        """
        self.model = validator_model
        self.model_name = model_name

    def validate(
        self,
        prediction: str | dict | list,
        ground_truth: str | dict | list,
        context: str = "",
    ) -> CrossValidationResult:
        """
        Run cross-validation debate on prediction vs ground truth.

        Args:
            prediction: LLM prediction (can be string, dict, or structured)
            ground_truth: Ground truth (can be string, dict, or structured)
            context: Additional context (e.g., panel info, experiment type)

        Returns:
            CrossValidationResult with 0-100 confidence score
        """
        # Format inputs
        pred_str = self._format_for_prompt(prediction)
        gt_str = self._format_for_prompt(ground_truth)

        # Build prompt
        prompt = self.DEBATE_PROMPT.format(
            context=context if context else "No additional context provided.",
            ground_truth=gt_str,
            prediction=pred_str,
        )

        # Get validation
        raw_response = self.model.generate(prompt, max_tokens=4096)

        # Parse response
        return self._parse_response(raw_response)

    def validate_gating_result(
        self,
        result_dict: dict[str, Any],
        ground_truth_hierarchy: dict | str,
        panel_markers: list[str] | None = None,
    ) -> CrossValidationResult:
        """
        Validate a gating benchmark result.

        Convenience method for validating experiment results.

        Args:
            result_dict: Result dictionary from benchmark (with predicted_gates, etc.)
            ground_truth_hierarchy: Ground truth gating hierarchy
            panel_markers: Optional list of panel markers for context

        Returns:
            CrossValidationResult
        """
        # Extract prediction info from result
        prediction_info = {
            "predicted_gates": result_dict.get("predicted_gates", []),
            "parsed_hierarchy": result_dict.get("parsed_hierarchy", {}),
        }

        # Build context
        context_parts = []
        if panel_markers:
            context_parts.append(f"Panel markers: {', '.join(panel_markers)}")
        if result_dict.get("test_case_id"):
            context_parts.append(f"Test case: {result_dict['test_case_id']}")

        context = "\n".join(context_parts)

        return self.validate(
            prediction=prediction_info,
            ground_truth=ground_truth_hierarchy,
            context=context,
        )

    def _format_for_prompt(self, data: str | dict | list) -> str:
        """Format data for inclusion in prompt."""
        if isinstance(data, str):
            return data
        return json.dumps(data, indent=2)

    def _parse_response(self, raw_response: str) -> CrossValidationResult:
        """Parse the validator's response into structured result."""
        # Extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)```', raw_response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                json_str = json_match.group(0)
            else:
                # No JSON found at all
                return CrossValidationResult(
                    confidence_score=50,  # Uncertain
                    reconciliation="Failed to parse validator response: no JSON found",
                    validator_model=self.model_name,
                    raw_response=raw_response,
                )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Return default result on parse failure
            return CrossValidationResult(
                confidence_score=50,  # Uncertain
                reconciliation="Failed to parse validator response: invalid JSON",
                validator_model=self.model_name,
                raw_response=raw_response,
            )

        # Parse discrepancies
        discrepancies = []
        n_critical = n_moderate = n_minor = 0

        for d in data.get("discrepancies", []):
            severity = d.get("severity", "minor")
            if severity == "critical":
                n_critical += 1
            elif severity == "moderate":
                n_moderate += 1
            else:
                n_minor += 1

            discrepancies.append(Discrepancy(
                aspect=d.get("aspect", "unknown"),
                prediction_value=str(d.get("prediction_value", "")),
                ground_truth_value=str(d.get("ground_truth_value", "")),
                severity=severity,
                leaning=int(d.get("leaning", 50)),
                reasoning=d.get("reasoning", ""),
            ))

        # Extract confidence score
        confidence_score = data.get("confidence_score", 50)
        # Clamp to valid range
        confidence_score = max(0, min(100, int(confidence_score)))

        return CrossValidationResult(
            confidence_score=confidence_score,
            discrepancies=discrepancies,
            n_critical=n_critical,
            n_moderate=n_moderate,
            n_minor=n_minor,
            prediction_defense=data.get("prediction_defense", ""),
            ground_truth_defense=data.get("ground_truth_defense", ""),
            reconciliation=data.get("reconciliation", ""),
            validator_model=self.model_name,
            raw_response=raw_response,
        )


class BatchCrossValidator:
    """
    Batch cross-validation for multiple results.

    Provides aggregate statistics and filtering capabilities.
    """

    def __init__(self, validator: CrossValidator):
        """Initialize with a CrossValidator instance."""
        self.validator = validator

    def validate_batch(
        self,
        results: list[dict[str, Any]],
        ground_truths: dict[str, dict],  # test_case_id -> ground truth
        panel_markers: dict[str, list[str]] | None = None,  # test_case_id -> markers
        progress_callback: callable | None = None,
    ) -> list[tuple[dict, CrossValidationResult]]:
        """
        Validate a batch of results.

        Args:
            results: List of result dictionaries
            ground_truths: Mapping of test_case_id to ground truth
            panel_markers: Optional mapping of test_case_id to panel markers
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of (result_dict, validation_result) tuples
        """
        validated = []
        total = len(results)

        for i, result in enumerate(results):
            test_case_id = result.get("test_case_id")

            if test_case_id not in ground_truths:
                continue

            gt = ground_truths[test_case_id]
            markers = panel_markers.get(test_case_id) if panel_markers else None

            validation = self.validator.validate_gating_result(
                result_dict=result.get("evaluation", result),
                ground_truth_hierarchy=gt,
                panel_markers=markers,
            )

            validated.append((result, validation))

            if progress_callback:
                progress_callback(i + 1, total)

        return validated

    def compute_statistics(
        self,
        validations: list[tuple[dict, CrossValidationResult]],
    ) -> dict[str, Any]:
        """
        Compute aggregate statistics from validation results.

        Args:
            validations: List of (result, validation) tuples

        Returns:
            Dictionary of aggregate statistics
        """
        if not validations:
            return {"error": "No validations to aggregate"}

        scores = [v.confidence_score for _, v in validations]

        # Count by interpretation
        interpretations = {}
        for _, v in validations:
            interp = v.interpretation
            interpretations[interp] = interpretations.get(interp, 0) + 1

        # Discrepancy counts
        total_critical = sum(v.n_critical for _, v in validations)
        total_moderate = sum(v.n_moderate for _, v in validations)
        total_minor = sum(v.n_minor for _, v in validations)

        return {
            "n_samples": len(validations),
            "mean_confidence": sum(scores) / len(scores),
            "min_confidence": min(scores),
            "max_confidence": max(scores),
            "median_confidence": sorted(scores)[len(scores) // 2],
            "std_confidence": (
                sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
            ) ** 0.5,
            "interpretations": interpretations,
            "predictions_trusted": sum(1 for _, v in validations if v.should_trust_prediction),
            "predictions_rejected": sum(1 for _, v in validations if not v.should_trust_prediction),
            "total_critical_discrepancies": total_critical,
            "total_moderate_discrepancies": total_moderate,
            "total_minor_discrepancies": total_minor,
        }

    def filter_by_confidence(
        self,
        validations: list[tuple[dict, CrossValidationResult]],
        min_score: int | None = None,
        max_score: int | None = None,
    ) -> list[tuple[dict, CrossValidationResult]]:
        """Filter validations by confidence score range."""
        filtered = []
        for result, validation in validations:
            if min_score is not None and validation.confidence_score < min_score:
                continue
            if max_score is not None and validation.confidence_score > max_score:
                continue
            filtered.append((result, validation))
        return filtered

    def get_controversial(
        self,
        validations: list[tuple[dict, CrossValidationResult]],
        range_min: int = 40,
        range_max: int = 60,
    ) -> list[tuple[dict, CrossValidationResult]]:
        """Get validations where confidence is uncertain (default: 40-60)."""
        return self.filter_by_confidence(validations, min_score=range_min, max_score=range_max)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_anthropic_validator(
    api_key: str | None = None,
    model: str = "claude-opus-4-20250514",
) -> CrossValidator:
    """
    Create a CrossValidator using Anthropic's API.

    Args:
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        model: Model to use (default: claude-opus-4)

    Returns:
        Configured CrossValidator
    """
    import os

    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter required")

    client = anthropic.Anthropic(api_key=api_key)

    class AnthropicValidatorModel:
        def __init__(self, client, model_name):
            self.client = client
            self.model_name = model_name

        def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

    validator_model = AnthropicValidatorModel(client, model)
    return CrossValidator(validator_model, model_name=model)


def quick_validate(
    prediction: str | dict,
    ground_truth: str | dict,
    context: str = "",
    model: str = "claude-opus-4-20250514",
) -> int:
    """
    Quick validation returning just the confidence score.

    Args:
        prediction: LLM prediction
        ground_truth: Ground truth
        context: Additional context
        model: Model to use

    Returns:
        Confidence score 0-100
    """
    validator = create_anthropic_validator(model=model)
    result = validator.validate(prediction, ground_truth, context)
    return result.confidence_score
