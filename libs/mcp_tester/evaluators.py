"""
Evaluators for scoring model outputs.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import re
import json


class Evaluator(ABC):
    """Base class for evaluators."""

    @abstractmethod
    def extract(self, response: str) -> Any:
        """Extract structured output from raw response."""
        pass

    @abstractmethod
    def score(self, extracted: Any, ground_truth: Any) -> dict[str, float]:
        """Score extracted output against ground truth."""
        pass


class AccuracyEvaluator(Evaluator):
    """
    Evaluator for exact-match accuracy on dict outputs.

    Extracts JSON from response and computes key-level accuracy.
    """

    def __init__(self, json_key: str = "assignments"):
        self.json_key = json_key

    def extract(self, response: str) -> dict:
        """Extract JSON dict from response."""
        # Try to find JSON in response
        patterns = [
            rf'\{{\s*"{self.json_key}"\s*:\s*(\{{[^{{}}]*\}})',
            r'\{[^{}]*\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    text = match.group()
                    data = json.loads(text)
                    if isinstance(data, dict):
                        if self.json_key in data:
                            return data[self.json_key]
                        return data
                except json.JSONDecodeError:
                    continue

        return {}

    def score(self, extracted: dict, ground_truth: dict) -> dict[str, float]:
        """Compute accuracy metrics."""
        if not ground_truth:
            return {"accuracy": 0.0}

        correct = sum(
            1 for k, v in ground_truth.items()
            if extracted.get(k) == v
        )

        return {
            "accuracy": correct / len(ground_truth),
            "correct_count": float(correct),
            "total_count": float(len(ground_truth))
        }


class SimilarityEvaluator(Evaluator):
    """
    Evaluator using custom similarity function.

    Args:
        extractor: Function to extract output from response
        scorer: Function(extracted, ground_truth) -> dict of scores
    """

    def __init__(
        self,
        extractor: Callable[[str], Any],
        scorer: Callable[[Any, Any], dict[str, float]]
    ):
        self._extractor = extractor
        self._scorer = scorer

    def extract(self, response: str) -> Any:
        return self._extractor(response)

    def score(self, extracted: Any, ground_truth: Any) -> dict[str, float]:
        return self._scorer(extracted, ground_truth)


class NumericEvaluator(Evaluator):
    """
    Evaluator for numeric outputs (e.g., scores, rankings).
    """

    def __init__(self, pattern: str = r"[-+]?\d*\.?\d+"):
        self.pattern = pattern

    def extract(self, response: str) -> float | None:
        match = re.search(self.pattern, response)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None

    def score(self, extracted: float | None, ground_truth: float) -> dict[str, float]:
        if extracted is None:
            return {"error": 1.0, "mae": float('inf')}

        error = abs(extracted - ground_truth)
        return {
            "mae": error,
            "error": error / abs(ground_truth) if ground_truth != 0 else error
        }


class CompositeEvaluator(Evaluator):
    """
    Combine multiple evaluators.
    """

    def __init__(self, evaluators: dict[str, Evaluator]):
        self.evaluators = evaluators

    def extract(self, response: str) -> dict[str, Any]:
        return {
            name: ev.extract(response)
            for name, ev in self.evaluators.items()
        }

    def score(self, extracted: dict, ground_truth: dict) -> dict[str, float]:
        scores = {}
        for name, ev in self.evaluators.items():
            ev_scores = ev.score(
                extracted.get(name),
                ground_truth.get(name)
            )
            for metric, value in ev_scores.items():
                scores[f"{name}_{metric}"] = value
        return scores
