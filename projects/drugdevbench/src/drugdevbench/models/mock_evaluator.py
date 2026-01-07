"""Mock evaluator for testing pipeline without API calls."""

import random
import time
from pathlib import Path

from drugdevbench.data.schemas import EvaluationResponse, PromptCondition, QuestionType
from drugdevbench.models.litellm_wrapper import EvaluatorConfig


# Mock responses based on question type
MOCK_RESPONSES = {
    QuestionType.FACTUAL_EXTRACTION: [
        "Based on the figure, the value is approximately 2.5 nM.",
        "The IC50 appears to be around 450 nM from the curve.",
        "The half-life is estimated at 4.2 hours.",
        "According to the legend, n=3 independent experiments were performed.",
    ],
    QuestionType.VISUAL_ESTIMATION: [
        "From visual inspection, I estimate the value to be approximately 3.0 μM.",
        "The Cmax appears to be around 1200 ng/mL based on the peak.",
        "I estimate approximately 75% of cells are in the positive quadrant.",
    ],
    QuestionType.QUALITY_ASSESSMENT: [
        "Yes, a loading control (β-actin) is clearly visible and appears consistent across lanes.",
        "The loading control shows some variability between lanes 3 and 4.",
        "No, I do not see a proper loading control in this blot.",
    ],
    QuestionType.INTERPRETATION: [
        "The data suggests dose-dependent inhibition with good curve fit.",
        "This indicates successful target engagement at the concentrations tested.",
        "The results demonstrate a clear pharmacokinetic profile with expected clearance.",
    ],
    QuestionType.ERROR_DETECTION: [
        "No significant quality concerns are apparent in this figure.",
        "The high background in lane 2 may affect quantification accuracy.",
        "Some bands appear overexposed which could limit dynamic range assessment.",
    ],
}


class MockEvaluator:
    """Mock evaluator that generates realistic responses without API calls.

    Used for testing the pipeline and generating sample reports.
    """

    def __init__(self, config: EvaluatorConfig | None = None):
        """Initialize the mock evaluator.

        Args:
            config: Evaluator configuration (mostly ignored for mock)
        """
        self.config = config or EvaluatorConfig()
        self._call_count = 0

    def evaluate(
        self,
        figure_id: str,
        question_id: str,
        image_path: str | Path,
        question: str,
        system_prompt: str,
        condition: PromptCondition,
        model: str | None = None,
        gold_answer: str = "",
        question_type: QuestionType | None = None,
    ) -> EvaluationResponse:
        """Generate a mock evaluation response.

        Args:
            figure_id: ID of the figure
            question_id: ID of the question
            image_path: Path to the figure image
            question: Question to ask
            system_prompt: System prompt (affects mock response quality)
            condition: Prompt condition being evaluated
            model: Model name (used for mock response variation)
            gold_answer: Expected correct answer
            question_type: Type of question (for realistic responses)

        Returns:
            Mock EvaluationResponse
        """
        self._call_count += 1
        model = model or self.config.default_model

        # Determine question type from question text if not provided
        if question_type is None:
            question_type = self._infer_question_type(question)

        # Generate mock response
        response_text = self._generate_mock_response(
            question_type, condition, gold_answer
        )

        # Simulate realistic token counts
        prompt_tokens = len(system_prompt.split()) * 2 + 500  # Rough estimate
        completion_tokens = len(response_text.split()) * 2

        # Estimate cost (mock values)
        cost_usd = (prompt_tokens * 0.001 + completion_tokens * 0.002) / 1000

        # Simulate latency
        latency_ms = random.randint(200, 800)

        return EvaluationResponse(
            figure_id=figure_id,
            question_id=question_id,
            model=model,
            condition=condition,
            response_text=response_text,
            gold_answer=gold_answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cached=False,
            metadata={
                "mock": True,
                "question_type": question_type.value if question_type else "unknown",
            },
        )

    def _infer_question_type(self, question: str) -> QuestionType:
        """Infer question type from question text."""
        question_lower = question.lower()

        if any(kw in question_lower for kw in ["what is", "how many", "reported"]):
            return QuestionType.FACTUAL_EXTRACTION
        elif any(kw in question_lower for kw in ["estimate", "approximate", "visual"]):
            return QuestionType.VISUAL_ESTIMATION
        elif any(kw in question_lower for kw in ["control", "appropriate", "quality"]):
            return QuestionType.QUALITY_ASSESSMENT
        elif any(kw in question_lower for kw in ["suggest", "indicate", "mean"]):
            return QuestionType.INTERPRETATION
        elif any(kw in question_lower for kw in ["concern", "error", "issue", "problem"]):
            return QuestionType.ERROR_DETECTION
        else:
            return QuestionType.FACTUAL_EXTRACTION

    def _generate_mock_response(
        self,
        question_type: QuestionType,
        condition: PromptCondition,
        gold_answer: str,
    ) -> str:
        """Generate a mock response based on question type and condition.

        Better conditions (full_stack) produce responses closer to gold answers.
        """
        responses = MOCK_RESPONSES.get(question_type, MOCK_RESPONSES[QuestionType.FACTUAL_EXTRACTION])

        # Base accuracy depends on condition
        accuracy_boost = {
            PromptCondition.VANILLA: 0.3,
            PromptCondition.BASE_ONLY: 0.5,
            PromptCondition.PERSONA_ONLY: 0.45,
            PromptCondition.BASE_PLUS_SKILL: 0.7,
            PromptCondition.FULL_STACK: 0.85,
            PromptCondition.WRONG_SKILL: 0.35,
        }

        base_accuracy = accuracy_boost.get(condition, 0.5)

        # Sometimes include the gold answer in the response
        if gold_answer and gold_answer != "[TO BE FILLED]":
            if random.random() < base_accuracy:
                # Include gold answer with some framing
                return f"Based on my analysis, {gold_answer}. {random.choice(responses)}"

        return random.choice(responses)

    @property
    def call_count(self) -> int:
        """Number of evaluate calls made."""
        return self._call_count
