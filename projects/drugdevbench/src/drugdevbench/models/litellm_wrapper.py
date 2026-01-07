"""LiteLLM wrapper for unified model interface."""

import base64
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

from drugdevbench.data.schemas import EvaluationResponse, PromptCondition
from drugdevbench.models.cache import ResponseCache

# Load environment variables
load_dotenv()


@dataclass
class ModelCost:
    """Cost information for a model."""

    input_per_1k: float  # Cost per 1K input tokens
    output_per_1k: float  # Cost per 1K output tokens
    tier: str  # Cost tier: "$", "$$", "$$$"


# Supported models with their LiteLLM keys and costs
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    # Claude models
    "claude-sonnet": {
        "litellm_key": "claude-3-5-sonnet-20241022",
        "cost": ModelCost(input_per_1k=0.003, output_per_1k=0.015, tier="$$"),
        "vision": True,
    },
    "claude-haiku": {
        "litellm_key": "claude-3-haiku-20240307",
        "cost": ModelCost(input_per_1k=0.00025, output_per_1k=0.00125, tier="$"),
        "vision": True,
    },
    # Gemini models
    "gemini-pro": {
        "litellm_key": "gemini/gemini-1.5-pro",
        "cost": ModelCost(input_per_1k=0.00125, output_per_1k=0.005, tier="$$"),
        "vision": True,
    },
    "gemini-flash": {
        "litellm_key": "gemini/gemini-1.5-flash",
        "cost": ModelCost(input_per_1k=0.000075, output_per_1k=0.0003, tier="$"),
        "vision": True,
    },
    # OpenAI models
    "gpt-4o": {
        "litellm_key": "gpt-4o",
        "cost": ModelCost(input_per_1k=0.005, output_per_1k=0.015, tier="$$"),
        "vision": True,
    },
    "gpt-4o-mini": {
        "litellm_key": "gpt-4o-mini",
        "cost": ModelCost(input_per_1k=0.00015, output_per_1k=0.0006, tier="$"),
        "vision": True,
    },
}


class EvaluatorConfig(BaseModel):
    """Configuration for the DrugDevBench evaluator."""

    default_model: str = "claude-haiku"
    use_cache: bool = True
    cache_dir: Path = Path("data/cache/responses")
    max_tokens: int = 1024
    temperature: float = 0.0


def _encode_image(image_path: Path | str) -> str:
    """Encode an image file to base64.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded image string
    """
    path = Path(image_path)
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _get_image_media_type(image_path: Path | str) -> str:
    """Get the media type for an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Media type string (e.g., 'image/png')
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(suffix, "image/png")


def _compute_cache_key(
    model: str,
    system_prompt: str,
    question: str,
    image_path: str,
) -> str:
    """Compute a cache key for a request.

    Args:
        model: Model name
        system_prompt: System prompt
        question: Question text
        image_path: Path to image

    Returns:
        Cache key string
    """
    # Hash the image content
    with open(image_path, "rb") as f:
        image_hash = hashlib.md5(f.read()).hexdigest()

    # Hash all components
    content = f"{model}|{system_prompt}|{question}|{image_hash}"
    return hashlib.sha256(content.encode()).hexdigest()


class DrugDevBenchEvaluator:
    """Evaluator for running DrugDevBench using LiteLLM."""

    def __init__(self, config: EvaluatorConfig | None = None):
        """Initialize the evaluator.

        Args:
            config: Evaluator configuration
        """
        self.config = config or EvaluatorConfig()
        self.cache = ResponseCache(self.config.cache_dir) if self.config.use_cache else None

    def _get_model_key(self, model: str) -> str:
        """Get the LiteLLM key for a model.

        Args:
            model: Short model name

        Returns:
            LiteLLM model key
        """
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model}. Supported: {list(SUPPORTED_MODELS.keys())}")
        return SUPPORTED_MODELS[model]["litellm_key"]

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the cost of a request.

        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if model not in SUPPORTED_MODELS:
            return 0.0
        cost = SUPPORTED_MODELS[model]["cost"]
        return (prompt_tokens * cost.input_per_1k / 1000) + (
            completion_tokens * cost.output_per_1k / 1000
        )

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
    ) -> EvaluationResponse:
        """Evaluate a figure with a question.

        Args:
            figure_id: ID of the figure
            question_id: ID of the question
            image_path: Path to the figure image
            question: Question to ask about the figure
            system_prompt: System prompt to use
            condition: Prompt condition being evaluated
            model: Model to use (defaults to config default)
            gold_answer: Expected correct answer

        Returns:
            EvaluationResponse with the model's answer
        """
        model = model or self.config.default_model
        image_path = Path(image_path)

        # Check cache
        cache_key = _compute_cache_key(model, system_prompt, question, str(image_path))
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return EvaluationResponse(
                    figure_id=figure_id,
                    question_id=question_id,
                    model=model,
                    condition=condition,
                    response_text=cached["response_text"],
                    gold_answer=gold_answer,
                    prompt_tokens=cached.get("prompt_tokens"),
                    completion_tokens=cached.get("completion_tokens"),
                    cost_usd=cached.get("cost_usd"),
                    cached=True,
                )

        # Prepare image for API
        image_base64 = _encode_image(image_path)
        media_type = _get_image_media_type(image_path)

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]

        # Make API call
        start_time = time.time()
        try:
            response = litellm.completion(
                model=self._get_model_key(model),
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as e:
            # Return error response
            return EvaluationResponse(
                figure_id=figure_id,
                question_id=question_id,
                model=model,
                condition=condition,
                response_text=f"ERROR: {str(e)}",
                gold_answer=gold_answer,
                cached=False,
                metadata={"error": str(e)},
            )

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract response data
        response_text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost_usd = self._estimate_cost(model, prompt_tokens, completion_tokens)

        # Cache the response
        if self.cache:
            self.cache.set(
                cache_key,
                {
                    "response_text": response_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost_usd,
                },
            )

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
        )


def estimate_benchmark_cost(
    n_figures: int,
    n_questions_per_figure: int,
    n_conditions: int,
    models: list[str],
    avg_prompt_tokens: int = 2000,  # Including image
    avg_completion_tokens: int = 200,
) -> dict[str, Any]:
    """Estimate the cost of running a benchmark.

    Args:
        n_figures: Number of figures
        n_questions_per_figure: Questions per figure
        n_conditions: Number of prompt conditions
        models: List of models to use
        avg_prompt_tokens: Average prompt tokens per request
        avg_completion_tokens: Average completion tokens per request

    Returns:
        Dictionary with cost estimates
    """
    total_requests = n_figures * n_questions_per_figure * n_conditions * len(models)
    costs_by_model = {}
    total_cost = 0.0

    for model in models:
        if model not in SUPPORTED_MODELS:
            continue
        cost_info = SUPPORTED_MODELS[model]["cost"]
        model_requests = n_figures * n_questions_per_figure * n_conditions
        model_cost = model_requests * (
            (avg_prompt_tokens * cost_info.input_per_1k / 1000)
            + (avg_completion_tokens * cost_info.output_per_1k / 1000)
        )
        costs_by_model[model] = model_cost
        total_cost += model_cost

    return {
        "total_requests": total_requests,
        "costs_by_model": costs_by_model,
        "total_cost_usd": total_cost,
        "assumptions": {
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
        },
    }
