"""
LiteLLM-based runner for local model evaluation.

Provides a unified interface for running ablation studies across:
- Anthropic models (Claude)
- OpenAI models (GPT-4)
- Local models via Ollama (Llama, DeepSeek)
- Hugging Face models

Usage:
    # For Anthropic models
    runner = LiteLLMAblationRunner(model="anthropic/claude-sonnet-4-20250514")

    # For local Llama via Ollama
    runner = LiteLLMAblationRunner(model="ollama/llama3.1:70b")

    # For DeepSeek via Ollama
    runner = LiteLLMAblationRunner(model="ollama/deepseek-r1:32b")

    # For OpenAI
    runner = LiteLLMAblationRunner(model="openai/gpt-4-turbo")

Note: This is a placeholder implementation. To use local models:
1. Install litellm: pip install litellm
2. For Ollama models: Install and run Ollama, then pull the model
3. Configure appropriate API keys/endpoints

Example Ollama setup:
    curl https://ollama.ai/install.sh | sh
    ollama pull llama3.1:70b
    ollama pull deepseek-r1:32b
"""

from dataclasses import dataclass
from typing import Optional, Any
import json
import time
import re
import os
from datetime import datetime
from pathlib import Path

# Placeholder imports - litellm would be imported here
# from litellm import completion, acompletion

from .test_cases import PanelDesignTestCase, TestSuite, TestCaseType
from .conditions import ExperimentalCondition, CONDITIONS, RetrievalMode
from .runner import TrialResult, ExperimentResults, OMIP_CORPUS


# Supported model providers and their configurations
SUPPORTED_MODELS = {
    # Anthropic Claude models
    "anthropic/claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "supports_tools": True,
        "max_tokens": 4096,
        "context_window": 200000,
    },
    "anthropic/claude-opus-4-20250514": {
        "provider": "anthropic",
        "supports_tools": True,
        "max_tokens": 4096,
        "context_window": 200000,
    },

    # OpenAI models
    "openai/gpt-4-turbo": {
        "provider": "openai",
        "supports_tools": True,
        "max_tokens": 4096,
        "context_window": 128000,
    },
    "openai/gpt-4o": {
        "provider": "openai",
        "supports_tools": True,
        "max_tokens": 4096,
        "context_window": 128000,
    },

    # Local models via Ollama (placeholders)
    "ollama/llama3.1:70b": {
        "provider": "ollama",
        "supports_tools": False,  # Tool use varies by model
        "max_tokens": 4096,
        "context_window": 128000,
        "api_base": "http://localhost:11434",
    },
    "ollama/llama3.1:8b": {
        "provider": "ollama",
        "supports_tools": False,
        "max_tokens": 4096,
        "context_window": 128000,
        "api_base": "http://localhost:11434",
    },
    "ollama/deepseek-r1:32b": {
        "provider": "ollama",
        "supports_tools": False,
        "max_tokens": 4096,
        "context_window": 64000,
        "api_base": "http://localhost:11434",
    },
    "ollama/deepseek-r1:7b": {
        "provider": "ollama",
        "supports_tools": False,
        "max_tokens": 4096,
        "context_window": 64000,
        "api_base": "http://localhost:11434",
    },
    "ollama/deepseek-coder-v2:16b": {
        "provider": "ollama",
        "supports_tools": False,
        "max_tokens": 4096,
        "context_window": 128000,
        "api_base": "http://localhost:11434",
    },

    # Hugging Face / vLLM (placeholders)
    "huggingface/meta-llama/Meta-Llama-3.1-70B-Instruct": {
        "provider": "huggingface",
        "supports_tools": False,
        "max_tokens": 4096,
        "context_window": 128000,
    },
}


class LiteLLMNotInstalledError(Exception):
    """Raised when litellm is required but not installed."""
    pass


class LocalModelNotAvailableError(Exception):
    """Raised when a local model is not running or accessible."""
    pass


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    provider: str
    supports_tools: bool
    max_tokens: int
    context_window: int
    api_base: Optional[str] = None
    api_key: Optional[str] = None


def get_model_config(model: str) -> ModelConfig:
    """Get configuration for a model, supporting custom models."""
    if model in SUPPORTED_MODELS:
        config = SUPPORTED_MODELS[model]
        return ModelConfig(
            model_id=model,
            provider=config["provider"],
            supports_tools=config["supports_tools"],
            max_tokens=config["max_tokens"],
            context_window=config["context_window"],
            api_base=config.get("api_base"),
        )

    # Handle custom model strings like "ollama/custom-model"
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return ModelConfig(
            model_id=model,
            provider=provider,
            supports_tools=False,  # Conservative default
            max_tokens=4096,
            context_window=8192,
            api_base="http://localhost:11434" if provider == "ollama" else None,
        )

    raise ValueError(f"Unknown model: {model}. Use format 'provider/model-name'")


class LiteLLMAblationRunner:
    """
    Unified ablation runner using LiteLLM for multi-provider support.

    This is a PLACEHOLDER implementation that demonstrates the interface.
    Actual litellm integration requires:
    1. pip install litellm
    2. Setting up appropriate API keys
    3. For Ollama: running the Ollama server locally

    Example:
        # Once litellm is installed:
        runner = LiteLLMAblationRunner(model="ollama/llama3.1:70b")
        results = runner.run_full_study(test_suite, conditions)
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-20250514",
        max_concurrent: int = 5,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.config = get_model_config(model)
        self.max_concurrent = max_concurrent

        # Set API key based on provider
        if api_key:
            self.api_key = api_key
        elif self.config.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif self.config.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            self.api_key = None

        # Check if litellm is available
        self._litellm_available = self._check_litellm()

        # For non-litellm fallback, use anthropic directly if available
        self._fallback_client = None
        if not self._litellm_available and self.config.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._fallback_client = Anthropic(api_key=self.api_key)
            except ImportError:
                pass

    def _check_litellm(self) -> bool:
        """Check if litellm is installed."""
        try:
            import litellm
            return True
        except ImportError:
            return False

    def _build_prompt(
        self,
        test_case: PanelDesignTestCase,
        condition: ExperimentalCondition
    ) -> str:
        """Construct prompt based on condition settings."""
        base_prompt = f"""You are designing a flow cytometry panel.

{test_case.biological_question}

Required markers and their expression levels:
{json.dumps(test_case.marker_expression, indent=2)}

Available fluorophores: All standard flow cytometry fluorophores (PE, FITC, APC, BV series, BUV series, etc.)

Instructions:
1. Assign each marker to exactly one fluorophore
2. Minimize spectral overlap between fluorophores
3. Match brighter fluorophores to lower-expression markers
4. Avoid using fluorophores with high spectral similarity on co-expressed markers

Return your answer as JSON:
{{"assignments": {{"marker1": "fluorophore1", "marker2": "fluorophore2", ...}}}}
"""

        # Add retrieval context based on condition
        if condition.retrieval_mode != RetrievalMode.NONE:
            weight_note = ""
            if condition.retrieval_weight > 1:
                weight_note = f" (HIGHLY VALIDATED - weight {condition.retrieval_weight}x)"

            retrieval_context = f"""
Reference panels from published literature{weight_note}:

{OMIP_CORPUS}

Use these validated panels to inform your fluorophore selection. These panels have been optimized by flow cytometry experts.
"""
            base_prompt = retrieval_context + "\n" + base_prompt

        return base_prompt

    def _parse_assignments(self, response: str) -> dict[str, str]:
        """Extract marker-fluorophore assignments from response."""
        json_patterns = [
            r'\{[^{}]*"assignments"\s*:\s*\{[^{}]*\}[^{}]*\}',
            r'"assignments"\s*:\s*(\{[^{}]*\})',
            r'\{[^{}]+\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    text = match.group()
                    if "assignments" not in text and match.lastindex:
                        text = match.group(1)

                    data = json.loads(text)

                    if isinstance(data, dict):
                        if "assignments" in data:
                            return data["assignments"]
                        if all(isinstance(v, str) for v in data.values()):
                            return data
                except json.JSONDecodeError:
                    continue

        # Fallback: try to extract marker: fluorophore patterns
        assignments = {}
        lines = response.split('\n')
        for line in lines:
            match = re.search(r'["\']?(\w+)["\']?\s*[:->]+\s*["\']?([\w\s-]+)["\']?', line)
            if match:
                marker = match.group(1).strip()
                fluor = match.group(2).strip()
                if not any(x in fluor.lower() for x in ['high', 'medium', 'low', 'marker', 'expression']):
                    assignments[marker] = fluor

        return assignments

    def _calculate_complexity_index(self, assignments: dict[str, str]) -> float:
        """Calculate complexity index for assignments."""
        from flow_panel_optimizer.data.fluorophore_database import (
            get_fluorophore,
            calculate_spectral_overlap,
            get_known_overlap,
        )

        fluorophores = list(assignments.values())
        ci = 0.0

        for i, f1 in enumerate(fluorophores):
            for j, f2 in enumerate(fluorophores):
                if i >= j:
                    continue

                sim = get_known_overlap(f1, f2)
                if sim is None:
                    fluor1 = get_fluorophore(f1)
                    fluor2 = get_fluorophore(f2)
                    if fluor1 and fluor2:
                        sim = calculate_spectral_overlap(fluor1, fluor2)
                    else:
                        sim = 0.0

                if sim > 0.5:
                    ci += sim ** 2

        return round(ci, 2)

    def _score_accuracy(self, predicted: dict, ground_truth: dict) -> float:
        """Calculate assignment accuracy."""
        if not ground_truth:
            return 0.0

        correct = sum(
            1 for marker, fluor in ground_truth.items()
            if predicted.get(marker) == fluor
        )
        return correct / len(ground_truth)

    def _run_trial_litellm(
        self,
        test_case: PanelDesignTestCase,
        condition: ExperimentalCondition,
    ) -> TrialResult:
        """
        Run a trial using litellm.

        PLACEHOLDER: This method shows the intended interface but requires
        litellm to be installed for actual execution.
        """
        if not self._litellm_available:
            raise LiteLLMNotInstalledError(
                "litellm is not installed. Install with: pip install litellm"
            )

        import litellm

        prompt = self._build_prompt(test_case, condition)
        start_time = time.time()

        try:
            # LiteLLM unified call
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                api_base=self.config.api_base,
                api_key=self.api_key,
            )

            latency = time.time() - start_time
            raw_response = response.choices[0].message.content

            assignments = self._parse_assignments(raw_response)
            accuracy = self._score_accuracy(assignments, test_case.ground_truth_assignments)
            predicted_ci = self._calculate_complexity_index(assignments)
            ground_truth_ci = test_case.ground_truth_complexity_index

            if ground_truth_ci > 0:
                ci_improvement = (ground_truth_ci - predicted_ci) / ground_truth_ci
            else:
                ci_improvement = 0.0

            return TrialResult(
                condition_name=condition.name,
                test_case_id=test_case.id,
                test_case_type=test_case.case_type.value,
                raw_response=raw_response,
                extracted_assignments=assignments,
                tool_calls_made=[],  # No tool use in basic litellm mode
                assignment_accuracy=accuracy,
                complexity_index=predicted_ci,
                ground_truth_ci=ground_truth_ci,
                ci_improvement=ci_improvement,
                latency_seconds=latency,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            )

        except Exception as e:
            latency = time.time() - start_time
            return TrialResult(
                condition_name=condition.name,
                test_case_id=test_case.id,
                test_case_type=test_case.case_type.value,
                raw_response=f"ERROR: {str(e)}",
                extracted_assignments={},
                tool_calls_made=[],
                assignment_accuracy=0.0,
                complexity_index=float('inf'),
                ground_truth_ci=test_case.ground_truth_complexity_index,
                ci_improvement=0.0,
                latency_seconds=latency,
                input_tokens=0,
                output_tokens=0,
            )

    def _run_trial_fallback(
        self,
        test_case: PanelDesignTestCase,
        condition: ExperimentalCondition,
    ) -> TrialResult:
        """Run trial using fallback Anthropic client."""
        if not self._fallback_client:
            raise LiteLLMNotInstalledError(
                "Neither litellm nor anthropic SDK available for this model"
            )

        prompt = self._build_prompt(test_case, condition)
        start_time = time.time()

        try:
            response = self._fallback_client.messages.create(
                model=self.model.replace("anthropic/", ""),
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            latency = time.time() - start_time
            raw_response = response.content[0].text

            assignments = self._parse_assignments(raw_response)
            accuracy = self._score_accuracy(assignments, test_case.ground_truth_assignments)
            predicted_ci = self._calculate_complexity_index(assignments)
            ground_truth_ci = test_case.ground_truth_complexity_index

            if ground_truth_ci > 0:
                ci_improvement = (ground_truth_ci - predicted_ci) / ground_truth_ci
            else:
                ci_improvement = 0.0

            return TrialResult(
                condition_name=condition.name,
                test_case_id=test_case.id,
                test_case_type=test_case.case_type.value,
                raw_response=raw_response,
                extracted_assignments=assignments,
                tool_calls_made=[],
                assignment_accuracy=accuracy,
                complexity_index=predicted_ci,
                ground_truth_ci=ground_truth_ci,
                ci_improvement=ci_improvement,
                latency_seconds=latency,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        except Exception as e:
            latency = time.time() - start_time
            return TrialResult(
                condition_name=condition.name,
                test_case_id=test_case.id,
                test_case_type=test_case.case_type.value,
                raw_response=f"ERROR: {str(e)}",
                extracted_assignments={},
                tool_calls_made=[],
                assignment_accuracy=0.0,
                complexity_index=float('inf'),
                ground_truth_ci=test_case.ground_truth_complexity_index,
                ci_improvement=0.0,
                latency_seconds=latency,
                input_tokens=0,
                output_tokens=0,
            )

    def run_trial(
        self,
        test_case: PanelDesignTestCase,
        condition: ExperimentalCondition,
    ) -> TrialResult:
        """Run a single trial, selecting appropriate backend."""
        if self._litellm_available:
            return self._run_trial_litellm(test_case, condition)
        elif self._fallback_client:
            return self._run_trial_fallback(test_case, condition)
        else:
            raise LiteLLMNotInstalledError(
                f"Cannot run model {self.model}. Install litellm: pip install litellm"
            )

    def run_full_study(
        self,
        test_suite: TestSuite,
        conditions: Optional[list[ExperimentalCondition]] = None,
        verbose: bool = True
    ) -> ExperimentResults:
        """Run complete ablation study."""
        if conditions is None:
            conditions = CONDITIONS

        # Filter conditions based on model capabilities
        if not self.config.supports_tools:
            # Remove MCP conditions for models without tool support
            conditions = [c for c in conditions if not c.mcp_enabled]
            if verbose:
                print(f"Note: Model {self.model} does not support tools. "
                      f"Running {len(conditions)} non-MCP conditions only.")

        trials = []
        total = len(test_suite.test_cases) * len(conditions)

        for i, test_case in enumerate(test_suite.test_cases):
            for j, condition in enumerate(conditions):
                progress = (i * len(conditions) + j + 1) / total
                if verbose:
                    print(f"[{progress:.1%}] {condition.name} / {test_case.id}...", end=" ")

                trial = self.run_trial(test_case, condition)
                trials.append(trial)

                if verbose:
                    print(f"CI={trial.complexity_index:.2f}, "
                          f"Acc={trial.assignment_accuracy:.1%}, "
                          f"{trial.latency_seconds:.1f}s")

        return ExperimentResults(
            experiment_name=f"litellm_{self.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trials=trials
        )

    @staticmethod
    def list_supported_models() -> list[str]:
        """List all pre-configured model IDs."""
        return list(SUPPORTED_MODELS.keys())

    @staticmethod
    def check_ollama_available(model: str = "llama3.1:70b") -> bool:
        """
        Check if Ollama is running and the specified model is available.

        Returns True if Ollama is accessible and the model is pulled.
        """
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return any(model in name for name in model_names)
        except Exception:
            pass
        return False


# Convenience functions for common use cases

def run_local_llama_experiment(
    test_suite: TestSuite,
    model: str = "ollama/llama3.1:70b",
    verbose: bool = True
) -> ExperimentResults:
    """
    Run an experiment using a local Llama model via Ollama.

    Prerequisites:
    1. Install Ollama: curl https://ollama.ai/install.sh | sh
    2. Pull the model: ollama pull llama3.1:70b
    3. Start Ollama: ollama serve
    4. Install litellm: pip install litellm

    Args:
        test_suite: Test cases to evaluate
        model: Ollama model identifier (e.g., "ollama/llama3.1:70b")
        verbose: Print progress

    Returns:
        ExperimentResults with trial data
    """
    runner = LiteLLMAblationRunner(model=model)

    # Check if Ollama is running
    model_name = model.replace("ollama/", "")
    if not LiteLLMAblationRunner.check_ollama_available(model_name):
        raise LocalModelNotAvailableError(
            f"Ollama model {model_name} not available. "
            f"Ensure Ollama is running and the model is pulled:\n"
            f"  ollama serve\n"
            f"  ollama pull {model_name}"
        )

    return runner.run_full_study(test_suite, verbose=verbose)


def run_deepseek_experiment(
    test_suite: TestSuite,
    model: str = "ollama/deepseek-r1:32b",
    verbose: bool = True
) -> ExperimentResults:
    """
    Run an experiment using DeepSeek model via Ollama.

    Prerequisites:
    1. Install Ollama: curl https://ollama.ai/install.sh | sh
    2. Pull the model: ollama pull deepseek-r1:32b
    3. Start Ollama: ollama serve
    4. Install litellm: pip install litellm
    """
    return run_local_llama_experiment(test_suite, model=model, verbose=verbose)


if __name__ == "__main__":
    # Example usage
    print("Supported models:")
    for model in LiteLLMAblationRunner.list_supported_models():
        config = get_model_config(model)
        tools_status = "with tools" if config.supports_tools else "no tools"
        print(f"  {model} ({tools_status})")

    print("\nTo run experiments with local models:")
    print("1. Install litellm: pip install litellm")
    print("2. For Ollama models:")
    print("   - Install Ollama: curl https://ollama.ai/install.sh | sh")
    print("   - Pull model: ollama pull llama3.1:70b")
    print("   - Run: ollama serve")
    print("3. Then run:")
    print("   from flow_panel_optimizer.evaluation.litellm_runner import LiteLLMAblationRunner")
    print("   runner = LiteLLMAblationRunner(model='ollama/llama3.1:70b')")
    print("   results = runner.run_full_study(test_suite)")
