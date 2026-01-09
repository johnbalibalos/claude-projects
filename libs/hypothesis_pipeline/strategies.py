"""
Prompt strategy implementations.

Each strategy defines how to structure the prompt for a specific
reasoning approach (CoT, WoT, direct, few-shot, etc.)
"""

from __future__ import annotations

import re
from typing import Any

from .base import PromptStrategy
from .models import ReasoningType


class DirectStrategy(PromptStrategy):
    """
    Direct prompting without explicit reasoning steps.

    Simply combines context and task, expecting a direct answer.
    """

    @property
    def reasoning_type(self) -> ReasoningType:
        return ReasoningType.DIRECT

    def build_prompt(
        self,
        base_prompt: str,
        context: str,
        output_schema: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        parts = []

        if context:
            parts.append(context)
            parts.append("")

        parts.append(base_prompt)

        if output_schema:
            parts.append("")
            parts.append("Return your answer in this format:")
            parts.append(output_schema)

        return "\n".join(parts)

    def extract_final_answer(self, response: str) -> str:
        # For direct, the whole response is the answer
        return response.strip()


class ChainOfThoughtStrategy(PromptStrategy):
    """
    Chain-of-Thought prompting.

    Encourages step-by-step reasoning before the final answer.
    Uses "Let's think step by step" pattern and extracts answer after reasoning.
    """

    def __init__(
        self,
        reasoning_prompt: str | None = None,
        answer_prefix: str = "Final Answer:",
    ):
        """
        Args:
            reasoning_prompt: Custom prompt to encourage reasoning.
                            Defaults to structured step-by-step guidance.
            answer_prefix: Prefix that marks the final answer in response.
        """
        self.reasoning_prompt = reasoning_prompt
        self.answer_prefix = answer_prefix

    @property
    def reasoning_type(self) -> ReasoningType:
        return ReasoningType.COT

    def build_prompt(
        self,
        base_prompt: str,
        context: str,
        output_schema: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        parts = []

        if context:
            parts.append(context)
            parts.append("")

        parts.append(base_prompt)
        parts.append("")

        # Add reasoning instructions
        if self.reasoning_prompt:
            parts.append(self.reasoning_prompt)
        else:
            parts.append("Think through this step by step:")
            parts.append("")
            parts.append("1. First, analyze the key information provided")
            parts.append("2. Consider what approach would be most appropriate")
            parts.append("3. Work through the reasoning systematically")
            parts.append("4. Arrive at your conclusion")

        parts.append("")
        parts.append(f"After your reasoning, provide your {self.answer_prefix}")

        if output_schema:
            parts.append("")
            parts.append("Use this format for your final answer:")
            parts.append(output_schema)

        return "\n".join(parts)

    def extract_final_answer(self, response: str) -> str:
        # Look for the answer prefix
        if self.answer_prefix in response:
            # Get everything after the prefix
            idx = response.rfind(self.answer_prefix)
            answer = response[idx + len(self.answer_prefix):].strip()
            return answer

        # Fallback: look for JSON block at end
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```\s*$', response)
        if json_match:
            return json_match.group(1).strip()

        # Fallback: look for JSON object at end
        json_obj_match = re.search(r'(\{[\s\S]*\})\s*$', response)
        if json_obj_match:
            return json_obj_match.group(1).strip()

        # Last resort: return full response
        return response.strip()


class WebOfThoughtStrategy(PromptStrategy):
    """
    Web/Tree of Thought prompting.

    Encourages exploring multiple reasoning paths and evaluating them.
    Useful for complex problems with multiple valid approaches.
    """

    def __init__(
        self,
        n_paths: int = 3,
        evaluation_criteria: list[str] | None = None,
    ):
        """
        Args:
            n_paths: Number of reasoning paths to explore
            evaluation_criteria: Criteria for evaluating paths
        """
        self.n_paths = n_paths
        self.evaluation_criteria = evaluation_criteria or [
            "completeness",
            "consistency",
            "correctness",
        ]

    @property
    def reasoning_type(self) -> ReasoningType:
        return ReasoningType.WOT

    def build_prompt(
        self,
        base_prompt: str,
        context: str,
        output_schema: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        parts = []

        if context:
            parts.append(context)
            parts.append("")

        parts.append(base_prompt)
        parts.append("")

        # Multi-path reasoning structure
        parts.append(f"Explore {self.n_paths} different approaches to this problem:")
        parts.append("")

        for i in range(1, self.n_paths + 1):
            parts.append(f"## Approach {i}")
            parts.append(f"[Describe approach {i} and work through it]")
            parts.append("")

        parts.append("## Evaluation")
        parts.append("Compare the approaches based on:")
        for criterion in self.evaluation_criteria:
            parts.append(f"- {criterion}")
        parts.append("")

        parts.append("## Final Answer")
        parts.append("Based on your evaluation, provide the best answer:")

        if output_schema:
            parts.append("")
            parts.append(output_schema)

        return "\n".join(parts)

    def extract_final_answer(self, response: str) -> str:
        # Look for Final Answer section
        if "## Final Answer" in response:
            idx = response.rfind("## Final Answer")
            answer = response[idx + len("## Final Answer"):].strip()
            # Clean up any trailing sections
            if "##" in answer:
                answer = answer[:answer.find("##")].strip()
            return answer

        # Fallback to JSON extraction
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```\s*$', response)
        if json_match:
            return json_match.group(1).strip()

        return response.strip()


class FewShotStrategy(PromptStrategy):
    """
    Few-shot prompting with examples.

    Provides examples before the task to guide the model.
    """

    def __init__(
        self,
        example_format: str = "Example {n}:\nInput: {input}\nOutput: {output}\n",
        use_cot_in_examples: bool = False,
    ):
        """
        Args:
            example_format: Format string for examples
            use_cot_in_examples: Whether examples include reasoning traces
        """
        self.example_format = example_format
        self.use_cot_in_examples = use_cot_in_examples

    @property
    def reasoning_type(self) -> ReasoningType:
        return ReasoningType.FEW_SHOT

    def build_prompt(
        self,
        base_prompt: str,
        context: str,
        output_schema: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        parts = []

        if context:
            parts.append(context)
            parts.append("")

        # Add examples
        if examples:
            parts.append("Here are some examples:")
            parts.append("")
            for i, example in enumerate(examples, 1):
                example_text = self.example_format.format(
                    n=i,
                    input=example.get("input", ""),
                    output=example.get("output", ""),
                    reasoning=example.get("reasoning", ""),
                )
                parts.append(example_text)
            parts.append("")

        parts.append("Now, complete this task:")
        parts.append("")
        parts.append(base_prompt)

        if output_schema:
            parts.append("")
            parts.append("Use this format:")
            parts.append(output_schema)

        return "\n".join(parts)

    def extract_final_answer(self, response: str) -> str:
        return response.strip()


class SelfConsistencyStrategy(PromptStrategy):
    """
    Self-consistency prompting.

    Generates multiple responses and finds consensus.
    Note: This is typically used with temperature > 0 and multiple samples.
    The strategy itself just sets up the prompt; sampling is handled externally.
    """

    def __init__(self, prompt_variation: str = "solve this problem"):
        self.prompt_variation = prompt_variation

    @property
    def reasoning_type(self) -> ReasoningType:
        return ReasoningType.SELF_CONSISTENCY

    def build_prompt(
        self,
        base_prompt: str,
        context: str,
        output_schema: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        parts = []

        if context:
            parts.append(context)
            parts.append("")

        parts.append(base_prompt)
        parts.append("")
        parts.append(f"Let's {self.prompt_variation} step by step.")

        if output_schema:
            parts.append("")
            parts.append("Provide your final answer in this format:")
            parts.append(output_schema)

        return "\n".join(parts)

    def extract_final_answer(self, response: str) -> str:
        # Similar to CoT extraction
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```\s*$', response)
        if json_match:
            return json_match.group(1).strip()

        json_obj_match = re.search(r'(\{[\s\S]*\})\s*$', response)
        if json_obj_match:
            return json_obj_match.group(1).strip()

        return response.strip()


class ReActStrategy(PromptStrategy):
    """
    ReAct (Reasoning + Acting) prompting.

    Interleaves reasoning with tool use actions.
    Structured as: Thought -> Action -> Observation -> ... -> Answer
    """

    def __init__(
        self,
        available_actions: list[str] | None = None,
        max_steps: int = 10,
    ):
        """
        Args:
            available_actions: List of available action names
            max_steps: Maximum reasoning steps
        """
        self.available_actions = available_actions or []
        self.max_steps = max_steps

    @property
    def reasoning_type(self) -> ReasoningType:
        return ReasoningType.REACT

    def build_prompt(
        self,
        base_prompt: str,
        context: str,
        output_schema: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        parts = []

        if context:
            parts.append(context)
            parts.append("")

        parts.append(base_prompt)
        parts.append("")

        parts.append("Solve this using the following format:")
        parts.append("")
        parts.append("Thought: [Your reasoning about what to do next]")
        parts.append("Action: [The action to take]")
        parts.append("Observation: [The result of the action]")
        parts.append("... (repeat Thought/Action/Observation as needed)")
        parts.append("Thought: I now have enough information to answer")
        parts.append("Final Answer: [Your answer]")

        if self.available_actions:
            parts.append("")
            parts.append("Available actions:")
            for action in self.available_actions:
                parts.append(f"- {action}")

        if output_schema:
            parts.append("")
            parts.append("Final answer format:")
            parts.append(output_schema)

        return "\n".join(parts)

    def extract_final_answer(self, response: str) -> str:
        # Look for Final Answer
        if "Final Answer:" in response:
            idx = response.rfind("Final Answer:")
            answer = response[idx + len("Final Answer:"):].strip()
            return answer

        return response.strip()


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================


def get_strategy(reasoning_type: ReasoningType, **config: Any) -> PromptStrategy:
    """
    Factory function to get a strategy by type.

    Args:
        reasoning_type: The type of reasoning strategy
        **config: Strategy-specific configuration

    Returns:
        Configured strategy instance
    """
    strategies = {
        ReasoningType.DIRECT: DirectStrategy,
        ReasoningType.COT: ChainOfThoughtStrategy,
        ReasoningType.WOT: WebOfThoughtStrategy,
        ReasoningType.FEW_SHOT: FewShotStrategy,
        ReasoningType.SELF_CONSISTENCY: SelfConsistencyStrategy,
        ReasoningType.REACT: ReActStrategy,
    }

    strategy_class = strategies.get(reasoning_type)
    if not strategy_class:
        raise ValueError(f"Unknown reasoning type: {reasoning_type}")

    return strategy_class(**config)


# For convenience, export pre-configured strategies
DIRECT = DirectStrategy()
COT = ChainOfThoughtStrategy()
WOT = WebOfThoughtStrategy()
