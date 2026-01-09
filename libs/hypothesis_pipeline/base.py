"""
Abstract base classes and protocols for the modular hypothesis pipeline.

Each component is designed to be independently swappable:
- PromptStrategy: How to structure reasoning (CoT, WoT, direct, etc.)
- RAGProvider: How to retrieve context (vector, hybrid, oracle, etc.)
- ContextBuilder: How to format context for the model
- ToolRegistry: Which tools/MCPs are available
- Evaluator: How to score outputs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from .models import (
    HypothesisCondition,
    TrialInput,
    ToolConfig,
    ReasoningType,
    ContextLevel,
    RAGMode,
)


# =============================================================================
# PROMPT STRATEGY
# =============================================================================


class PromptStrategy(ABC):
    """
    Abstract base for reasoning/prompting strategies.

    Strategies transform a base prompt into a structured prompt that
    encourages specific reasoning patterns (CoT, WoT, few-shot, etc.)
    """

    @property
    @abstractmethod
    def reasoning_type(self) -> ReasoningType:
        """The type of reasoning this strategy implements."""
        ...

    @abstractmethod
    def build_prompt(
        self,
        base_prompt: str,
        context: str,
        output_schema: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Build the complete prompt with reasoning structure.

        Args:
            base_prompt: The core task prompt
            context: Formatted context to include
            output_schema: Expected output format (JSON schema, etc.)
            examples: Few-shot examples if applicable

        Returns:
            Complete prompt with reasoning structure
        """
        ...

    @abstractmethod
    def extract_final_answer(self, response: str) -> str:
        """
        Extract the final answer from a response.

        For strategies like CoT, this extracts the answer after reasoning.
        For direct strategies, this may just return the response as-is.

        Args:
            response: Raw model response

        Returns:
            Extracted final answer
        """
        ...


# =============================================================================
# RAG PROVIDER
# =============================================================================


@runtime_checkable
class RAGProvider(Protocol):
    """
    Protocol for RAG (Retrieval Augmented Generation) providers.

    Providers retrieve relevant documents/context based on a query.
    """

    @property
    def rag_mode(self) -> RAGMode:
        """The type of RAG this provider implements."""
        ...

    def retrieve(
        self,
        query: str,
        trial_input: TrialInput,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The retrieval query
            trial_input: The trial input (may contain metadata for retrieval)
            top_k: Number of documents to retrieve
            **kwargs: Provider-specific options

        Returns:
            List of retrieved document strings
        """
        ...


class NoRAGProvider:
    """No-op RAG provider that returns empty results."""

    @property
    def rag_mode(self) -> RAGMode:
        return RAGMode.NONE

    def retrieve(
        self,
        query: str,
        trial_input: TrialInput,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        return []


# =============================================================================
# CONTEXT BUILDER
# =============================================================================


class ContextBuilder(ABC):
    """
    Abstract base for building context from various sources.

    Context builders combine:
    - Base context (from the trial input)
    - RAG-retrieved documents
    - Tool descriptions
    - Any additional context
    """

    @property
    @abstractmethod
    def context_level(self) -> ContextLevel:
        """The level of context this builder produces."""
        ...

    @abstractmethod
    def build_context(
        self,
        trial_input: TrialInput,
        rag_documents: list[str] | None = None,
        tool_descriptions: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Build the complete context string.

        Args:
            trial_input: The trial input with raw data
            rag_documents: Retrieved documents from RAG
            tool_descriptions: Descriptions of available tools
            **kwargs: Builder-specific options

        Returns:
            Formatted context string
        """
        ...


# =============================================================================
# TOOL REGISTRY
# =============================================================================


class ToolRegistry:
    """
    Registry of available tools/MCPs.

    Allows dynamic registration and selection of tools.
    Tools can be enabled/disabled per condition.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolConfig] = {}
        self._executors: dict[str, callable] = {}

    def register(
        self,
        tool: ToolConfig,
        executor: callable | None = None,
    ) -> None:
        """
        Register a tool.

        Args:
            tool: Tool configuration
            executor: Optional function to execute the tool
        """
        self._tools[tool.name] = tool
        if executor:
            self._executors[tool.name] = executor

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)
        self._executors.pop(name, None)

    def get_tool(self, name: str) -> ToolConfig | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools(self, names: list[str] | None = None) -> list[ToolConfig]:
        """
        Get tools by names or all enabled tools.

        Args:
            names: Specific tool names, or None for all enabled

        Returns:
            List of tool configurations
        """
        if names:
            return [self._tools[n] for n in names if n in self._tools]
        return [t for t in self._tools.values() if t.enabled]

    def get_anthropic_tools(self, names: list[str] | None = None) -> list[dict]:
        """Get tools in Anthropic API format."""
        return [t.to_anthropic_tool() for t in self.get_tools(names)]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        executor = self._executors.get(name)
        if executor:
            return executor(arguments)
        return {"error": f"No executor registered for tool: {name}"}

    @property
    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


# =============================================================================
# EVALUATOR
# =============================================================================


class Evaluator(ABC):
    """
    Abstract base for response evaluators.

    Evaluators extract structured output from responses and
    compute scoring metrics.
    """

    @abstractmethod
    def extract(self, response: str) -> Any:
        """
        Extract structured output from raw response.

        Args:
            response: Raw model response (after strategy extraction)

        Returns:
            Extracted structured output
        """
        ...

    @abstractmethod
    def score(
        self,
        extracted: Any,
        ground_truth: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute scoring metrics.

        Args:
            extracted: Extracted output from the response
            ground_truth: Expected output
            **kwargs: Additional scoring context

        Returns:
            Dictionary of metric name -> score
        """
        ...


# =============================================================================
# MODEL CLIENT
# =============================================================================


@runtime_checkable
class ModelClient(Protocol):
    """Protocol for LLM clients."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, list[dict], int, int]:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            tools: Optional tools in API format
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, tool_calls, input_tokens, output_tokens)
        """
        ...
