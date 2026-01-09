"""
Context builder implementations.

Context builders combine various sources of information into
a formatted context string for the model.

Supports composable context through layering and filtering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .base import ContextBuilder
from .models import ContextLevel, TrialInput


class BaseContextBuilder(ContextBuilder):
    """
    Base context builder with common functionality.

    Provides a template method pattern for context building.
    """

    def __init__(
        self,
        include_rag: bool = True,
        include_tools: bool = True,
        rag_header: str = "## Retrieved Context",
        tools_header: str = "## Available Tools",
        max_rag_docs: int = 5,
        max_rag_chars: int = 10000,
    ):
        self.include_rag = include_rag
        self.include_tools = include_tools
        self.rag_header = rag_header
        self.tools_header = tools_header
        self.max_rag_docs = max_rag_docs
        self.max_rag_chars = max_rag_chars

    @property
    @abstractmethod
    def context_level(self) -> ContextLevel:
        ...

    @abstractmethod
    def format_trial_context(self, trial_input: TrialInput, **kwargs: Any) -> str:
        """Format the base context from trial input."""
        ...

    def build_context(
        self,
        trial_input: TrialInput,
        rag_documents: list[str] | None = None,
        tool_descriptions: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Build complete context by combining all sources."""
        parts = []

        # 1. Add trial-specific context
        trial_context = self.format_trial_context(trial_input, **kwargs)
        if trial_context:
            parts.append(trial_context)

        # 2. Add RAG documents
        if self.include_rag and rag_documents:
            rag_context = self._format_rag_context(rag_documents)
            if rag_context:
                parts.append("")
                parts.append(self.rag_header)
                parts.append("")
                parts.append(rag_context)

        # 3. Add tool descriptions
        if self.include_tools and tool_descriptions:
            parts.append("")
            parts.append(self.tools_header)
            parts.append("")
            for desc in tool_descriptions:
                parts.append(f"- {desc}")

        return "\n".join(parts)

    def _format_rag_context(self, documents: list[str]) -> str:
        """Format RAG documents with truncation."""
        docs = documents[:self.max_rag_docs]
        formatted = []
        total_chars = 0

        for i, doc in enumerate(docs, 1):
            remaining = self.max_rag_chars - total_chars
            if remaining <= 0:
                break

            if len(doc) > remaining:
                doc = doc[:remaining] + "..."

            formatted.append(f"[Document {i}]\n{doc}")
            total_chars += len(doc)

        return "\n\n".join(formatted)


class MinimalContextBuilder(BaseContextBuilder):
    """
    Minimal context - bare essentials only.

    Typically just the core input data without metadata.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(include_rag=False, include_tools=False, **kwargs)

    @property
    def context_level(self) -> ContextLevel:
        return ContextLevel.MINIMAL

    def format_trial_context(self, trial_input: TrialInput, **kwargs: Any) -> str:
        # Just return the raw prompt as context
        return trial_input.prompt


class StandardContextBuilder(BaseContextBuilder):
    """
    Standard context - includes key metadata.

    Adds relevant metadata from trial input alongside core data.
    """

    def __init__(
        self,
        metadata_fields: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.metadata_fields = metadata_fields or []

    @property
    def context_level(self) -> ContextLevel:
        return ContextLevel.STANDARD

    def format_trial_context(self, trial_input: TrialInput, **kwargs: Any) -> str:
        parts = [trial_input.prompt]

        if self.metadata_fields and trial_input.metadata:
            meta_parts = []
            for field in self.metadata_fields:
                if field in trial_input.metadata:
                    value = trial_input.metadata[field]
                    meta_parts.append(f"{field}: {value}")

            if meta_parts:
                parts.append("")
                parts.append("## Additional Information")
                parts.extend(meta_parts)

        return "\n".join(parts)


class RichContextBuilder(BaseContextBuilder):
    """
    Rich context - includes all available information.

    Adds comprehensive metadata, notes, and context.
    """

    def __init__(
        self,
        format_fn: Callable[[TrialInput], str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.format_fn = format_fn

    @property
    def context_level(self) -> ContextLevel:
        return ContextLevel.RICH

    def format_trial_context(self, trial_input: TrialInput, **kwargs: Any) -> str:
        if self.format_fn:
            return self.format_fn(trial_input)

        parts = [trial_input.prompt]

        # Include all metadata
        if trial_input.metadata:
            parts.append("")
            parts.append("## Context")
            for key, value in trial_input.metadata.items():
                if not key.startswith("_"):  # Skip internal fields
                    parts.append(f"- {key}: {value}")

        return "\n".join(parts)


class OracleContextBuilder(BaseContextBuilder):
    """
    Oracle context - perfect context for upper-bound testing.

    Provides exactly the context needed to answer correctly.
    Useful for measuring extraction capability vs retrieval.
    """

    def __init__(
        self,
        oracle_field: str = "oracle_context",
        **kwargs: Any,
    ):
        super().__init__(include_rag=False, **kwargs)
        self.oracle_field = oracle_field

    @property
    def context_level(self) -> ContextLevel:
        return ContextLevel.ORACLE

    def format_trial_context(self, trial_input: TrialInput, **kwargs: Any) -> str:
        parts = [trial_input.prompt]

        oracle = trial_input.metadata.get(self.oracle_field)
        if oracle:
            parts.append("")
            parts.append("## Reference Information")
            parts.append("")
            if isinstance(oracle, list):
                parts.extend(oracle)
            else:
                parts.append(str(oracle))

        return "\n".join(parts)


class ComposableContextBuilder(ContextBuilder):
    """
    Composable context builder that chains multiple builders.

    Allows layering context from multiple sources.
    """

    def __init__(
        self,
        builders: list[ContextBuilder],
        separator: str = "\n\n---\n\n",
    ):
        self.builders = builders
        self.separator = separator

    @property
    def context_level(self) -> ContextLevel:
        # Return the highest level from component builders
        levels = [b.context_level for b in self.builders]
        level_order = [
            ContextLevel.NONE,
            ContextLevel.MINIMAL,
            ContextLevel.STANDARD,
            ContextLevel.RICH,
            ContextLevel.ORACLE,
        ]
        max_idx = max(level_order.index(l) for l in levels)
        return level_order[max_idx]

    def build_context(
        self,
        trial_input: TrialInput,
        rag_documents: list[str] | None = None,
        tool_descriptions: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        contexts = []

        for builder in self.builders:
            ctx = builder.build_context(
                trial_input,
                rag_documents=rag_documents,
                tool_descriptions=tool_descriptions,
                **kwargs,
            )
            if ctx:
                contexts.append(ctx)

        return self.separator.join(contexts)


class CallbackContextBuilder(ContextBuilder):
    """
    Flexible context builder using a callback function.

    Allows any custom context building logic without subclassing.
    """

    def __init__(
        self,
        build_fn: Callable[[TrialInput, list[str] | None, list[str] | None], str],
        level: ContextLevel = ContextLevel.STANDARD,
    ):
        self.build_fn = build_fn
        self._level = level

    @property
    def context_level(self) -> ContextLevel:
        return self._level

    def build_context(
        self,
        trial_input: TrialInput,
        rag_documents: list[str] | None = None,
        tool_descriptions: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        return self.build_fn(trial_input, rag_documents, tool_descriptions)


# =============================================================================
# CONTEXT BUILDER REGISTRY
# =============================================================================


def get_context_builder(level: ContextLevel, **kwargs: Any) -> ContextBuilder:
    """
    Factory function to get a context builder by level.

    Args:
        level: The context level
        **kwargs: Builder-specific configuration

    Returns:
        Configured builder instance
    """
    builders = {
        ContextLevel.NONE: lambda **kw: MinimalContextBuilder(include_rag=False, include_tools=False),
        ContextLevel.MINIMAL: MinimalContextBuilder,
        ContextLevel.STANDARD: StandardContextBuilder,
        ContextLevel.RICH: RichContextBuilder,
        ContextLevel.ORACLE: OracleContextBuilder,
    }

    builder_class = builders.get(level)
    if not builder_class:
        raise ValueError(f"Unknown context level: {level}")

    return builder_class(**kwargs)
