"""
RAG (Retrieval Augmented Generation) provider implementations.

Providers retrieve relevant context based on queries.
Multiple backends supported: vector stores, hybrid search, oracle (perfect), etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Protocol

from .models import RAGMode, TrialInput


class RAGProvider(ABC):
    """
    Abstract base class for RAG providers.

    Providers retrieve documents relevant to a query.
    """

    @property
    @abstractmethod
    def rag_mode(self) -> RAGMode:
        """The type of RAG this provider implements."""
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        trial_input: TrialInput,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        """
        Retrieve relevant documents.

        Args:
            query: The retrieval query
            trial_input: Context for retrieval (may contain metadata)
            top_k: Number of documents to retrieve
            **kwargs: Provider-specific options

        Returns:
            List of document strings
        """
        ...

    def get_context_for_trial(
        self,
        trial_input: TrialInput,
        top_k: int = 5,
    ) -> list[str]:
        """
        Convenience method: retrieve based on trial's prompt.

        Args:
            trial_input: The trial input
            top_k: Number of documents

        Returns:
            List of retrieved documents
        """
        return self.retrieve(trial_input.prompt, trial_input, top_k)


class NoRAGProvider(RAGProvider):
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


class OracleRAGProvider(RAGProvider):
    """
    Oracle RAG provider for upper-bound testing.

    Returns "perfect" retrieval results, typically the exact context
    needed to answer the question. Useful for testing extraction capability
    separately from retrieval capability.
    """

    def __init__(
        self,
        oracle_fn: Callable[[TrialInput], list[str]] | None = None,
        oracle_field: str = "oracle_context",
    ):
        """
        Args:
            oracle_fn: Function to get oracle context from trial input
            oracle_field: Field in trial_input.metadata containing oracle context
        """
        self.oracle_fn = oracle_fn
        self.oracle_field = oracle_field

    @property
    def rag_mode(self) -> RAGMode:
        return RAGMode.ORACLE

    def retrieve(
        self,
        query: str,
        trial_input: TrialInput,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        # Use custom function if provided
        if self.oracle_fn:
            return self.oracle_fn(trial_input)

        # Otherwise look in metadata
        oracle = trial_input.metadata.get(self.oracle_field, [])
        if isinstance(oracle, str):
            return [oracle]
        return oracle if oracle else []


class NegativeRAGProvider(RAGProvider):
    """
    Negative RAG provider for lower-bound/control testing.

    Returns irrelevant or wrong documents to test model's
    robustness to noise and ability to recognize bad context.
    """

    def __init__(
        self,
        negative_docs: list[str] | None = None,
        negative_fn: Callable[[TrialInput], list[str]] | None = None,
    ):
        """
        Args:
            negative_docs: Static list of negative documents
            negative_fn: Function to generate negative docs for a trial
        """
        self.negative_docs = negative_docs or []
        self.negative_fn = negative_fn

    @property
    def rag_mode(self) -> RAGMode:
        return RAGMode.NEGATIVE

    def retrieve(
        self,
        query: str,
        trial_input: TrialInput,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        if self.negative_fn:
            return self.negative_fn(trial_input)[:top_k]
        return self.negative_docs[:top_k]


class VectorRAGProvider(RAGProvider):
    """
    Vector-based RAG provider using embeddings.

    Supports various vector store backends through a unified interface.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        search_fn: Callable[[list[float], int], list[tuple[str, float]]],
        score_threshold: float = 0.0,
    ):
        """
        Args:
            embed_fn: Function to embed a query string
            search_fn: Function to search by embedding, returns (doc, score) pairs
            score_threshold: Minimum similarity score to include
        """
        self.embed_fn = embed_fn
        self.search_fn = search_fn
        self.score_threshold = score_threshold

    @property
    def rag_mode(self) -> RAGMode:
        return RAGMode.VECTOR

    def retrieve(
        self,
        query: str,
        trial_input: TrialInput,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        embedding = self.embed_fn(query)
        results = self.search_fn(embedding, top_k)

        # Filter by threshold and extract documents
        return [
            doc for doc, score in results
            if score >= self.score_threshold
        ]


class HybridRAGProvider(RAGProvider):
    """
    Hybrid RAG combining vector and keyword search.

    Merges results from multiple retrieval strategies.
    """

    def __init__(
        self,
        vector_provider: VectorRAGProvider,
        keyword_fn: Callable[[str, int], list[str]],
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """
        Args:
            vector_provider: Vector search provider
            keyword_fn: Keyword search function
            vector_weight: Weight for vector results
            keyword_weight: Weight for keyword results
        """
        self.vector_provider = vector_provider
        self.keyword_fn = keyword_fn
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    @property
    def rag_mode(self) -> RAGMode:
        return RAGMode.HYBRID

    def retrieve(
        self,
        query: str,
        trial_input: TrialInput,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        # Get results from both methods
        vector_k = int(top_k * (1 + self.vector_weight))
        keyword_k = int(top_k * (1 + self.keyword_weight))

        vector_results = self.vector_provider.retrieve(query, trial_input, vector_k)
        keyword_results = self.keyword_fn(query, keyword_k)

        # Combine with deduplication (preserving order)
        seen = set()
        combined = []

        # Interleave based on weights
        v_idx, k_idx = 0, 0
        v_step = 1 / (self.vector_weight + 0.001)
        k_step = 1 / (self.keyword_weight + 0.001)
        v_acc, k_acc = 0.0, 0.0

        while len(combined) < top_k and (v_idx < len(vector_results) or k_idx < len(keyword_results)):
            v_acc += 1
            k_acc += 1

            # Add from vector if due
            if v_acc >= v_step and v_idx < len(vector_results):
                doc = vector_results[v_idx]
                if doc not in seen:
                    combined.append(doc)
                    seen.add(doc)
                v_idx += 1
                v_acc = 0

            # Add from keyword if due
            if k_acc >= k_step and k_idx < len(keyword_results):
                doc = keyword_results[k_idx]
                if doc not in seen:
                    combined.append(doc)
                    seen.add(doc)
                k_idx += 1
                k_acc = 0

        return combined[:top_k]


class CallbackRAGProvider(RAGProvider):
    """
    Flexible RAG provider using a callback function.

    Allows any custom retrieval logic without subclassing.
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, TrialInput, int], list[str]],
        mode: RAGMode = RAGMode.VECTOR,
    ):
        """
        Args:
            retrieve_fn: Function(query, trial_input, top_k) -> list[str]
            mode: The RAG mode to report
        """
        self.retrieve_fn = retrieve_fn
        self._mode = mode

    @property
    def rag_mode(self) -> RAGMode:
        return self._mode

    def retrieve(
        self,
        query: str,
        trial_input: TrialInput,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[str]:
        return self.retrieve_fn(query, trial_input, top_k)


# =============================================================================
# RAG PROVIDER REGISTRY
# =============================================================================


class RAGRegistry:
    """Registry of RAG providers for easy switching."""

    def __init__(self) -> None:
        self._providers: dict[str, RAGProvider] = {
            "none": NoRAGProvider(),
        }

    def register(self, name: str, provider: RAGProvider) -> None:
        """Register a provider."""
        self._providers[name] = provider

    def get(self, name: str) -> RAGProvider:
        """Get a provider by name."""
        provider = self._providers.get(name)
        if not provider:
            raise ValueError(f"Unknown RAG provider: {name}. Available: {list(self._providers.keys())}")
        return provider

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())


# Global registry instance
_rag_registry = RAGRegistry()


def register_rag_provider(name: str, provider: RAGProvider) -> None:
    """Register a RAG provider globally."""
    _rag_registry.register(name, provider)


def get_rag_provider(name: str) -> RAGProvider:
    """Get a RAG provider by name."""
    return _rag_registry.get(name)


def get_provider_for_mode(mode: RAGMode, **kwargs: Any) -> RAGProvider:
    """
    Factory function to get a provider by mode.

    Args:
        mode: The RAG mode
        **kwargs: Provider-specific configuration

    Returns:
        Configured provider instance
    """
    if mode == RAGMode.NONE:
        return NoRAGProvider()
    elif mode == RAGMode.ORACLE:
        return OracleRAGProvider(**kwargs)
    elif mode == RAGMode.NEGATIVE:
        return NegativeRAGProvider(**kwargs)
    elif mode == RAGMode.VECTOR:
        raise ValueError("VectorRAGProvider requires embed_fn and search_fn")
    elif mode == RAGMode.HYBRID:
        raise ValueError("HybridRAGProvider requires vector_provider and keyword_fn")
    else:
        raise ValueError(f"Unknown RAG mode: {mode}")
