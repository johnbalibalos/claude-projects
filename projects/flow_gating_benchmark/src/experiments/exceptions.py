"""
Custom exception classes for the experiment runner.

Using specific exception types allows calling code to distinguish
between different failure modes:
- ConfigurationError: Setup/configuration issues (missing API keys, bad config)
- LLMClientError: Issues during LLM API calls
- ExperimentError: General experiment execution failures
"""

from __future__ import annotations


class ExperimentError(Exception):
    """Base exception for experiment-related errors."""

    pass


class ConfigurationError(ExperimentError):
    """
    Raised when there's a configuration or setup issue.

    Examples:
        - Missing API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
        - Missing required packages
        - Invalid configuration values
        - CLI tools not installed
    """

    pass


class LLMClientError(ExperimentError):
    """
    Raised when an LLM API call fails.

    Examples:
        - API rate limits exceeded
        - Invalid API responses
        - Timeout errors
        - Model not available
    """

    pass


class ParseError(ExperimentError):
    """
    Raised when parsing LLM output fails.

    Examples:
        - Invalid JSON in response
        - Missing required fields
        - Schema validation failures
    """

    pass
