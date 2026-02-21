"""
Unified exception hierarchy for the BioLLM research platform.

Provides a consistent set of exceptions across all libs/ and projects/.
Project-specific exceptions should inherit from these base classes.
"""


class BioLLMException(Exception):
    """Base exception for all BioLLM platform errors."""


class ConfigurationError(BioLLMException):
    """Raised when there's a configuration or setup issue.

    Examples: missing API keys, invalid config files, bad parameters.
    """


class APIError(BioLLMException):
    """Raised when an external API call fails.

    Examples: LLM provider errors, HTTP failures, rate limits.
    """


class ValidationError(BioLLMException):
    """Raised when data validation fails.

    Examples: invalid input format, schema mismatch, constraint violations.
    """


class BioLLMTimeoutError(BioLLMException):
    """Raised when an operation times out.

    Named BioLLMTimeoutError to avoid shadowing the builtin TimeoutError.
    """


class ParseError(BioLLMException):
    """Raised when parsing LLM output or data fails.

    Examples: invalid JSON from LLM, malformed XML, unexpected response format.
    """
