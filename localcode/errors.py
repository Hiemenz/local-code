"""Exception hierarchy for localcode."""

from __future__ import annotations


class LocalCodeError(Exception):
    """Base class for all localcode errors."""


class ConfigError(LocalCodeError):
    """Invalid or missing configuration."""


class ProviderError(LocalCodeError):
    """An LLM provider failed."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolError(LocalCodeError):
    """A tool invocation failed."""


class PathSecurityError(ToolError):
    """A tool tried to access a path outside the workspace root."""


class ApprovalDenied(LocalCodeError):
    """The user denied a requested action."""


class Interrupted(LocalCodeError):
    """The user interrupted the current turn."""
