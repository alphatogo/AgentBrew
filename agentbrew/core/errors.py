"""Core exceptions."""


class AgentBrewError(Exception):
    """Base exception for AgentBrew."""


class ConfigError(AgentBrewError):
    """Raised when a run configuration is invalid."""


class RegistryError(AgentBrewError):
    """Raised when a requested registry item cannot be found."""


class TaskLoadError(AgentBrewError):
    """Raised when task specs cannot be loaded or parsed."""


class EnvironmentError(AgentBrewError):
    """Raised when an environment lifecycle step fails."""


class AgentExecutionError(AgentBrewError):
    """Raised when the agent execution loop fails."""

