"""GitHub environment."""

from agentbrew.core.registry import environment_registry

from .environment import GitHubEnvironment

environment_registry.register("github", GitHubEnvironment, replace=True)

__all__ = ["GitHubEnvironment"]
