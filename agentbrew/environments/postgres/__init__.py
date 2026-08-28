"""PostgreSQL environment support."""

from agentbrew.core.registry import environment_registry

from .environment import PostgresEnvironment

environment_registry.register("postgres", PostgresEnvironment, replace=True)

__all__ = ["PostgresEnvironment"]
