"""Notion environment."""

from agentbrew.core.registry import environment_registry

from .environment import NotionEnvironment

environment_registry.register("notion", NotionEnvironment, replace=True)

__all__ = ["NotionEnvironment"]
