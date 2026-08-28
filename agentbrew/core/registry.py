"""Runtime registries."""

from __future__ import annotations

from collections.abc import Callable

from .environment import Environment
from .errors import RegistryError

EnvironmentFactory = Callable[[], Environment]


class EnvironmentRegistry:
    """Register and build domain environments."""

    def __init__(self) -> None:
        self._factories: dict[str, EnvironmentFactory] = {}

    def register(self, name: str, factory: EnvironmentFactory, *, replace: bool = False) -> None:
        if name in self._factories and not replace:
            raise RegistryError(f"Environment already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str) -> Environment:
        if name not in self._factories:
            raise RegistryError(f"Unknown environment: {name}")
        return self._factories[name]()

    def names(self) -> list[str]:
        return sorted(self._factories)


environment_registry = EnvironmentRegistry()

