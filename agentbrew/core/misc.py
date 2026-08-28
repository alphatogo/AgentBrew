"""Compatibility helpers for migrated component classes."""

from __future__ import annotations

import inspect
from abc import ABCMeta
from collections import defaultdict
from typing import Any

_COMPONENTS: dict[str, list[type]] = defaultdict(list)


class AutodocABCMeta(ABCMeta):
    """Metaclass that inherits docstrings from base classes."""

    def __new__(mcs, classname: str, bases: tuple[type, ...], cls_dict: dict[str, Any]):
        cls = super().__new__(mcs, classname, bases, cls_dict)
        for name, member in cls_dict.items():
            if getattr(member, "__doc__", None) is None:
                for base in bases[::-1]:
                    attr = getattr(base, name, None)
                    if attr is not None:
                        member.__doc__ = attr.__doc__
                        break
        return cls


class ComponentABCMeta(AutodocABCMeta):
    """Register non-abstract classes by top-level package section."""

    def __new__(mcs, classname: str, bases: tuple[type, ...], cls_dict: dict[str, Any]):
        cls = super().__new__(mcs, classname, bases, cls_dict)
        if not inspect.isabstract(cls):
            parts = cls.__module__.split(".")
            module = parts[1] if len(parts) > 1 else parts[0]
            if cls not in _COMPONENTS[module]:
                _COMPONENTS[module].append(cls)
        return cls

    @staticmethod
    def get_class(module_name: str) -> list[type]:
        return _COMPONENTS[module_name]


class ExportConfigMixin:
    """Mixin for exporting class config."""

    def export_config(self) -> dict[str, Any]:
        config = {"name": self.__class__.__name__}
        if hasattr(self, "config"):
            config["config"] = self.config.to_dict()
        else:
            config["config"] = None
        return config


class BaseBuilder:
    """Small compatibility builder for migrated component managers."""

    @staticmethod
    def _name_to_class(classes: list[type]) -> dict[str, type]:
        mapping: dict[str, type] = {}
        for cls in classes:
            names = [cls.__name__]
            alias = getattr(cls, "alias", None)
            if isinstance(alias, str):
                names.append(alias)
            elif isinstance(alias, (list, tuple, set)):
                names.extend(str(item) for item in alias)
            for name in names:
                mapping[name] = cls
        return mapping
