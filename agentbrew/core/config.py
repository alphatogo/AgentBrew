"""Dataclass config helpers used by migrated agents."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseConfig:
    """Small compatible base class for dataclass configs."""

    @classmethod
    def load(cls, data: dict[str, Any] | str | None):
        if data is None:
            return cls()
        if isinstance(data, dict):
            return cls.from_dict(data)
        return cls.from_json(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)

    @classmethod
    def from_json(cls, data: str):
        return cls.from_dict(json.loads(data))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

