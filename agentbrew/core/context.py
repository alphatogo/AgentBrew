"""Runtime context for runs and environment lifecycles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Context(BaseModel):
    """Environment variables and metadata shared across a run."""

    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_env(self, name: str, default: str = "") -> str:
        """Return a variable from the context first, then from process env."""
        return self.env.get(name, os.environ.get(name, default))

    def set_env(self, name: str, value: str, *, export: bool = False) -> None:
        """Set a context env value, optionally mirroring it to process env."""
        self.env[name] = value
        if export:
            os.environ[name] = value

    def merged_env(self) -> dict[str, str]:
        """Return process env overlaid with context env."""
        merged = dict(os.environ)
        merged.update(self.env)
        return merged

    @classmethod
    def from_env_file(cls, path: str | Path | None = None) -> "Context":
        """Create context from a simple dotenv-style file."""
        values: dict[str, str] = {}
        if path is None:
            return cls(env=values)

        env_path = Path(path)
        if not env_path.exists():
            raise FileNotFoundError(f"Env file not found: {env_path}")

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            values[key.strip()] = value
        return cls(env=values)

