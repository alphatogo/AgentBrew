"""Task specification and loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .errors import TaskLoadError


class MCPServerSpec(BaseModel):
    """MCP server requested by a task."""

    name: str
    transport: str = "stdio"
    config: dict[str, Any] = Field(default_factory=dict)


class EvaluationSpec(BaseModel):
    """Verifier requested by a task."""

    enabled: bool = True
    verifier: str | None = None
    func: str | None = None
    op: str | None = None
    value: Any = None
    op_args: Any = None
    desc: str = ""


class TaskSpec(BaseModel):
    """Domain-neutral task description."""

    id: str | None = None
    domain: str | None = None
    category: str = "general"
    question: str
    output_format: dict[str, Any] = Field(default_factory=dict)
    mcp_servers: list[MCPServerSpec] = Field(default_factory=list)
    evaluation: EvaluationSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any], *, source_path: str | None = None) -> "TaskSpec":
        """Load old MCP-Universe task JSON into the new task model."""
        evaluation = None
        evaluators = data.get("evaluators") or []
        if evaluators:
            first = evaluators[0]
            evaluation = EvaluationSpec(
                enabled=True,
                func=first.get("func"),
                op=first.get("op"),
                value=first.get("value"),
                op_args=first.get("op_args"),
                desc=first.get("desc", ""),
                verifier=first.get("op"),
            )

        return cls(
            id=data.get("id") or (Path(source_path).stem if source_path else None),
            domain=data.get("domain"),
            category=data.get("category", "general"),
            question=data["question"],
            output_format=data.get("output_format") or {},
            mcp_servers=[MCPServerSpec.model_validate(item) for item in data.get("mcp_servers", [])],
            evaluation=evaluation,
            metadata={
                "legacy": {
                    "prepares": data.get("prepares", []),
                    "cleanups": data.get("cleanups", []),
                    "evaluators": data.get("evaluators", []),
                    "use_specified_server": data.get("use_specified_server", False),
                }
            },
            source_path=source_path,
        )


class TaskLoader:
    """Load task specs from files, directories, manifests, or inline config."""

    @staticmethod
    def load_file(path: str | Path) -> TaskSpec:
        task_path = Path(path)
        try:
            raw = task_path.read_text(encoding="utf-8")
            if task_path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(raw)
            else:
                data = json.loads(raw)
        except Exception as exc:
            raise TaskLoadError(f"Failed to load task file {task_path}: {exc}") from exc

        if "question" not in data:
            raise TaskLoadError(f"Task file missing question: {task_path}")
        return TaskSpec.from_legacy_dict(data, source_path=str(task_path))

    @staticmethod
    def load_directory(path: str | Path, glob: str = "*.json") -> list[TaskSpec]:
        root = Path(path)
        if not root.exists():
            raise TaskLoadError(f"Task directory not found: {root}")
        files = sorted(p for p in root.rglob(glob) if p.is_file())
        return [TaskLoader.load_file(path) for path in files]

    @staticmethod
    def load_inline(tasks: list[dict[str, Any]]) -> list[TaskSpec]:
        return [TaskSpec.from_legacy_dict(item) for item in tasks]

