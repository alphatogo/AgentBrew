"""Artifact writers for runs and tasks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TaskRunRecord(BaseModel):
    """Serializable summary for one task run."""

    task_id: str | None
    task_source: str | None = None
    domain: str
    mode: str
    status: str
    started_at: str
    ended_at: str | None = None
    trace_id: str | None = None
    result: Any = None
    evaluation_results: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)


class ArtifactStore:
    """Write run artifacts to a stable directory layout."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for_task(self, task_id: str | None) -> Path:
        safe_id = task_id or "task"
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe_id)
        return self.root / "tasks" / safe_id

    def write_json(self, path: str | Path, data: Any) -> Path:
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = self.root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, BaseModel):
            data = data.model_dump(mode="json")
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return output_path

    def append_jsonl(self, path: str | Path, data: Any) -> Path:
        output_path = Path(path)
        if not output_path.is_absolute():
            output_path = self.root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, BaseModel):
            data = data.model_dump(mode="json")
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
        return output_path

    def write_task_record(self, record: TaskRunRecord) -> Path:
        task_dir = self.path_for_task(record.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / "summary.json"
        output_path.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_path

    def write_run_manifest(self, data: dict[str, Any]) -> Path:
        payload = {"created_at": datetime.now().isoformat(), **data}
        return self.write_json("manifest.json", payload)
