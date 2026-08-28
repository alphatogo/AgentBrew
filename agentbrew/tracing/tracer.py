"""Minimal tracer compatible with migrated agents."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Tracer:
    """Collect internal spans and training-ready agent trajectory events."""

    def __init__(
        self,
        trace_id: str | None = None,
        *,
        include_prompts: bool = False,
    ) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        self.include_prompts = include_prompts
        self.records: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []

    @contextmanager
    def sprout(self):
        child = self.__class__(
            trace_id=self.trace_id,
            include_prompts=self.include_prompts,
        )
        child.add({"type": "span_start", "timestamp": time.time()})
        try:
            yield child
        finally:
            self.records.extend(child.records)
            self.trajectory.extend(child.trajectory)

    def add(self, data: dict[str, Any]) -> None:
        self.records.append({"timestamp": time.time(), "data": data})

    def record_llm(
        self,
        parsed_response: dict[str, Any],
        *,
        messages: list[dict[str, Any]] | None = None,
        response: Any = None,
    ) -> None:
        """Record one main-agent decision."""
        event = {
            "state": "llm",
            "timestamp": time.time(),
            "parsed_response": parsed_response,
        }
        if self.include_prompts:
            event["messages"] = messages or []
            event["response"] = response
        self.trajectory.append(event)

    def record_tool(
        self,
        *,
        server: str,
        tool_name: str,
        arguments: dict[str, Any],
        content: str,
        is_error: bool,
    ) -> None:
        """Record one tool result."""
        self.trajectory.append(
            {
                "state": "tool",
                "timestamp": time.time(),
                "server": server,
                "tool_name": tool_name,
                "arguments": arguments,
                "content": content,
                "isError": is_error,
            }
        )

    def write_trajectory(
        self,
        path: str | Path,
        *,
        task_id: str,
        question: str,
        final_answer: str,
    ) -> Path:
        """Write one task trajectory in the legacy-compatible format."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tasks": [
                {
                    "task_id": task_id,
                    "question": question,
                    "final_answer": final_answer,
                    "trajectory": self.trajectory,
                }
            ]
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path
