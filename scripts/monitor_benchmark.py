#!/usr/bin/env python3
"""Watch an AgentBrew benchmark run and emit compact anomaly/progress events."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ANOMALY_RE = re.compile(
    r"Repeated tool call|Failed to initialize|Connection closed|Traceback|"
    r"object_not_found|validation_error|rate.?limit|Notion setup failed|"
    r"Search index failed|status[\"']?\s*[:=]\s*(?:4\d\d|5\d\d)",
    re.IGNORECASE,
)


def emit(event: str, **payload: Any) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def summarize(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    evaluations = data.get("evaluation_results") or []
    return {
        "task_id": data.get("task_id") or path.parent.name,
        "execution_status": data.get("status"),
        "evaluation_passed": bool(evaluations)
        and all(bool(item.get("passed")) for item in evaluations),
        "evaluations": [
            {
                "verifier": item.get("verifier"),
                "passed": bool(item.get("passed")),
                "reason": item.get("reason", ""),
                "error": item.get("error", ""),
            }
            for item in evaluations
        ],
        "agent_result": str(data.get("result") or "")[:500],
        "error": str(data.get("error") or "")[:500],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--expected-tasks", type=int, default=0)
    args = parser.parse_args()

    root = args.output_root.resolve()
    run_log = root / "run.log"
    task_root = root / "tasks"
    seen_summaries: set[Path] = set()
    log_offset = 0
    last_progress: tuple[int, int] | None = None

    emit(
        "monitor_started",
        output_root=str(root),
        pid=args.pid,
        expected_tasks=args.expected_tasks,
    )

    while True:
        if run_log.exists():
            size = run_log.stat().st_size
            if size < log_offset:
                log_offset = 0
            with run_log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(log_offset)
                for raw_line in handle:
                    line = ANSI_RE.sub("", raw_line).strip()
                    if line and ANOMALY_RE.search(line):
                        emit("anomaly", message=line[:2000])
                log_offset = handle.tell()

        summaries = sorted(task_root.glob("*/summary.json")) if task_root.exists() else []
        for summary_path in summaries:
            if summary_path in seen_summaries:
                continue
            try:
                emit("task_completed", **summarize(summary_path))
                seen_summaries.add(summary_path)
            except (OSError, ValueError, TypeError) as exc:
                emit(
                    "summary_read_error",
                    path=str(summary_path),
                    error=str(exc),
                )

        task_count = len(summaries)
        passed_count = 0
        for summary_path in summaries:
            try:
                evaluations = json.loads(
                    summary_path.read_text(encoding="utf-8")
                ).get("evaluation_results") or []
                if evaluations and all(item.get("passed") for item in evaluations):
                    passed_count += 1
            except (OSError, ValueError, TypeError):
                pass

        progress = (task_count, passed_count)
        if progress != last_progress:
            emit(
                "progress",
                completed=task_count,
                passed=passed_count,
                failed=task_count - passed_count,
                expected=args.expected_tasks,
            )
            last_progress = progress

        alive = process_alive(args.pid)
        complete = bool(args.expected_tasks) and task_count >= args.expected_tasks
        if complete or not alive:
            emit(
                "monitor_finished",
                process_alive=alive,
                completed=task_count,
                passed=passed_count,
                failed=task_count - passed_count,
                expected=args.expected_tasks,
            )
            return 0 if complete else 1

        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
