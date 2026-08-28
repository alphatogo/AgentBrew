"""Utilities for structured PostgreSQL MCPMark verification."""

from __future__ import annotations

import contextlib
import importlib
import io
import re
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

import psycopg2  # type: ignore


@dataclass(frozen=True)
class VerificationStep:
    """One structured verification step."""

    name: str
    func_name: str
    args_factory: Callable[[object], tuple] | None = None


class SubCriteriaCollector:
    """Collect sub-check results in the same format used by MCPMark Notion."""

    def __init__(self, task_name: str = ""):
        self.task_name = task_name
        self._items: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, error_msg: str = "") -> bool:
        ok = bool(condition)
        self._items.append((name, ok, "" if ok else error_msg))
        return ok

    def summary(self) -> tuple[bool, str]:
        total = len(self._items)
        passed = sum(1 for _, ok, _ in self._items if ok)
        all_passed = passed == total

        print(f"\n{'-' * 62}", file=sys.stderr, flush=True)
        label = f"  Task: {self.task_name}" if self.task_name else "  Task result"
        status = "PASS" if all_passed else "FAIL"
        print(f"{label}  [{status}]", file=sys.stderr, flush=True)
        print(f"  Sub-criteria: {passed}/{total} passed", file=sys.stderr, flush=True)
        for name, ok, msg in self._items:
            mark = "✓" if ok else "✗"
            suffix = f"  ({msg})" if (not ok and msg) else ""
            print(f"    {mark} {name}{suffix}", file=sys.stderr, flush=True)
        print(f"{'-' * 62}", file=sys.stderr, flush=True)

        lines = [f"[SUBCRITERIA:{passed}/{total}]"]
        for name, ok, msg in self._items:
            if ok:
                lines.append(f"PASS | {name}")
            elif msg:
                lines.append(f"FAIL | {name} | {msg}")
            else:
                lines.append(f"FAIL | {name}")
        return all_passed, "\n".join(lines)


def _get_connection_params(module: object) -> dict:
    if not hasattr(module, "get_connection_params"):
        raise AttributeError(f"{module.__name__} does not define get_connection_params()")
    conn_params = module.get_connection_params()
    if not conn_params.get("database"):
        raise ValueError("POSTGRES_DATABASE environment variable not set")
    return conn_params


def run_step_based_verify(
    module_path: str,
    task_name: str,
    steps: Sequence[VerificationStep],
) -> tuple[bool, str]:
    """Run all declared verification steps and return a structured summary."""
    module = importlib.import_module(module_path)
    collector = SubCriteriaCollector(task_name)
    conn = None
    try:
        conn = psycopg2.connect(**_get_connection_params(module))
        if hasattr(conn, "autocommit"):
            conn.autocommit = False

        for step in steps:
            func = getattr(module, step.func_name)
            args = step.args_factory(module) if step.args_factory else ()
            try:
                result = func(conn, *args)
            finally:
                with contextlib.suppress(Exception):
                    conn.rollback()

            if not isinstance(result, tuple) or len(result) < 2:
                raise ValueError(
                    f"{module_path}.{step.func_name} must return at least (passed, message)"
                )
            passed, msg = bool(result[0]), str(result[1] or "")
            collector.check(step.name, passed, msg)

        return collector.summary()
    except Exception as exc:  # pragma: no cover - failure path is the point here
        collector.check("verification_runtime", False, str(exc))
        return collector.summary()
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


def _normalize_output_lines(output: str) -> list[str]:
    lines: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[|]\s*", "", line)
        lines.append(line)
    return lines


def _parse_legacy_output(output: str) -> list[tuple[str, bool, str]]:
    items: list[tuple[str, bool, str]] = []
    seen: set[tuple[str, bool, str]] = set()
    for line in _normalize_output_lines(output):
        status: bool | None = None
        payload = line
        if payload.startswith(("✓", "✅")):
            status = True
            payload = payload[1:].strip(" :.-")
        elif payload.startswith(("✗", "❌")):
            status = False
            payload = payload[1:].strip(" :.-")
        elif payload.startswith("PASS:"):
            status = True
            payload = payload[len("PASS:"):].strip()
        elif payload.startswith("FAIL:"):
            status = False
            payload = payload[len("FAIL:"):].strip()
        elif "Summary:" in payload or payload.startswith("Verifying"):
            continue

        if status is None or not payload:
            continue

        name = payload
        msg = ""
        if not status and ": " in payload:
            name, msg = payload.split(": ", 1)
        item = (name.strip(), status, msg.strip())
        if item not in seen:
            seen.add(item)
            items.append(item)
    return items


def run_legacy_verify(module_path: str, task_name: str) -> tuple[bool, str]:
    """Run an existing verify() and convert its printed checks into sub-criteria."""
    module = importlib.import_module(module_path)
    collector = SubCriteriaCollector(task_name)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            passed, reason = module.verify()
    except Exception as exc:  # pragma: no cover - failure path is the point here
        collector.check("verification_runtime", False, str(exc))
        return collector.summary()

    combined_output = stdout_buffer.getvalue()
    if stderr_buffer.getvalue():
        combined_output += "\n" + stderr_buffer.getvalue()

    parsed_items = _parse_legacy_output(combined_output)
    if parsed_items:
        for name, ok, msg in parsed_items:
            collector.check(name, ok, msg)
    else:
        collector.check("legacy_verify", bool(passed), str(reason or ""))

    if not passed and reason:
        # Preserve the top-level verifier result if parsing didn't capture the failure.
        if all(ok for _, ok, _ in collector._items):
            collector.check("legacy_verify", False, str(reason))

    return collector.summary()


def expected_customers_args(module: object) -> tuple:
    expected_customers = module.load_expected_customers()
    if not expected_customers:
        raise ValueError("Failed to load expected customer data")
    return (expected_customers,)
