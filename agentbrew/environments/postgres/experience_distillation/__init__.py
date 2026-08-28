"""PostgreSQL-specific trajectory rendering and retrospective-task prompt."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


__all__ = [
    "DEFAULT_CREDIT_METHOD",
    "build_credit_prompt_prefix",
    "build_task_inference_prompt",
    "format_trajectory",
    "is_eligible_task",
    "is_error",
    "normalize_task_inference_output",
    "summarize_changes",
]


DEFAULT_CREDIT_METHOD = "best_prefix"


def _structured(content: str) -> Any:
    try:
        return json.loads(content)
    except Exception:
        try:
            return ast.literal_eval(content)
        except Exception:
            return None


def is_error(step: dict[str, Any]) -> bool:
    content = str(step.get("content") or "")
    stripped = content.lstrip()
    if bool(step.get("isError", False)) or stripped.startswith(("Error:", "ERROR:")):
        return True
    parsed = _structured(content)
    if isinstance(parsed, dict):
        if parsed.get("error"):
            return True
        status = str(parsed.get("status", "")).lower()
        code = str(parsed.get("code", "")).lower()
        message = str(parsed.get("message", "")).lower()
        return (
            status in {"error", "failed", "failure"}
            or code in {"error", "failed", "failure"}
            or message.startswith("error:")
            or "exception" in message
        )
    return False


def is_eligible_task(task: dict[str, Any]) -> bool:
    return task.get("final_answer") is not None and bool(task["trajectory"])


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _clip(value: Any, limit: int = 180) -> str:
    text = _compact(value)
    return text if len(text) <= limit else text[:limit] + "...[TRUNCATED]"


def _sanitize(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value[:8]]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in list(value.items())[:20]}
    return value


def summarize_arguments(step: dict[str, Any]) -> str:
    tool = step.get("tool_name", "")
    args = step.get("arguments", {}) or {}
    if tool in {"execute_sql", "explain_query"}:
        summary = {"sql": _clip(args.get("sql", ""), 280)}
        if "analyze" in args:
            summary["analyze"] = args["analyze"]
    elif tool == "get_object_details":
        summary = {
            "schema_name": args.get("schema_name"),
            "object_name": args.get("object_name"),
            "object_type": args.get("object_type"),
        }
    elif tool == "list_objects":
        summary = {
            "schema_name": args.get("schema_name"),
            "object_type": args.get("object_type"),
        }
    else:
        summary = _sanitize(args)
    return json.dumps(summary, ensure_ascii=False)


def summarize_feedback(step: dict[str, Any]) -> str:
    content = str(step.get("content") or "")
    parsed = _structured(content)
    if parsed is None:
        return _clip(content, 360)
    if isinstance(parsed, list):
        preview = [_sanitize(item) for item in parsed[:2]]
        return json.dumps(
            {
                "type": "list",
                "count": len(parsed),
                "first_row_keys": (
                    list(preview[0].keys())[:10]
                    if preview and isinstance(preview[0], dict)
                    else []
                ),
                "preview": preview,
            },
            ensure_ascii=False,
        )
    if isinstance(parsed, dict):
        summary = {"type": "dict", "keys": list(parsed.keys())[:12]}
        for key in ("message", "status", "code", "schema", "name", "type"):
            if key in parsed:
                summary[key] = _sanitize(parsed[key])
        return json.dumps(summary, ensure_ascii=False)
    return _clip(parsed, 360)


def format_trajectory(trajectory: list[dict[str, Any]]) -> str:
    lines = []
    for index, step in enumerate(trajectory, start=1):
        lines.extend(
            [
                f"[Step {index}] Tool: {step.get('tool_name')}",
                f"Args Summary: {summarize_arguments(step)}",
                f"Feedback Summary (Error: {is_error(step)}): {summarize_feedback(step)}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _sql_effect(sql: str) -> str:
    normalized = _compact(sql)
    patterns = [
        (r"create\s+(?:or\s+replace\s+)?table\s+([^\s(]+)", "created table"),
        (r"create\s+(?:or\s+replace\s+)?materialized\s+view\s+([^\s(]+)", "created materialized view"),
        (r"create\s+(?:or\s+replace\s+)?view\s+([^\s(]+)", "created view"),
        (r"create\s+(?:unique\s+)?index\s+([^\s(]+)", "created index"),
        (r"create\s+(?:or\s+replace\s+)?function\s+([^\s(]+)", "created function"),
        (r"create\s+(?:constraint\s+)?trigger\s+([^\s(]+)", "created trigger"),
        (r"alter\s+table\s+([^\s(]+)", "altered table"),
        (r"insert\s+into\s+([^\s(]+)", "inserted into"),
        (r"update\s+([^\s(]+)", "updated"),
        (r"delete\s+from\s+([^\s(]+)", "deleted from"),
        (r"drop\s+table\s+([^\s(]+)", "dropped table"),
        (r"drop\s+view\s+([^\s(]+)", "dropped view"),
        (r"drop\s+index\s+([^\s(]+)", "dropped index"),
    ]
    for pattern, label in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return f"{label}: {match.group(1)}"
    if normalized.lower().startswith(("select ", "with ")):
        return f"ran query: {_clip(normalized)}"
    return _clip(normalized)


def summarize_changes(trajectory: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    inspected: list[str] = []
    for step in trajectory:
        if is_error(step):
            continue
        tool = step.get("tool_name", "")
        args = step.get("arguments", {}) or {}
        if tool == "execute_sql":
            lines.append(f"- {_sql_effect(str(args.get('sql', '')))}")
        elif tool == "explain_query":
            lines.append(f"- profiled query plan: {_clip(args.get('sql', ''))}")
        elif tool == "get_object_details" and args.get("object_name"):
            inspected.append(str(args["object_name"]))
        elif tool == "list_objects":
            lines.append(
                f"- inspected {args.get('object_type') or 'objects'} "
                f"in schema {args.get('schema_name') or '[unknown]'}"
            )
        elif tool == "list_schemas":
            lines.append("- inspected available schemas")
    if inspected:
        lines.append(f"- inspected object details for: {list(dict.fromkeys(inspected))[:12]}")
    return "\n".join(lines) if lines else "- No confirmed successful PostgreSQL work detected."


def build_task_inference_prompt(
    original_task: str,
    trajectory_str: str,
    change_summary: str,
) -> str:
    return f"""You revise benchmark-style PostgreSQL tasks from agent tool trajectories.

Your job is to minimally edit the ORIGINAL TASK so it matches what the agent actually completed.

Main goal:
- The revised task must be highly consistent with the trajectory.
- The revised task must stay close to the original task in tone, high-level intent, and overall task type.
- Keep the original task wording, structure, and framing as much as possible.
- Only keep concrete subtasks that are directly supported by the trajectory.
- Remove or weaken unsupported requirements.
- Do not rewrite the whole task from scratch.
- Focus on what the agent truly accomplished in the database.

Use the trajectory as evidence, with priority on:
1. successful database changes and successful analysis steps
2. successful inspections only when they help identify what was actually completed

Rules:
- Write the task in English.
- Begin exactly with: `Please use PostgreSQL tools to finish the following task:`
- Output only the revised task text.
- Preserve the original task wording whenever possible.
- Preserve the original high-level task category whenever possible: optimization stays optimization, migration stays migration, analysis stays analysis, security stays security.
- Preserve the original tone and style whenever possible.
- If the agent only completed part of a multi-step task, keep only the completed subset.
- Do not claim performance improvements, verification success, or correctness unless directly supported.
- Do not invent tables, views, indexes, functions, triggers, or results not evidenced by the trajectory.
- Do not drift into a different task just because the trajectory explored adjacent objects.
- Prefer deleting unsupported details over replacing them with new speculative details.
- Prefer weakening exact claims into smaller supported claims instead of adding new deliverables.
- The final task should feel like the original task with unsupported parts removed, not like a new task.
- For multi-part checklists, keep only the checklist items that are directly supported by successful steps.
- For analysis / optimization / audit tasks, do not keep the full investigation checklist unless the trajectory truly covered it.
- If the trajectory only completed schema inspection, one query profile, one index creation, or one partial rewrite, the revised task must reflect only that narrower completed scope.
- If the trajectory never verified correctness or performance improvement, remove or weaken verification and improvement claims.
- If the trajectory only created some of the requested artifacts, keep only those created artifacts.

Consistency checklist:
- High-level intent should still match the original task.
- Concrete deliverables should closely match successful trajectory steps.
- If a requirement is not supported by the trajectory, remove it or soften it.
- If the trajectory only completed setup, inspection, or partial SQL changes, the revised task should reflect only that partial scope.
- If the original task contains numbered requirements, the revised task should usually contain fewer items unless the trajectory clearly completed most of them.

Example

ORIGINAL TASK:
Optimize a sales reporting workflow. The task requires:
1. Create a monthly summary table
2. Create two supporting indexes
3. Build a refresh function
4. Verify the dashboard query runs under 50ms

OBSERVED CHANGES:
- created table: monthly_summary
- created index: idx_sales_customer_id
- created index: idx_sales_order_date

REVISED TASK:
Please use PostgreSQL tools to finish the following task:
Optimize a sales reporting workflow. The task requires:
1. Create a monthly summary table
2. Create two supporting indexes

Example 2

ORIGINAL TASK:
Implement a customer migration workflow that imports source records, validates them, inserts transformed rows into the target table, and verifies all 200 records were migrated.

OBSERVED CHANGES:
- inspected source and target tables
- created a staging table
- inserted a subset of transformed rows into the target table

REVISED TASK:
Please use PostgreSQL tools to finish the following task:
Implement a customer migration workflow that imports source records and inserts transformed rows into the target table.

Now revise the following task.

ORIGINAL TASK:
{original_task}

OBSERVED CHANGES:
{change_summary}

TRAJECTORY:
{trajectory_str}

REVISED TASK:
"""


def normalize_task_inference_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    required_prefix = "Please use PostgreSQL tools to finish the following task:"
    for prefix in (
        required_prefix,
        "REQUEST:",
        "User Request:",
        "Request:",
        "Instruction:",
        "Simulated User Request:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return required_prefix + "\n" + text


def build_credit_prompt_prefix(trajectory_str: str) -> str:
    if trajectory_str:
        return f"""An AI agent is solving a PostgreSQL database task with tool calls.
Based on the tool actions observed so far, what was the user's original PostgreSQL task?

Actions observed:
{trajectory_str}

The user's original PostgreSQL task was:
"""
    return """An AI agent is about to solve a PostgreSQL database task with tool calls.
Without seeing any actions, what was the user's original PostgreSQL task?

The user's original PostgreSQL task was:
"""
