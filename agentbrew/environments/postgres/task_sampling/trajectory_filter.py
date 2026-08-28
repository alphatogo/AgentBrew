#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter completely unusable SQL trajectories with a local OpenAI-compatible vLLM model.

Goal:
- Reject trajectories that made essentially no meaningful progress toward the task goal.
- Reject tasks that are themselves clearly broken or effectively non-executable.
- Keep trajectories that made any substantive partial progress, even if incomplete or later blocked.

Examples to reject:
- Mostly schema exploration plus repeated errors, with no meaningful successful task action.
- Blocked by missing environment support (e.g. pg_stat_statements preload) before doing anything useful.
- Final answer just reports a blocker and the tool trace shows no partial completion.

Examples to keep:
- Completed some deliverables but not all.
- Created/fixed part of the required objects before hitting a blocker.
- Found real issues or applied real SQL changes that move the task forward.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from openai import AsyncOpenAI


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/postgres_trajectory_sample/trajectories"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/postgres_trajectory_sample_filtered"

DEFAULT_MODEL_NAME = "./Qwen3-32B"
DEFAULT_BASE_URL = "http://localhost:2024/v1"
DEFAULT_API_KEY = ""
DEFAULT_MAX_CONCURRENCY = 32
DEFAULT_MAX_COMPLETION_TOKENS = 280

WRITE_SQL_PREFIXES = (
    "create ",
    "alter ",
    "insert ",
    "update ",
    "delete ",
    "drop ",
    "truncate ",
    "merge ",
    "grant ",
    "revoke ",
    "comment ",
    "refresh materialized view",
    "reindex ",
)
SUBSTANTIVE_SQL_KEYWORDS = (
    "create table",
    "create view",
    "create materialized view",
    "create index",
    "create trigger",
    "create function",
    "create policy",
    "insert into",
    "update ",
    "delete from",
    "alter table",
    "refresh materialized view",
    "explain",
)

SYSTEM_PROMPT = """You are judging SQL agent trajectories for SFT filtering.

Task:
- Decide whether a trajectory should be KEPT or REJECTED.

Reject ONLY when the trajectory is completely unusable for learning task progress:
- It made essentially no meaningful progress toward the task goal.
- It only explored schema or repeated failed SQL without any substantive successful action.
- It was blocked by missing environment support before accomplishing anything useful.
- It never created, fixed, inserted, updated, validated, or discovered anything that materially advances the task.

Also reject when the TASK ITSELF is clearly broken or effectively non-executable:
- The task requires optimizing/profiling a baseline SQL query that is itself invalid and cannot run.
- The task has self-contradictory requirements that prevent meaningful completion.
- The task depends on unavailable environment features in a way that makes the required deliverables impossible, not just harder.

Keep when there is any meaningful partial progress:
- It completed part of the task.
- It created some required objects or populated some required data.
- It discovered real missing indexes / integrity issues / schema facts that are directly useful.
- It made durable SQL changes before later failing.
- It did half the work, then got blocked by environment issues.

Important:
- Be conservative with rejection.
- Partial completion should be KEPT.
- Repeated errors alone are not enough for rejection if some real progress happened.
- If rejecting due to task design, say that clearly in the reason.

Return strict JSON only:
{"label":"KEEP"|"REJECT","confidence":0.0-1.0,"reason":"short explanation"}
"""


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def shorten(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def compress_multiline_text(text: str, max_chars: int = 500, max_lines: int = 8) -> str:
    raw = str(text).strip()
    if not raw:
        return raw
    lines = raw.splitlines()
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines - len(head)) :]
        raw = "\n".join(head + ["..."] + tail)
    if len(raw) > max_chars:
        keep_head = max_chars // 2
        keep_tail = max_chars - keep_head - 5
        raw = raw[:keep_head].rstrip() + "\n...\n" + raw[-keep_tail:].lstrip()
    return raw


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dumps_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_existing_results(report_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    existing: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not report_path.exists():
        return existing
    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (item.get("source_file", ""), int(item.get("task_index", -1)))
            if key[0] and key[1] >= 0:
                existing[key] = item
    return existing


def detect_sql_kind(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    for prefix in WRITE_SQL_PREFIXES:
        if normalized.startswith(prefix):
            return "write"
    if normalized.startswith("select ") or normalized.startswith("with "):
        return "read"
    return "other"


def is_substantive_sql(sql: str) -> bool:
    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    return any(keyword in normalized for keyword in SUBSTANTIVE_SQL_KEYWORDS)


def tool_is_error(step: Dict[str, Any]) -> bool:
    content = str(step.get("content", ""))
    lowered = content.lstrip()
    return bool(step.get("isError")) or lowered.startswith("Error:") or lowered.startswith("ERROR:")


def sanitize_tool_step(step: Dict[str, Any]) -> Dict[str, Any]:
    tool_name = step.get("tool_name")
    arguments = step.get("arguments", {})
    content = step.get("content", "")
    sanitized = {
        "state": "tool",
        "tool_name": tool_name,
        "arguments": arguments,
        "content": compress_multiline_text(content, max_chars=700, max_lines=10),
        "isError": tool_is_error(step),
    }
    server = step.get("server")
    if server is not None:
        sanitized["server"] = server
    return sanitized


def sanitize_task_for_llm(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "question": task.get("question"),
        "final_answer": task.get("final_answer"),
        "trajectory": [
            sanitize_tool_step(step)
            for step in task.get("trajectory", [])
            if step.get("state") == "tool"
        ],
    }


def summarize_tool_step(step: Dict[str, Any]) -> str:
    tool_name = step.get("tool_name", "")
    if tool_name == "execute_sql":
        sql = step.get("arguments", {}).get("sql", "")
        sql_one_line = re.sub(r"\s+", " ", sql.strip())
        status = "ERROR" if tool_is_error(step) else "OK"
        content = shorten(compress_multiline_text(step.get("content", ""), max_chars=180, max_lines=4), 180)
        return f"execute_sql [{status}] sql={shorten(sql_one_line, 220)} result={content}"
    status = "ERROR" if tool_is_error(step) else "OK"
    args = shorten(json.dumps(step.get("arguments", {}), ensure_ascii=False), 160)
    content = shorten(compress_multiline_text(step.get("content", ""), max_chars=160, max_lines=4), 160)
    return f"{tool_name} [{status}] args={args} result={content}"


def extract_sql_blocks(text: str) -> List[str]:
    return [block.strip() for block in re.findall(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)]


@dataclass
class TrajectoryFeatures:
    total_tools: int
    total_errors: int
    successful_execute_sql: int
    successful_write_sql: int
    successful_substantive_sql: int
    schema_tools: int
    blocker_mentions: List[str]
    explain_query_errors: int
    baseline_query_invalid_signal: bool
    task_sql_block_count: int


def extract_features(task: Dict[str, Any]) -> TrajectoryFeatures:
    tools = [step for step in task.get("trajectory", []) if step.get("state") == "tool"]
    question = str(task.get("question", ""))
    total_errors = 0
    successful_execute_sql = 0
    successful_write_sql = 0
    successful_substantive_sql = 0
    schema_tools = 0
    blocker_mentions: List[str] = []
    explain_query_errors = 0
    baseline_query_invalid_signal = False
    task_sql_block_count = len(extract_sql_blocks(question))

    for step in tools:
        tool_name = step.get("tool_name", "")
        content = str(step.get("content", ""))
        is_error = tool_is_error(step)
        if is_error:
            total_errors += 1
            lowered = content.lower()
            if any(
                token in lowered
                for token in (
                    "pg_stat_statements must be loaded",
                    "shared_preload_libraries",
                    "extension is not available",
                    "must first be installed on the system",
                    "permission denied",
                    "no such file or directory",
                )
            ):
                blocker_mentions.append(shorten(content, 160))
            if tool_name == "explain_query":
                explain_query_errors += 1
                lowered = content.lower()
                if any(
                    token in lowered
                    for token in (
                        "subquery uses ungrouped column",
                        "must appear in the group by clause",
                        "syntax error",
                        "column does not exist",
                        "relation does not exist",
                        "operator does not exist",
                    )
                ):
                    baseline_query_invalid_signal = True
        if tool_name in {"list_schemas", "list_objects", "get_object_details"}:
            schema_tools += 1
        if tool_name == "execute_sql" and not is_error:
            successful_execute_sql += 1
            sql = step.get("arguments", {}).get("sql", "")
            if detect_sql_kind(sql) == "write":
                successful_write_sql += 1
            if is_substantive_sql(sql):
                successful_substantive_sql += 1

    return TrajectoryFeatures(
        total_tools=len(tools),
        total_errors=total_errors,
        successful_execute_sql=successful_execute_sql,
        successful_write_sql=successful_write_sql,
        successful_substantive_sql=successful_substantive_sql,
        schema_tools=schema_tools,
        blocker_mentions=blocker_mentions[:3],
        explain_query_errors=explain_query_errors,
        baseline_query_invalid_signal=baseline_query_invalid_signal,
        task_sql_block_count=task_sql_block_count,
    )


def build_prompt(task: Dict[str, Any]) -> str:
    llm_task = sanitize_task_for_llm(task)
    question = llm_task.get("question", "")
    final_answer = llm_task.get("final_answer")
    tools = llm_task.get("trajectory", [])
    features = extract_features(task)
    sql_blocks = extract_sql_blocks(question)

    payload = {
        "question": shorten(question, 2500),
        "task_sql_blocks": [shorten(block, 1200) for block in sql_blocks[:2]],
        "final_answer": None if final_answer is None else shorten(str(final_answer), 1000),
        "trajectory_stats": {
            "total_tool_steps": features.total_tools,
            "total_errors": features.total_errors,
            "successful_execute_sql": features.successful_execute_sql,
            "successful_write_sql": features.successful_write_sql,
            "successful_substantive_sql": features.successful_substantive_sql,
            "schema_tools": features.schema_tools,
            "blocker_mentions": features.blocker_mentions,
            "explain_query_errors": features.explain_query_errors,
            "baseline_query_invalid_signal": features.baseline_query_invalid_signal,
            "task_sql_block_count": features.task_sql_block_count,
        },
        "tool_trajectory": tools,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_model_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if not text:
        return {
            "label": "REJECT",
            "confidence": 0.0,
            "reason": "empty model output",
        }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {
                "label": "REJECT",
                "confidence": 0.0,
                "reason": f"non_json_model_output: {shorten(text, 200)}",
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "label": "REJECT",
                "confidence": 0.0,
                "reason": f"malformed_json_model_output: {shorten(text, 200)}",
            }


def heuristic_label(features: TrajectoryFeatures, final_answer: Any) -> Optional[Tuple[str, str]]:
    final_text = "" if final_answer is None else str(final_answer).lower()

    if (
        features.task_sql_block_count >= 1
        and features.baseline_query_invalid_signal
        and features.successful_substantive_sql == 0
        and features.successful_write_sql == 0
    ):
        return ("REJECT", "task appears broken or non-executable: baseline SQL/query artifact is invalid")

    # Strong keep: any substantive successful SQL is enough to preserve partial progress.
    if features.successful_substantive_sql >= 2 or features.successful_write_sql >= 1:
        return ("KEEP", "has substantive successful SQL progress")

    # Strong reject: essentially exploration/errors only, with blocker final answer and no substantive progress.
    if (
        features.successful_substantive_sql == 0
        and features.successful_write_sql == 0
        and features.total_tools >= 4
        and features.total_errors >= max(2, features.total_tools // 3)
        and (
            "pg_stat_statements" in final_text
            or "shared_preload_libraries" in final_text
            or "manual intervention" in final_text
            or "cannot proceed" in final_text
            or "must be added" in final_text
        )
    ):
        return ("REJECT", "blocked by environment and made no substantive progress")

    return None


async def judge_task(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    max_completion_tokens: int,
    file_path: Path,
    task: Dict[str, Any],
    file_label: str,
    index: int,
    judge_all_with_llm: bool,
) -> Dict[str, Any]:
    features = extract_features(task)
    heuristic = None if judge_all_with_llm else heuristic_label(features, task.get("final_answer"))
    prompt = build_prompt(task)

    if heuristic is not None:
        label, reason = heuristic
        return {
            "source_file": file_label,
            "task_index": index,
            "task_id": task.get("task_id"),
            "label": label,
            "confidence": 0.99,
            "reason": reason,
            "judger": "heuristic",
            "features": features.__dict__,
        }

    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model_name,
                max_completion_tokens=max_completion_tokens,
                temperature=0.7,
                top_p=0.9,
                extra_body={"top_k": 20,
                            "repetition_penalty": 1.05,
                            "chat_template_kwargs": {"enable_thinking": False},
                            },
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

        raw_text = response.choices[0].message.content or ""
        parsed = parse_model_json(raw_text)
        label = str(parsed.get("label", "REJECT")).upper()
        if label not in {"KEEP", "REJECT"}:
            label = "REJECT"
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        return {
            "source_file": file_label,
            "task_index": index,
            "task_id": task.get("task_id"),
            "label": label,
            "confidence": confidence,
            "reason": str(parsed.get("reason", "")).strip(),
            "judger": "llm",
            "raw_model_output": raw_text,
            "features": features.__dict__,
        }
    except Exception as exc:
        return {
            "source_file": file_label,
            "task_index": index,
            "task_id": task.get("task_id"),
            "label": "REJECT",
            "confidence": 0.0,
            "reason": f"llm_judge_exception: {type(exc).__name__}: {shorten(str(exc), 200)}",
            "judger": "llm_exception",
            "features": features.__dict__,
        }


async def run_async(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    files = sorted(input_root.glob("*/trajectories.json"))
    if not files:
        # AgentBrew writes the same {"tasks": [...]} payload as one JSON per task.
        # File discovery is the only format adaptation; judgment logic is unchanged.
        files = sorted(input_root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No trajectory JSON files found under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    kept_root = output_root / "kept"
    rejected_root = output_root / "rejected"
    report_path = output_root / "filter_report.jsonl"
    summary_path = output_root / "filter_summary.json"
    existing_results = load_existing_results(report_path) if args.resume else {}

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=normalize_base_url(args.base_url),
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    flush_every = 50

    jobs = []
    payloads: Dict[str, Dict[str, Any]] = {}
    for file_path in files:
        payload = load_json(file_path)
        payloads[str(file_path)] = payload
        for idx, task in enumerate(payload.get("tasks", [])):
            rel = str(file_path.relative_to(input_root))
            if (rel, idx) in existing_results:
                continue
            jobs.append(
                judge_task(
                    client=client,
                    semaphore=semaphore,
                    model_name=args.model_name,
                    max_completion_tokens=args.max_completion_tokens,
                    file_path=file_path,
                    task=task,
                    file_label=rel,
                    index=idx,
                    judge_all_with_llm=args.judge_all_with_llm,
                )
            )
            if args.limit_tasks is not None and len(jobs) >= args.limit_tasks:
                break
        if args.limit_tasks is not None and len(jobs) >= args.limit_tasks:
            break

    results: List[Dict[str, Any]] = list(existing_results.values())
    total = len(jobs)
    completed = 0
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_mode = "a" if args.resume else "w"
    with report_path.open(report_mode, encoding="utf-8") as report_file:
        for coro in asyncio.as_completed(jobs):
            result = await coro
            results.append(result)
            completed += 1

            payload = payloads[str(input_root / result["source_file"])]
            task = payload.get("tasks", [])[result["task_index"]]
            enriched = {
                **result,
                "question_preview": shorten(task.get("question", ""), 220),
                "final_answer_preview": None
                if task.get("final_answer") is None
                else shorten(str(task.get("final_answer")), 180),
            }
            report_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            if completed % flush_every == 0 or completed == total:
                report_file.flush()
                os.fsync(report_file.fileno())

            if completed % max(1, args.progress_every) == 0 or completed == total:
                print(
                    f"[{completed}/{total}] processed"
                    + (f" (resume reused={len(existing_results)})" if args.resume else ""),
                    file=sys.stderr,
                )

    grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["source_file"], {})[result["task_index"]] = result

    kept_files = 0
    rejected_files = 0
    kept_tasks = 0
    rejected_tasks = 0
    for file_path in files:
        rel = str(file_path.relative_to(input_root))
        payload = payloads[str(file_path)]
        per_task = grouped.get(rel, {})
        kept_payload = {**payload, "tasks": []}
        rejected_payload = {**payload, "tasks": []}

        for idx, task in enumerate(payload.get("tasks", [])):
            decision = per_task[idx]
            if decision["label"] == "KEEP":
                kept_payload["tasks"].append(task)
                kept_tasks += 1
            else:
                rejected_payload["tasks"].append(task)
                rejected_tasks += 1

        rel_path = Path(rel)
        if kept_payload["tasks"]:
            dumps_json(kept_payload, kept_root / rel_path)
            kept_files += 1
        if rejected_payload["tasks"]:
            dumps_json(rejected_payload, rejected_root / rel_path)
            rejected_files += 1

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_files": len(files),
        "resume": args.resume,
        "reused_existing_results": len(existing_results),
        "newly_judged_tasks": total,
        "kept_files": kept_files,
        "rejected_files": rejected_files,
        "kept_tasks": kept_tasks,
        "rejected_tasks": rejected_tasks,
        "label_counts": dict(Counter(result["label"] for result in results)),
        "judger_counts": dict(Counter(result["judger"] for result in results)),
        "model_name": args.model_name,
    }
    dumps_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter completely unusable SQL trajectories with vLLM/Qwen3-32B.")
    parser.add_argument("--input-root", type=str, default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit-tasks", type=int, default=None, help="Optional cap on newly judged tasks.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing filter_report.jsonl decisions and only judge missing tasks.")
    parser.add_argument(
        "--judge-all-with-llm",
        action="store_true",
        help="Send every task to the 32B model instead of letting heuristic rules directly decide some tasks.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
