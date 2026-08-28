"""End-to-end, in-place experience distillation for trajectory files."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from tqdm.auto import tqdm

from .credit_assignment import CREDIT_METHODS, assign_task_credit
from .task_inference import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_client,
    get_environment,
    infer_retrospective_task,
    prepare_task,
)


DEFAULT_INPUT_PATH = "outputs/notion_trajectory_sample/trajectories"

_DISTILLATION_TASK_KEYS = (
    "hindsight_task",
    "retrospective_task",
    "L_minus_1_prior",
    "credit_method",
    "experience_distillation",
)


def _task_list(payload: Any, source_path: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            return [task for task in tasks if isinstance(task, dict)]
    raise ValueError(f"Expected {{'tasks': [...]}} trajectory payload in {source_path}")


def _is_llm_action(step: dict[str, Any]) -> bool:
    if step.get("state") != "llm":
        return False
    parsed = step.get("parsed_response")
    return isinstance(parsed, dict) and isinstance(parsed.get("action"), dict)


def trajectory_is_distilled(task: dict[str, Any]) -> bool:
    """Return whether a task already has hindsight metadata and all action credits."""
    if not (task.get("hindsight_task") or task.get("retrospective_task")):
        return False
    if "L_minus_1_prior" not in task or "credit_method" not in task:
        return False
    action_steps = [
        step
        for step in task.get("trajectory", [])
        if isinstance(step, dict) and _is_llm_action(step)
    ]
    return bool(action_steps) and all(
        "credit" in step and "L_prefix" in step for step in action_steps
    )


def trajectory_is_processed(task: dict[str, Any]) -> bool:
    """Return whether either credit or a terminal filtering record was written."""
    metadata = task.get("experience_distillation")
    if isinstance(metadata, dict) and metadata.get("status") == "filtered":
        if "hindsight_task" not in task or task.get("hindsight_task") != "":
            return False
        pending_action: dict[str, Any] | None = None
        for step in task.get("trajectory", []):
            if not isinstance(step, dict):
                continue
            if _is_llm_action(step):
                pending_action = step
            elif step.get("state") == "tool":
                if (
                    bool(step.get("isError", False))
                    and pending_action is not None
                    and pending_action.get("credit") != 0.0
                ):
                    return False
                pending_action = None
        return True
    return trajectory_is_distilled(task)


def persist_error_labels(task: dict[str, Any], environment: ModuleType) -> bool:
    """Persist environment-derived labels on every tool response."""
    changed = False
    for step in task.get("trajectory", []):
        if not isinstance(step, dict) or step.get("state") != "tool":
            continue
        is_error = bool(environment.is_error(step))
        if step.get("isError") is not is_error:
            step["isError"] = is_error
            changed = True
    return changed


def attach_filtered_error_credits(
    task: dict[str, Any], environment: ModuleType
) -> bool:
    """Assign deterministic zero only to failed actions in filtered data."""
    changed = False
    pending_action: dict[str, Any] | None = None
    for step in task.get("trajectory", []):
        if not isinstance(step, dict):
            continue
        if _is_llm_action(step):
            pending_action = step
            continue
        if step.get("state") != "tool":
            continue
        if environment.is_error(step) and pending_action is not None:
            if pending_action.get("credit") != 0.0:
                pending_action["credit"] = 0.0
                changed = True
            if pending_action.get("L_prefix") is not None:
                pending_action["L_prefix"] = None
                changed = True
        pending_action = None
    return changed


def reset_distillation(
    task: dict[str, Any], environment: ModuleType
) -> bool:
    """Remove derived values and persist environment-specific error labels.

    This is used by ``--overwrite`` so a detector change cannot leave stale
    hindsight tasks or credits on trajectories that are no longer eligible.
    """
    changed = False
    for key in _DISTILLATION_TASK_KEYS:
        if key in task:
            task.pop(key)
            changed = True

    for step in task.get("trajectory", []):
        if not isinstance(step, dict):
            continue
        if _is_llm_action(step) and "credit" in step:
            step.pop("credit")
            changed = True
        if _is_llm_action(step) and "L_prefix" in step:
            step.pop("L_prefix")
            changed = True
    changed = persist_error_labels(task, environment) or changed
    return changed


def build_distillation_metadata(
    normalized: dict[str, Any], environment: ModuleType, *, eligible: bool
) -> dict[str, Any]:
    metadata = {
        "environment": environment.__name__.split(".")[-2],
        "tool_count": normalized["tool_count"],
        "error_count": normalized["error_count"],
        "error_ratio": round(normalized["error_ratio"], 4),
        "eligible": eligible,
        "status": "completed" if eligible else "filtered",
    }
    if not eligible:
        reason_fn = getattr(environment, "eligibility_failure_reason", None)
        metadata["filter_reason"] = (
            reason_fn(normalized) if callable(reason_fn) else "environment_filter"
        )
    return metadata


def attach_action_credits(
    full_trajectory: list[dict[str, Any]],
    credited_tool_trajectory: list[dict[str, Any]],
    environment: ModuleType,
) -> list[dict[str, Any]]:
    """Attach each tool result's credit to the LLM action that produced it.

    The original interleaved trajectory is preserved. Tool errors are relabeled
    with the environment-specific detector before their corresponding LLM
    actions receive zero credit.
    """
    output = [dict(step) for step in full_trajectory]
    for step in output:
        if _is_llm_action(step):
            # An unmatched action is not allowed to inherit a later action's
            # credit. Zero is the safe value for incomplete trajectories.
            step["credit"] = 0.0
            step["L_prefix"] = None

    pending_action_index: int | None = None
    tool_index = 0
    for index, step in enumerate(output):
        if _is_llm_action(step):
            pending_action_index = index
            continue
        if step.get("state") != "tool":
            continue
        if tool_index >= len(credited_tool_trajectory):
            raise ValueError("Full trajectory has more tool steps than the credited trajectory")

        credited_tool = credited_tool_trajectory[tool_index]
        tool_index += 1
        is_error = bool(environment.is_error(step))
        step["isError"] = is_error
        credit = 0.0 if is_error else float(credited_tool.get("credit", 0.0))
        l_prefix = None if is_error else credited_tool.get("L_prefix")

        if pending_action_index is not None:
            output[pending_action_index]["credit"] = round(credit, 4)
            output[pending_action_index]["L_prefix"] = l_prefix
            pending_action_index = None

    if tool_index != len(credited_tool_trajectory):
        raise ValueError("Credited trajectory has more tool steps than the full trajectory")
    return output


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically replace one trajectory JSON without leaving a partial file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


async def distill_task(
    raw_task: dict[str, Any],
    source_path: str,
    source_index: int,
    environment: ModuleType,
    client: Any,
    model: str,
    chat_semaphore: asyncio.Semaphore,
    logprob_semaphore: asyncio.Semaphore,
    credit_method: str,
    hindsight_timeout: float,
    logprob_timeout: float,
    retries: int,
) -> bool:
    """Infer a hindsight task, assign tool credit, and update one raw task."""
    labels_changed = persist_error_labels(raw_task, environment)
    normalized = prepare_task(raw_task, source_path, source_index, environment)
    if normalized is None:
        metadata = {
            "environment": environment.__name__.split(".")[-2],
            "tool_count": 0,
            "error_count": 0,
            "error_ratio": 0.0,
            "eligible": False,
            "status": "filtered",
            "filter_reason": "no_tool_actions",
        }
        changed = labels_changed or raw_task.get("experience_distillation") != metadata
        changed = changed or raw_task.get("hindsight_task") != ""
        raw_task["hindsight_task"] = ""
        raw_task["experience_distillation"] = metadata
        return changed

    eligible = bool(environment.is_eligible_task(normalized))
    metadata = build_distillation_metadata(normalized, environment, eligible=eligible)
    if not eligible:
        error_credits_changed = attach_filtered_error_credits(raw_task, environment)
        changed = (
            labels_changed
            or error_credits_changed
            or raw_task.get("experience_distillation") != metadata
        )
        changed = changed or raw_task.get("hindsight_task") != ""
        raw_task["hindsight_task"] = ""
        raw_task["experience_distillation"] = metadata
        return changed

    hindsight_task = await infer_retrospective_task(
        task=normalized,
        environment=environment,
        client=client,
        model=model,
        semaphore=chat_semaphore,
        timeout=hindsight_timeout,
        retries=retries,
    )
    credit_input = dict(normalized)
    credit_input["hindsight_task"] = hindsight_task
    credited = await assign_task_credit(
        task=credit_input,
        environment=environment,
        client=client,
        model=model,
        semaphore=logprob_semaphore,
        method=credit_method,
        timeout=logprob_timeout,
        retries=retries,
    )
    if credited is None:
        return False

    raw_task["trajectory"] = attach_action_credits(
        raw_task.get("trajectory", []),
        credited["trajectory"],
        environment,
    )
    raw_task["hindsight_task"] = hindsight_task
    raw_task["L_minus_1_prior"] = credited["L_minus_1_prior"]
    raw_task["credit_method"] = credit_method
    raw_task["experience_distillation"] = metadata
    return True


async def run(args: argparse.Namespace) -> None:
    root = Path(args.input_path)
    if root.is_file():
        paths = [root]
    elif root.is_dir():
        paths = sorted(root.rglob(args.input_glob))
    else:
        raise FileNotFoundError(root)
    if args.limit > 0:
        paths = paths[: args.limit]

    environment = get_environment(args.environment)
    credit_method = args.credit_method or getattr(
        environment, "DEFAULT_CREDIT_METHOD", "adjacent"
    )
    client = build_client(args.api_key, args.base_url, max(args.timeout, 180.0))
    file_semaphore = asyncio.Semaphore(args.max_concurrent_files)
    task_semaphore = asyncio.Semaphore(args.max_concurrent_tasks)
    chat_semaphore = asyncio.Semaphore(args.max_concurrent_chat)
    logprob_semaphore = asyncio.Semaphore(args.max_concurrent_logprob)
    progress = tqdm(total=len(paths), desc="experience-distillation", unit="file", dynamic_ncols=True)
    progress_lock = asyncio.Lock()
    counts = {"completed": 0, "skipped": 0, "failed": 0}

    async def update(status: str) -> None:
        async with progress_lock:
            counts[status] += 1
            progress.set_postfix(counts, refresh=False)
            progress.update()

    async def process_path(path: Path) -> None:
        try:
            async with file_semaphore:
                payload = json.loads(path.read_text(encoding="utf-8"))
            tasks = _task_list(payload, str(path))
            if not tasks:
                await update("skipped")
                return
            if not args.overwrite and all(trajectory_is_processed(task) for task in tasks):
                await update("skipped")
                return

            changed = False
            async with task_semaphore:
                for index, task in enumerate(tasks):
                    if args.overwrite:
                        changed = reset_distillation(task, environment) or changed
                    if not args.overwrite and trajectory_is_processed(task):
                        continue
                    changed = (
                        await distill_task(
                            raw_task=task,
                            source_path=str(path),
                            source_index=index,
                            environment=environment,
                            client=client,
                            model=args.model,
                            chat_semaphore=chat_semaphore,
                            logprob_semaphore=logprob_semaphore,
                            credit_method=credit_method,
                            hindsight_timeout=args.hindsight_timeout,
                            logprob_timeout=args.logprob_timeout,
                            retries=args.retries,
                        )
                        or changed
                    )
            if not changed:
                await update("skipped")
                return
            async with file_semaphore:
                await asyncio.to_thread(write_json_atomic, path, payload)
            await update("completed")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[warn] {path}: {exc}")
            await update("failed")

    try:
        await asyncio.gather(*(process_path(path) for path in paths))
    finally:
        progress.close()
        await client.close()

    print(
        f"Finished {len(paths)} files: completed={counts['completed']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer hindsight tasks and write per-LLM-action credit into trajectory JSON files."
    )
    parser.add_argument("--environment", default="notion")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--input-glob", default="*.json")
    parser.add_argument(
        "--credit-method",
        choices=sorted(CREDIT_METHODS),
        default=None,
        help="Override the environment default (Notion/PostgreSQL: best_prefix).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--hindsight-timeout", type=float, default=120.0)
    parser.add_argument("--logprob-timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-concurrent-files", type=int, default=8)
    parser.add_argument("--max-concurrent-tasks", type=int, default=64)
    parser.add_argument("--max-concurrent-chat", type=int, default=16)
    parser.add_argument("--max-concurrent-logprob", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
