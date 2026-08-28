"""Infer retrospective tasks from tool calls and environment responses."""

from __future__ import annotations

import argparse
import asyncio
import glob
import importlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
from openai import AsyncOpenAI


DEFAULT_MODEL = "./Qwen3-32B"
DEFAULT_BASE_URL = "http://localhost:2024/v1"
DEFAULT_API_KEY = ""


def get_environment(name: str) -> ModuleType:
    try:
        return importlib.import_module(
            f"agentbrew.environments.{name}.experience_distillation"
        )
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown environment {name!r}. Expected an "
            f"agentbrew.environments.{name}.experience_distillation module."
        ) from exc


def resolve_input_files(input_path: str) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(glob.glob(str(path / "**" / "*.json"), recursive=True))


def read_tasks(json_path: str) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw.get("tasks", raw.get("results", raw))
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"Unsupported JSON shape in {json_path}")
    return [item for item in raw if isinstance(item, dict)]


def prepare_task(
    task: dict[str, Any],
    source_path: str,
    source_index: int,
    environment: ModuleType,
) -> dict[str, Any] | None:
    tool_steps: list[dict[str, Any]] = []
    for step in task.get("trajectory", []):
        if not isinstance(step, dict) or step.get("state") != "tool":
            continue
        normalized = dict(step)
        normalized["isError"] = bool(environment.is_error(normalized))
        tool_steps.append(normalized)
    if not tool_steps:
        return None

    task_id = task.get("task_id") or f"{source_path}:{source_index}"
    error_count = sum(bool(step["isError"]) for step in tool_steps)
    return {
        "task_id": str(task_id),
        "question": task.get("question", ""),
        "final_answer": task.get("final_answer"),
        "trajectory": tool_steps,
        "tool_count": len(tool_steps),
        "error_count": error_count,
        "error_ratio": error_count / len(tool_steps),
        "source_path": source_path,
        "source_index": source_index,
    }


def load_tasks(input_path: str, environment: ModuleType) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for json_path in resolve_input_files(input_path):
        try:
            raw_tasks = read_tasks(json_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        for index, raw_task in enumerate(raw_tasks):
            task = prepare_task(raw_task, json_path, index, environment)
            if task is not None and environment.is_eligible_task(task):
                tasks.append(task)
    return tasks


def build_client(api_key: str, base_url: str, timeout: float) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(timeout=timeout),
        ),
    )


async def infer_retrospective_task(
    task: dict[str, Any],
    environment: ModuleType,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
    retries: int,
) -> str:
    # Error labels are environment-specific and must be finalized before any
    # retrospective inference. Failed actions are retained in the serialized
    # result for auditability, but they are not evidence of completed work.
    valid_trajectory = [
        step for step in task["trajectory"] if not environment.is_error(step)
    ]
    if not valid_trajectory:
        raise ValueError("No successful tool actions available for task inference")

    trajectory_text = environment.format_trajectory(valid_trajectory)
    change_summary = environment.summarize_changes(valid_trajectory)
    prompt = environment.build_task_inference_prompt(
        task["question"],
        trajectory_text,
        change_summary,
    )

    for attempt in range(retries):
        try:
            async with semaphore:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=3000,
                        temperature=0.7,
                        top_p=0.9,
                        extra_body={
                            "top_k": 20,
                            "repetition_penalty": 1.05,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    ),
                    timeout=timeout,
                )
            content = environment.normalize_task_inference_output(
                response.choices[0].message.content or ""
            )
            if len(content) < 20:
                raise ValueError(f"Generated task too short: {content!r}")
            return content
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1.5)
    raise RuntimeError("Task inference exhausted retries")


async def process_task(
    task: dict[str, Any],
    environment: ModuleType,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
    retries: int,
) -> dict[str, Any] | None:
    try:
        retrospective_task = await infer_retrospective_task(
            task,
            environment,
            client,
            model,
            semaphore,
            timeout,
            retries,
        )
    except Exception as exc:
        print(f"[warn] task inference failed for {task['task_id']}: {exc}")
        return None

    result = dict(task)
    result["original_task"] = result.pop("question")
    result["original_answer"] = result.pop("final_answer")
    result["retrospective_task"] = retrospective_task
    result["error_ratio"] = round(result["error_ratio"], 4)
    return result


def load_existing_results(output_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_results(output_path: str, results: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, output_path)


async def run(args: argparse.Namespace) -> None:
    environment = get_environment(args.environment)
    tasks = load_tasks(args.input_path, environment)
    results = load_existing_results(args.output_path)
    completed = {str(item.get("task_id")) for item in results}
    pending = [task for task in tasks if task["task_id"] not in completed]

    client = build_client(args.api_key, args.base_url, max(args.timeout, 180.0))
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)
    write_lock = asyncio.Lock()

    async def process_and_save(task: dict[str, Any]) -> None:
        result = await process_task(
            task,
            environment,
            client,
            args.model,
            semaphore,
            args.timeout,
            args.retries,
        )
        if result is None:
            return
        async with write_lock:
            results.append(result)
            save_results(args.output_path, results)
            print(f"[task-inference] {len(results)}/{len(tasks)} {task['task_id']}")

    try:
        await asyncio.gather(*(process_and_save(task) for task in pending))
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer retrospective tasks from tool-use trajectories."
    )
    parser.add_argument("--environment", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--max-concurrent-requests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
