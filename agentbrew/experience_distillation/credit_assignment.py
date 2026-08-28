"""Assign action credit from retrospective-task token NLL."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from types import ModuleType
from typing import Any, Callable

from openai import AsyncOpenAI

from .task_inference import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_client,
    get_environment,
    read_tasks,
    resolve_input_files,
    save_results,
)


def adjacent_credit(l_minus_1: float, prefix_nlls: list[float]) -> list[float]:
    previous = l_minus_1
    credits: list[float] = []
    for current in prefix_nlls:
        credits.append(round(previous - current, 4))
        previous = current
    return credits


def best_prefix_credit(l_minus_1: float, prefix_nlls: list[float]) -> list[float]:
    baseline = l_minus_1
    credits: list[float] = []
    for current in prefix_nlls:
        credit = round(baseline - current, 4)
        credits.append(credit)
        if credit >= 0:
            baseline = current
    return credits


CREDIT_METHODS: dict[str, Callable[[float, list[float]], list[float]]] = {
    "adjacent": adjacent_credit,
    "best_prefix": best_prefix_credit,
}


async def calculate_target_nll(
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    prompt_prefix: str,
    target_task: str,
    timeout: float,
    retries: int,
) -> float:
    full_prompt = prompt_prefix + target_task
    target_start_offset = len(prompt_prefix)

    for attempt in range(retries):
        try:
            async with semaphore:
                response = await asyncio.wait_for(
                    client.completions.create(
                        model=model,
                        prompt=full_prompt,
                        max_tokens=1,
                        echo=True,
                        logprobs=1,
                        temperature=0.0,
                    ),
                    timeout=timeout,
                )
            logprobs = response.choices[0].logprobs
            target_nll = 0.0
            token_count = 0
            for offset, logprob in zip(logprobs.text_offset, logprobs.token_logprobs):
                if offset >= target_start_offset and logprob is not None:
                    target_nll -= float(logprob)
                    token_count += 1
            if token_count == 0:
                raise ValueError("No target-task token logprobs returned")
            return target_nll / token_count
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[warn] calculate_target_nll failed: {exc}")
                return 0.0
            await asyncio.sleep(1.0)
    return 0.0


async def assign_task_credit(
    task: dict[str, Any],
    environment: ModuleType,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    method: str,
    timeout: float,
    retries: int,
) -> dict[str, Any] | None:
    target_task = task.get("retrospective_task") or task.get("hindsight_task")
    if not target_task:
        print(f"[warn] missing retrospective_task for {task.get('task_id')}")
        return None

    trajectory = [dict(step) for step in task.get("trajectory", [])]
    valid_steps = [step for step in trajectory if not environment.is_error(step)]
    if not valid_steps:
        return None

    prior_prompt = environment.build_credit_prompt_prefix("")
    prefix_prompts = [
        environment.build_credit_prompt_prefix(
            environment.format_trajectory(valid_steps[: index + 1])
        )
        for index in range(len(valid_steps))
    ]
    nlls = await asyncio.gather(
        calculate_target_nll(
            client,
            model,
            semaphore,
            prior_prompt,
            target_task,
            timeout,
            retries,
        ),
        *[
            calculate_target_nll(
                client,
                model,
                semaphore,
                prompt,
                target_task,
                timeout,
                retries,
            )
            for prompt in prefix_prompts
        ],
    )
    l_minus_1 = float(nlls[0])
    prefix_nlls = [float(value) for value in nlls[1:]]
    credits = CREDIT_METHODS[method](l_minus_1, prefix_nlls)

    valid_index = 0
    merged: list[dict[str, Any]] = []
    for step in trajectory:
        output_step = dict(step)
        output_step["isError"] = bool(environment.is_error(output_step))
        if output_step["isError"]:
            output_step["L_prefix"] = None
            output_step["credit"] = 0.0
        else:
            output_step["L_prefix"] = round(prefix_nlls[valid_index], 4)
            output_step["credit"] = credits[valid_index]
            valid_index += 1
        merged.append(output_step)

    result = dict(task)
    result["credit_method"] = method
    result["L_minus_1_prior"] = round(l_minus_1, 4)
    result["trajectory"] = merged
    return result


def load_credit_inputs(input_path: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in resolve_input_files(input_path):
        try:
            tasks.extend(read_tasks(path))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return tasks


async def run(args: argparse.Namespace) -> None:
    environment = get_environment(args.environment)
    tasks = load_credit_inputs(args.input_path)
    results: list[dict[str, Any]] = []
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, list):
                results = existing
        except (OSError, json.JSONDecodeError):
            pass
    completed = {str(item.get("task_id")) for item in results}
    pending = [task for task in tasks if str(task.get("task_id")) not in completed]

    client = build_client(args.api_key, args.base_url, max(args.timeout, 180.0))
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)
    write_lock = asyncio.Lock()

    async def process_and_save(task: dict[str, Any]) -> None:
        try:
            result = await assign_task_credit(
                task,
                environment,
                client,
                args.model,
                semaphore,
                args.credit_method,
                args.timeout,
                args.retries,
            )
        except Exception as exc:
            print(f"[warn] credit assignment failed for {task.get('task_id')}: {exc}")
            return
        if result is None:
            return
        async with write_lock:
            results.append(result)
            save_results(args.output_path, results)
            print(f"[credit-assignment] {len(results)}/{len(tasks)} {task.get('task_id')}")

    try:
        await asyncio.gather(*(process_and_save(task) for task in pending))
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign per-action credit from retrospective-task NLL."
    )
    parser.add_argument("--environment", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--credit-method",
        choices=sorted(CREDIT_METHODS),
        default="adjacent",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--max-concurrent-requests", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
