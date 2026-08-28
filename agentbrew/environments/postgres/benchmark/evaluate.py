"""Run bundled PostgreSQL benchmark verifiers in isolated subprocesses."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from agentbrew.core.environment import EvaluationResult
from agentbrew.core.task import TaskSpec

_RESULT_PREFIX = "AGENTBREW_POSTGRES_VERIFIER_RESULT="


async def run_verifiers(
    task: TaskSpec,
    *,
    env: dict[str, str],
    timeout_seconds: float,
) -> list[EvaluationResult]:
    """Run all evaluator operators without sharing process-global environment."""
    configs = task.metadata.get("legacy", {}).get("evaluators") or []
    results: list[EvaluationResult] = []
    for config in configs:
        verifier = str(config.get("op") or "")
        if not verifier:
            continue
        process_env = dict(os.environ)
        process_env.update(env)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "agentbrew.environments.postgres.benchmark.evaluator.subprocess",
            "--verifier",
            verifier,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            results.append(
                EvaluationResult(
                    passed=False,
                    verifier=verifier,
                    error=f"Verifier timed out after {timeout_seconds:g} seconds",
                )
            )
            continue

        output = stdout.decode("utf-8", errors="replace")
        error_output = stderr.decode("utf-8", errors="replace")
        payload: dict[str, Any] | None = None
        for line in reversed(output.splitlines()):
            if line.startswith(_RESULT_PREFIX):
                payload = json.loads(line[len(_RESULT_PREFIX) :])
                break
        if process.returncode or payload is None:
            results.append(
                EvaluationResult(
                    passed=False,
                    verifier=verifier,
                    error=(error_output or output or "Verifier returned no result").strip(),
                )
            )
            continue
        results.append(
            EvaluationResult(
                passed=bool(payload.get("passed")),
                verifier=verifier,
                reason=str(payload.get("reason") or ""),
                error=str(payload.get("error") or ""),
                metadata={"subprocess_stderr": error_output[-4000:]},
            )
        )
    return results
