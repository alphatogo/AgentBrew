"""Notion task evaluation: dispatches to per-task verifiers."""

from __future__ import annotations

import importlib
import logging

from notion_client import Client

from agentbrew.core.environment import EvaluationResult
from agentbrew.core.task import TaskSpec

logger = logging.getLogger(__name__)

# "mcpmark.notion.verify_employee_onboarding" → task_id = "employee_onboarding"
_VERIFIER_PREFIX = "mcpmark.notion.verify_"


async def run_verifier(task: TaskSpec, eval_api_key: str) -> list[EvaluationResult]:
    """Resolve and run the verifier for a completed Notion task."""
    if not task.evaluation or not task.evaluation.verifier:
        return []

    verifier_name = task.evaluation.verifier  # e.g. "mcpmark.notion.verify_employee_onboarding"
    category = task.category  # e.g. "company_in_a_box"

    if not verifier_name.startswith(_VERIFIER_PREFIX):
        return [EvaluationResult(
            passed=False,
            verifier=verifier_name,
            error=f"Unrecognised verifier name format: {verifier_name!r}",
        )]

    task_id = verifier_name[len(_VERIFIER_PREFIX):]  # "employee_onboarding"

    if not category or not task_id:
        return [EvaluationResult(
            passed=False,
            verifier=verifier_name,
            error=f"Cannot resolve verifier: category={category!r}, task_id={task_id!r}",
        )]

    module_path = (
        f"agentbrew.environments.notion.benchmark.evaluator.{category}.{task_id}.verify"
    )

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        logger.error("Verifier module not found: %s — %s", module_path, exc)
        return [EvaluationResult(
            passed=False,
            verifier=verifier_name,
            error=f"Verifier module not found: {module_path}",
        )]

    try:
        notion = Client(auth=eval_api_key)
        passed, reason = mod.verify(notion, None)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Verifier %s raised: %s", verifier_name, exc)
        return [EvaluationResult(
            passed=False,
            verifier=verifier_name,
            error=str(exc),
        )]

    return [EvaluationResult(
        passed=passed,
        verifier=verifier_name,
        reason=reason,
    )]
