"""GitHub benchmark evaluation through the migrated Self-MCP pipeline."""

from __future__ import annotations

from agentbrew.core.context import Context
from agentbrew.core.environment import EvaluationResult
from agentbrew.core.task import TaskSpec
from .evaluator import Evaluator
from .evaluator.github import functions as _github_functions


async def run_evaluators(
    task: TaskSpec,
    result: str,
    context: Context,
) -> list[EvaluationResult]:
    """Run every legacy evaluator exactly through Self-MCP's evaluator chain."""
    configs = task.metadata.get("legacy", {}).get("evaluators") or []
    evaluations: list[EvaluationResult] = []
    token = _github_functions.set_evaluation_context(context)
    try:
        for config in configs:
            evaluator = Evaluator(config, context=context)
            legacy_result = await evaluator.evaluate(result)
            evaluations.append(
                EvaluationResult(
                    passed=legacy_result.passed,
                    verifier=config.get("op", ""),
                    reason=legacy_result.reason,
                    error=legacy_result.error,
                    metadata={
                        "config": legacy_result.config.model_dump(mode="json"),
                        "response": legacy_result.response,
                    },
                )
            )
    finally:
        _github_functions.reset_evaluation_context(token)
    return evaluations
