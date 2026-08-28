"""GitHub environment preparation."""

from __future__ import annotations

from agentbrew.core.context import Context
from agentbrew.core.environment import RuntimeState
from agentbrew.core.task import TaskSpec


async def prepare_github_task(
    task: TaskSpec | None,
    context: Context,
) -> RuntimeState:
    """Bind one configured GitHub account to the current task."""
    token = context.get_env("GITHUB_PERSONAL_ACCESS_TOKEN")
    account = context.get_env("GITHUB_PERSONAL_ACCOUNT_NAME")
    if not token:
        raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")
    if not account:
        raise ValueError("Missing GITHUB_PERSONAL_ACCOUNT_NAME")

    return RuntimeState(
        domain="github",
        task_id=task.id if task else None,
        resources={"account": account, "tool_calls": []},
        env={
            "GITHUB_PERSONAL_ACCESS_TOKEN": token,
            "GITHUB_PERSONAL_ACCOUNT_NAME": account,
        },
    )
