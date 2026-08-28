"""GitHub task cleanup based on resources created in the execution trace."""

from __future__ import annotations

import asyncio
from typing import Any

import requests

from agentbrew.core.context import Context
from agentbrew.core.environment import RuntimeState
from agentbrew.core.logger import get_logger
from agentbrew.core.task import TaskSpec

logger = get_logger(__name__)

_API_URL = "https://api.github.com"
_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "AgentBrew",
}


def _resolve_cleanup_value(value: Any, tool_call: dict[str, Any]) -> Any:
    """Resolve Self-MCP cleanup references such as ``$name`` and ``$repo``."""
    if isinstance(value, dict):
        return {
            key: _resolve_cleanup_value(item, tool_call)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_cleanup_value(item, tool_call) for item in value]
    if not isinstance(value, str) or not value.startswith("$"):
        return value

    current: Any
    parts = [part.strip() for part in value.split("->") if part.strip()]
    source = parts[0][1:]
    if source == "return":
        current = tool_call.get("content")
    else:
        current = (tool_call.get("arguments") or {}).get(source)

    for operation in parts[1:]:
        name, _, raw_arg = operation.partition("(")
        argument = raw_arg.rstrip(")").strip()
        if name.strip().lower() == "get":
            current = current[argument]
        elif name.strip().lower() == "array":
            current = current[int(argument)]
        else:
            raise ValueError(f"Unsupported cleanup operation: {operation}")
    return current


def _delete_repository(token: str, owner: str, repo: str, timeout: float) -> bool:
    response = requests.delete(
        f"{_API_URL}/repos/{owner}/{repo}",
        headers={**_HEADERS, "Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    if response.status_code in {204, 404}:
        return True
    raise RuntimeError(
        f"Failed to delete GitHub repository {owner}/{repo}: "
        f"HTTP {response.status_code} {response.text}"
    )


async def cleanup_github_task(
    task: TaskSpec | None,
    state: RuntimeState,
    context: Context,
) -> None:
    """Run Self-MCP-style cleanup rules against matching successful tool calls."""
    if task is None:
        return

    legacy = task.metadata.get("legacy", {})
    cleanups = legacy.get("cleanups") or []
    tool_calls = state.resources.get("tool_calls") or []
    if not cleanups or not tool_calls:
        return

    token = context.get_env("GITHUB_PERSONAL_ACCESS_TOKEN")
    account = context.get_env("GITHUB_PERSONAL_ACCOUNT_NAME")
    timeout = float(context.get_env("GITHUB_CLEANUP_TIMEOUT_SECONDS", "30"))
    deleted: set[tuple[str, str]] = set()

    for tool_call in reversed(tool_calls):
        if tool_call.get("isError"):
            continue
        for cleanup in cleanups:
            if cleanup.get("server") != tool_call.get("server"):
                continue
            if cleanup.get("tool") != tool_call.get("tool_name"):
                continue
            if cleanup.get("cleanup_func") != "delete_repository":
                continue

            try:
                arguments = _resolve_cleanup_value(
                    cleanup.get("cleanup_args") or {},
                    tool_call,
                )
                repo = str(arguments.get("repo") or "").strip()
                owner = str(arguments.get("owner") or account).strip()
                if not repo or not owner or owner.casefold() != account.casefold():
                    logger.warning("Skipping unsafe GitHub cleanup target: %s/%s", owner, repo)
                    continue
                target = (owner, repo)
                if target in deleted:
                    continue
                await asyncio.to_thread(_delete_repository, token, owner, repo, timeout)
                deleted.add(target)
                logger.info("Deleted task repository %s/%s", owner, repo)
            except Exception:  # pylint: disable=broad-exception-caught
                # A cleanup failure (rate limit, missing delete_repo scope, network
                # error, ...) must never abort the run: Self-MCP's task cleanup logs
                # and moves on, and callers here run this inside `finally`, so an
                # uncaught exception would silently drop every remaining queued task.
                logger.exception(
                    "Failed to run GitHub cleanup for tool call %s on %s/%s",
                    tool_call.get("tool_name"),
                    tool_call.get("server"),
                    cleanup.get("cleanup_func"),
                )
