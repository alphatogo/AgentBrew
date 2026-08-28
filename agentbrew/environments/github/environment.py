"""GitHub environment plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agentbrew.core.context import Context
from agentbrew.core.environment import Environment, RuntimeState
from agentbrew.core.task import TaskSpec

from .accounts import DEFAULT_SERVER_CONFIG
from .cleanup import cleanup_github_task
from .benchmark.evaluate import run_evaluators
from .prepare import prepare_github_task


READ_ONLY_TOOLS = [
    "get_file_contents",
    "get_label",
    "get_latest_release",
    "get_me",
    "get_release_by_tag",
    "get_tag",
    "issue_read",
    "list_branches",
    "list_commits",
    "list_issue_types",
    "list_issues",
    "list_pull_requests",
    "list_releases",
    "list_tags",
    "pull_request_read",
]

OWNERSHIP_CONSTRAINT = (
    "IMPORTANT OWNERSHIP RULE: Any write operations (creating/updating files, "
    "opening PRs, submitting issues, pushing commits, etc.) MUST be performed "
    "only on repositories that the user owns — either a freshly created repository "
    "or a fork of an external one. Reading or inspecting files from external "
    "repositories is allowed. Never instruct the user to open a PR or submit an "
    "issue on a repository they do not own."
)


class GitHubEnvironment(Environment):
    """GitHub MCP environment with one account assigned to each worker."""

    name = "github"

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(
            char.lower() if char.isalnum() else "_"
            for char in value
        ).strip("_") or "github"

    @staticmethod
    def _private_config() -> dict[str, Any]:
        data = yaml.safe_load(DEFAULT_SERVER_CONFIG.read_text(encoding="utf-8")) or {}
        return data.get("_github", {}) or {}

    @staticmethod
    def _set_context_default(context: Context, name: str, value: Any) -> None:
        if value is not None and not context.get_env(name):
            context.env[name] = str(value).strip()

    def apply_runtime_config(self, context: Context) -> None:
        """Load the default account before worker-specific values are applied."""
        config = self._private_config()
        mode = str(context.metadata.get("mode") or "")
        account_config = config.get("benchmark", {}) or {}
        if mode in {"task_sample", "trajectory_sample"}:
            workers = (config.get("sampling", {}) or {}).get("workers", {}) or {}
            account_config = workers.get("1", {}) or workers.get(1, {}) or account_config
        self._set_context_default(
            context,
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            account_config.get("token"),
        )
        self._set_context_default(
            context,
            "GITHUB_PERSONAL_ACCOUNT_NAME",
            account_config.get("account"),
        )

    async def prepare(self, task: TaskSpec | None, context: Context) -> RuntimeState:
        self.apply_runtime_config(context)
        return await prepare_github_task(task, context)

    async def prepare_task_sampling(
        self,
        seed: dict[str, Any],
        context: Context,
    ) -> RuntimeState:
        state = await self.prepare(None, context)
        state.task_id = str(seed.get("id")) if seed.get("id") else None
        state.resources.update({"seed": seed, "task_sampling": True})
        return state

    async def cleanup(self, state: RuntimeState, context: Context) -> None:
        await cleanup_github_task(
            state.resources.get("task"),
            state,
            context,
        )

    async def after_execution(
        self,
        task: TaskSpec,
        result: Any,
        state: RuntimeState,
        context: Context,
        tracer: Any,
    ) -> None:
        """Retain successful GitHub tool calls for Self-MCP-style cleanup."""
        del result, context
        state.resources["task"] = task
        state.resources["tool_calls"] = [
            event
            for event in getattr(tracer, "trajectory", [])
            if event.get("state") == "tool" and event.get("server") == "github"
        ]

    async def evaluate(
        self,
        task: TaskSpec,
        result: Any,
        state: RuntimeState,
        context: Context,
    ) -> list[Any]:
        del state
        return await run_evaluators(task, str(result), context)

    def mcp_servers(self, task: TaskSpec | None, state: RuntimeState) -> list[dict[str, Any]]:
        if task and task.mcp_servers:
            return [server.model_dump(mode="json") for server in task.mcp_servers]
        server: dict[str, Any] = {"name": "github"}
        if state.resources.get("task_sampling"):
            server["tools"] = READ_ONLY_TOOLS
        return [server]

    def work_item_group_key(
        self,
        item: TaskSpec | dict[str, Any],
        *,
        mode: str,
        partition_by: str = "category",
    ) -> str:
        """Spread individual items across account workers."""
        del mode, partition_by
        if isinstance(item, TaskSpec):
            return item.id
        return str(item.get("id") or item.get("target_repository") or id(item))

    def worker_env(
        self,
        worker_id: int,
        base_context: Context,
        items: list[TaskSpec | dict[str, Any]],
        *,
        mode: str,
    ) -> dict[str, str]:
        """Bind one GitHub account to a worker for its full serial queue."""
        del items
        self.apply_runtime_config(base_context)
        if mode == "benchmark" and worker_id == 1:
            return {
                "WORKER_ID": "1",
                "GITHUB_PERSONAL_ACCESS_TOKEN": base_context.get_env(
                    "GITHUB_PERSONAL_ACCESS_TOKEN"
                ),
                "GITHUB_PERSONAL_ACCOUNT_NAME": base_context.get_env(
                    "GITHUB_PERSONAL_ACCOUNT_NAME"
                ),
            }

        workers = (
            (self._private_config().get("sampling", {}) or {}).get("workers", {}) or {}
        )
        worker = workers.get(str(worker_id), {}) or workers.get(worker_id, {}) or {}
        token = worker.get("token")
        account = worker.get("account")
        if not token or not account:
            raise ValueError(f"No GitHub account configured for worker {worker_id}")
        return {
            "WORKER_ID": str(worker_id),
            "GITHUB_PERSONAL_ACCESS_TOKEN": str(token).strip(),
            "GITHUB_PERSONAL_ACCOUNT_NAME": str(account).strip(),
        }

    @staticmethod
    def _resolve_path(path: str) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            return raw
        candidates = [
            Path.cwd() / raw,
            Path(__file__).resolve().parents[2] / raw,
            Path(__file__).parent / raw,
        ]
        return next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    def _load_data_files(self, path: str) -> list[dict[str, Any]]:
        resolved = self._resolve_path(path)
        files = sorted(resolved.glob("*.json")) if resolved.is_dir() else [resolved]
        records: list[dict[str, Any]] = []
        for file_path in files:
            if file_path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            else:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(item for item in data if isinstance(item, dict))
            elif isinstance(data, dict):
                records.append(data)
        return records

    def build_task_sampling_seeds(
        self,
        task_sampling: Any,
        context: Context,
    ) -> list[dict[str, Any]]:
        """Attach configured few-shot examples to concrete GitHub sampling seeds."""
        del context
        seed_config = task_sampling.seeds if hasattr(task_sampling, "seeds") else {}
        items = seed_config.get("items")
        if not isinstance(items, list):
            seed_path = seed_config.get("path")
            items = self._load_data_files(seed_path) if seed_path else []

        count = getattr(task_sampling, "num_tasks", None) or len(items) or 1
        if not items:
            items = [{"id": f"github_sample_{idx:04d}"} for idx in range(1, count + 1)]

        examples: list[dict[str, Any]] = []
        few_shot_path = getattr(task_sampling, "few_shot_path", None)
        if few_shot_path:
            examples = self._load_data_files(few_shot_path)
        seeds: list[dict[str, Any]] = []
        for idx in range(count):
            seed = dict(items[idx % len(items)])
            source_id = str(seed.get("id") or seed.get("target_repository") or "github")
            seed["source_id"] = source_id
            seed["id"] = f"{self._slugify(source_id)}__sample_{idx + 1:04d}"
            if examples:
                selected_index = idx % len(examples)
                selected = examples[selected_index]
                seed["few_shot_examples"] = [selected]
                seed["benchmark_example"] = (
                    selected.get("question")
                    or selected.get("benchmark_example")
                    or seed.get("benchmark_example")
                )
                seed["workflow_pattern"] = (
                    seed.get("workflow_pattern")
                    or selected.get("workflow_pattern")
                )
            seeds.append(seed)
        return seeds

    async def build_task_sampling_prompt(
        self,
        seed: dict[str, Any],
        state: RuntimeState,
        context: Context,
    ) -> str:
        """Build the exact question format used by Self-MCP task generation."""
        del state, context
        if seed.get("question"):
            return str(seed["question"])

        workflow_pattern = seed.get("workflow_pattern")
        benchmark_example = seed.get("benchmark_example")
        target_repository = seed.get("target_repository")
        if not workflow_pattern or not benchmark_example or not target_repository:
            raise ValueError(
                "GitHub task sampling requires question or workflow_pattern, "
                "benchmark_example, and target_repository"
            )
        ownership_rule = seed.get("ownership_rule") or OWNERSHIP_CONSTRAINT
        return (
            "─── Workflow Pattern ───\n"
            f"{workflow_pattern}\n\n"
            "─── Benchmark Example (follow this pattern with different content) ───\n"
            f"{benchmark_example}\n\n"
            "─── Target Library ───\n"
            f"{target_repository}\n\n"
            "─── Ownership Rule ───\n"
            f"{ownership_rule}"
        )

    def process_task_sampling_result(
        self,
        seed: dict[str, Any],
        response: str,
        state: RuntimeState,
        context: Context,
    ) -> dict[str, Any]:
        """Normalize the agent's final task response."""
        del seed, state, context
        text = response.strip()
        try:
            parsed = json.loads(text)
            question = str(parsed.get("answer") or parsed.get("question") or "").strip()
        except json.JSONDecodeError:
            question = text
        accepted = bool(question)
        return {
            "accepted": accepted,
            "question": question,
            "raw_response": response,
            "judgment": {
                "accepted": accepted,
                "hard_reject_reasons": [] if accepted else ["empty generated task"],
            },
        }

    def task_sampling_output(
        self,
        seed: dict[str, Any],
        processed: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        task_id = str(seed.get("id") or "github_task")
        cleanups = seed.get("cleanups") or [
            {
                "server": "github",
                "tool": "create_repository",
                "cleanup_func": "delete_repository",
                "cleanup_args": {"repo": "$name"},
            },
            {
                "server": "github",
                "tool": "fork_repository",
                "cleanup_func": "delete_repository",
                "cleanup_args": {"repo": "$repo"},
            },
        ]
        return f"{task_id}.json", {
            "category": seed.get("category", "repository_management"),
            "question": processed.get("question"),
            "use_specified_server": True,
            "mcp_servers": [{"name": "github"}],
            "evaluators": [],
            "cleanups": cleanups,
        }
