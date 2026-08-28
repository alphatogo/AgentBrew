"""Environment plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from .context import Context
from .task import TaskSpec


class RuntimeState(BaseModel):
    """Serializable state returned by an environment prepare step."""

    domain: str
    task_id: str | None = None
    resources: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Minimal evaluation result used by core."""

    passed: bool
    verifier: str = ""
    reason: str = ""
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Environment(ABC):
    """Domain plugin boundary."""

    name: str

    @abstractmethod
    async def prepare(self, task: TaskSpec | None, context: Context) -> RuntimeState:
        """Prepare the environment for a task or sampling seed."""

    async def prepare_task_sampling(
        self,
        seed: dict[str, Any],
        context: Context,
    ) -> RuntimeState:
        """Prepare deterministic context for task generation."""
        return RuntimeState(
            domain=self.name,
            task_id=str(seed.get("id")) if seed.get("id") else None,
            resources={"seed": seed},
        )

    @abstractmethod
    async def cleanup(self, state: RuntimeState, context: Context) -> None:
        """Clean up resources created during prepare/execution."""

    async def after_execution(
        self,
        task: TaskSpec,
        result: Any,
        state: RuntimeState,
        context: Context,
        tracer: Any,
    ) -> None:
        """Capture domain-specific execution state before evaluation and cleanup."""
        return None

    @abstractmethod
    def mcp_servers(self, task: TaskSpec | None, state: RuntimeState) -> list[dict[str, Any]]:
        """Return MCP servers available to the agent."""

    async def evaluate(
        self,
        task: TaskSpec,
        result: Any,
        state: RuntimeState,
        context: Context,
    ) -> list[EvaluationResult]:
        """Evaluate a completed task. Domains may override this."""
        return []

    async def build_task_sampling_prompt(
        self,
        seed: dict[str, Any],
        state: RuntimeState,
        context: Context,
    ) -> str:
        """Build a task sampling prompt. Domains may override this."""
        raise NotImplementedError(f"{self.name} does not implement task sampling")

    def process_task_sampling_result(
        self,
        seed: dict[str, Any],
        response: str,
        state: RuntimeState,
        context: Context,
    ) -> dict[str, Any]:
        """Parse and validate a generated task. Domains may override this."""
        return {
            "accepted": True,
            "question": response,
            "raw_response": response,
            "judgment": {"accepted": True},
        }

    def task_sampling_messages(
        self,
        prompt: str,
        seed: dict[str, Any],
        state: RuntimeState,
        context: Context,
    ) -> list[dict[str, str]]:
        """Build model messages for one task-generation attempt."""
        del seed, state, context
        return [{"role": "user", "content": prompt}]

    def task_sampling_max_attempts(self) -> int:
        """Return the number of semantic generation attempts."""
        return 1

    def build_task_sampling_retry_prompt(
        self,
        original_prompt: str,
        processed: dict[str, Any],
    ) -> str | None:
        """Return a corrected prompt after validation rejection, if supported."""
        del original_prompt, processed
        return None

    async def after_task_sampling(
        self,
        *,
        output_root: str,
        output_task_dir: str,
        context: Context,
    ) -> None:
        """Run optional domain-specific post-generation processing."""
        del output_root, output_task_dir, context
        return None

    def task_sampling_output(
        self,
        seed: dict[str, Any],
        processed: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Return the relative path and serialized accepted task."""
        task_id = str(seed.get("id") or "sample")
        return f"{task_id}.json", {
            "category": seed.get("category"),
            "question": processed.get("question"),
            "evaluators": [],
        }

    def build_task_sampling_seeds(
        self,
        task_sampling: Any,
        context: Context,
    ) -> list[dict[str, Any]]:
        """Build concrete sampling seeds from domain-specific sampling controls."""
        items = task_sampling.seeds.get("items") if hasattr(task_sampling, "seeds") else None
        if isinstance(items, list):
            return items
        count = getattr(task_sampling, "num_tasks", None) or 1
        return [{"id": f"sample_{idx:04d}"} for idx in range(1, count + 1)]

    def work_item_group_key(
        self,
        item: TaskSpec | dict[str, Any],
        *,
        mode: str,
        partition_by: str = "category",
    ) -> str:
        """Return the partition key used to place work on parallel workers."""
        if isinstance(item, TaskSpec):
            return str(getattr(item, partition_by, None) or item.category or "default")
        return str(item.get(partition_by) or item.get("category") or "default")

    def worker_env(
        self,
        worker_id: int,
        base_context: Context,
        items: list[TaskSpec | dict[str, Any]],
        *,
        mode: str,
    ) -> dict[str, str]:
        """Return env overrides for a worker. Domains may override this."""
        return {"WORKER_ID": str(worker_id)}

    async def before_worker_item(
        self,
        item: TaskSpec | dict[str, Any],
        context: Context,
        *,
        mode: str,
    ) -> None:
        """Hook called before each item in a worker. Domains may override this."""
        return None
