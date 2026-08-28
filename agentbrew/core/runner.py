"""Unified run orchestration."""

from __future__ import annotations

import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from agentbrew.callbacks.base import BaseCallback
from agentbrew.llms import BaseLLM, ModelManager
from .artifacts import ArtifactStore, TaskRunRecord
from .context import Context
from .environment import Environment, RuntimeState
from .registry import environment_registry
from .run_config import RunConfig
from .task import TaskLoader, TaskSpec
from .workers import TaskJob
from agentbrew.tracing import Tracer


class AgentResponseProtocol(Protocol):
    """Minimal response shape expected from agents."""

    trace_id: str | None

    def get_response_str(self) -> str:
        """Return the final agent response as text."""


class AgentProtocol(Protocol):
    """Minimal agent interface used by the core runner."""

    async def initialize(self, mcp_servers: list[dict[str, Any]] | None = None) -> None:
        """Initialize agent resources."""

    async def execute(self, question: str, **kwargs: Any) -> AgentResponseProtocol | str | dict[str, Any]:
        """Execute one task or sampling prompt."""

    def reset(self) -> None:
        """Reset per-task state."""

    async def change_servers(self, mcp_servers: list[dict[str, Any]]) -> None:
        """Switch MCP servers for the next task."""

    async def cleanup(self) -> None:
        """Clean up agent resources."""


class AgentFactory(Protocol):
    """Build an agent for a run."""

    def __call__(self, config: RunConfig, environment: Environment, context: Context) -> AgentProtocol:
        """Create an agent."""


class RunResult:
    """In-memory result for a run."""

    def __init__(self) -> None:
        self.records: list[TaskRunRecord] = []


class Runner:
    """Shared runner for task sampling, trajectory sampling, and benchmark modes."""

    def __init__(
        self,
        config: RunConfig,
        *,
        agent_factory: AgentFactory | None = None,
        environment: Environment | None = None,
        context: Context | None = None,
        callbacks: list[BaseCallback] | None = None,
    ) -> None:
        self.config = config
        self.config.validate_for_mode()
        self.context = context or Context.from_env_file(config.env_file)
        self.context.env.update(config.env)
        self.context.metadata.update(config.metadata)
        self.context.metadata.setdefault("mode", config.mode)
        self.context.metadata.setdefault("domain", config.domain)
        self.context.metadata.setdefault("output_root", config.output.root)
        self.context.metadata.setdefault(
            "output_task_dir", config.task_sampling.output_task_dir or "."
        )
        self.context.metadata.setdefault("overwrite", config.execution.overwrite)
        self.environment = environment or environment_registry.create(config.domain)
        if agent_factory is None:
            from agentbrew.agents.factory import default_agent_factory

            agent_factory = default_agent_factory
        self.agent_factory = agent_factory
        self.artifacts = ArtifactStore(config.output.root)
        self.callbacks: list[BaseCallback] = callbacks or []
        self._task_sampling_executor: ThreadPoolExecutor | None = None

    async def run(self) -> RunResult:
        """Run according to config.mode."""
        if self.config.output.reports:
            self.artifacts.write_run_manifest(
                {
                    "mode": self.config.mode,
                    "domain": self.config.domain,
                    "config": self.config.model_dump(mode="json"),
                }
            )

        if self.config.mode == "task_sample":
            return await self._run_task_sampling()
        return await self._run_task_execution()

    async def _run_task_sampling(self) -> RunResult:
        seeds = self._load_sampling_seeds()
        buckets = self._partition_work_items(
            seeds,
            self.config.execution.workers,
            mode="task_sample",
        )
        self._task_sampling_executor = ThreadPoolExecutor(
            max_workers=self.config.execution.workers,
            thread_name_prefix="agentbrew-task-sample",
        )
        try:
            worker_records = await asyncio.gather(
                *[
                    self._run_task_sampling_worker(worker_id, bucket)
                    for worker_id, bucket in enumerate(buckets, start=1)
                    if bucket
                ]
            )
        finally:
            self._task_sampling_executor.shutdown(wait=True)
            self._task_sampling_executor = None
        result = RunResult()
        result.records = [record for records in worker_records for record in records]
        await self.environment.after_task_sampling(
            output_root=self.config.output.root,
            output_task_dir=self.config.task_sampling.output_task_dir or ".",
            context=self.context,
        )
        return result

    async def _run_task_execution(self) -> RunResult:
        if self.config.execution.workers > 1:
            return await self._run_task_execution_parallel()

        result = RunResult()
        agent = self._build_agent()
        try:
            for job in self._load_task_jobs():
                record = await self._run_one_task(job.task, agent)
                result.records.append(record)
        finally:
            await agent.cleanup()
        return result

    async def _run_one_task(self, task: TaskSpec, agent: AgentProtocol) -> TaskRunRecord:
        started_at = datetime.now().isoformat()
        record = TaskRunRecord(
            task_id=task.id,
            task_source=task.source_path,
            domain=self.config.domain,
            mode=self.config.mode,
            status="running",
            started_at=started_at,
        )
        if self.callbacks:
            print(f"\n{'#' * 66}", flush=True)
            print(f"Task: {task.id}", flush=True)
            if task.source_path:
                print(f"Source: {task.source_path}", flush=True)
            print(f"{'#' * 66}", flush=True)
            print(f"Question: {task.question}\n", flush=True)
        state: RuntimeState | None = None
        tracer = Tracer(include_prompts=self.config.output.include_prompts)
        try:
            state = await self.environment.prepare(task, self.context)
            response_text, trace_id = await self._execute_agent(
                agent,
                self.environment.mcp_servers(task, state),
                task.question,
                tracer,
                output_format=task.output_format or None,
            )
            record.result = response_text
            record.trace_id = trace_id
            if self.config.evaluation.enabled:
                eval_results = await self.environment.evaluate(task, response_text, state, self.context)
                record.evaluation_results = [item.model_dump(mode="json") for item in eval_results]
            record.status = "succeeded"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            record.status = "failed"
            record.error = f"{exc}\n{traceback.format_exc()}"
        finally:
            if state is not None:
                try:
                    await self.environment.after_execution(
                        task,
                        record.result,
                        state,
                        self.context,
                        tracer,
                    )
                finally:
                    await self.environment.cleanup(state, self.context)
            record.ended_at = datetime.now().isoformat()
            if self.config.output.traces:
                self._write_trajectory(task, record, tracer)
            if self.config.output.task_results:
                self.artifacts.write_task_record(record)
        return record

    async def _run_task_with_timeout(
        self,
        task: TaskSpec,
        agent: AgentProtocol,
        environment: Environment,
        context: Context,
    ) -> TaskRunRecord:
        """Run one task with the configured whole-task timeout."""
        timeout = self.config.execution.task_timeout_seconds
        coroutine = self._run_one_task_with_runtime(
            task,
            agent,
            environment,
            context,
        )
        if not timeout:
            return await coroutine
        try:
            return await asyncio.wait_for(coroutine, timeout=timeout)
        except TimeoutError:
            now = datetime.now().isoformat()
            return TaskRunRecord(
                task_id=task.id,
                task_source=task.source_path,
                domain=self.config.domain,
                mode=self.config.mode,
                status="failed",
                started_at=now,
                ended_at=now,
                error=f"Task timed out after {timeout} seconds",
            )

    async def _run_task_execution_parallel(self) -> RunResult:
        jobs = self._load_task_jobs()
        tasks = [job.task for job in jobs]
        buckets = self._partition_work_items(tasks, self.config.execution.workers, mode=self.config.mode)
        worker_tasks = [
            self._run_task_execution_worker(worker_id, bucket)
            for worker_id, bucket in enumerate(buckets, start=1)
            if bucket
        ]
        worker_records = await asyncio.gather(*worker_tasks)
        result = RunResult()
        for records in worker_records:
            result.records.extend(records)
        return result

    async def _run_task_sampling_worker(
        self,
        worker_id: int,
        seeds: list[dict[str, Any]],
    ) -> list[TaskRunRecord]:
        context, environment = self._build_worker_context_environment(worker_id, seeds, mode="task_sample")
        use_agent = self.config.task_sampling.engine == "agent"
        llm = None if use_agent else self._build_llm(context)
        agent = self.agent_factory(self.config, environment, context) if use_agent else None
        records: list[TaskRunRecord] = []
        try:
            for idx, seed in enumerate(seeds, start=1):
                await environment.before_worker_item(seed, context, mode="task_sample")
                record = await self._run_one_sampling_seed(
                    seed,
                    idx,
                    environment,
                    llm,
                    context,
                    agent=agent,
                )
                records.append(record)
        finally:
            if agent is not None:
                await agent.cleanup()
        return records

    async def _run_task_execution_worker(
        self,
        worker_id: int,
        tasks: list[TaskSpec],
    ) -> list[TaskRunRecord]:
        context, environment, agent = self._build_worker_runtime(worker_id, tasks, mode=self.config.mode)
        records: list[TaskRunRecord] = []
        try:
            for task in tasks:
                await environment.before_worker_item(task, context, mode=self.config.mode)
                record = await self._run_task_with_timeout(
                    task,
                    agent,
                    environment,
                    context,
                )
                records.append(record)
        finally:
            await agent.cleanup()
        return records

    async def _run_one_sampling_seed(
        self,
        seed: dict[str, Any],
        idx: int,
        environment: Environment,
        llm: BaseLLM | None,
        context: Context,
        *,
        agent: AgentProtocol | None = None,
    ) -> TaskRunRecord:
        task_id = str(seed.get("id") or f"sample_{idx:04d}")
        started_at = datetime.now().isoformat()
        state: RuntimeState | None = None
        record = TaskRunRecord(
            task_id=task_id,
            domain=self.config.domain,
            mode=self.config.mode,
            status="running",
            started_at=started_at,
        )
        try:
            state = await environment.prepare_task_sampling(seed, context)
            original_prompt = await environment.build_task_sampling_prompt(seed, state, context)
            active_prompt = original_prompt
            processed: dict[str, Any] = {}
            for attempt in range(1, environment.task_sampling_max_attempts() + 1):
                if agent is None:
                    if llm is None:
                        raise RuntimeError("Task sampling requires an LLM or agent")
                    messages = environment.task_sampling_messages(
                        active_prompt, seed, state, context
                    )
                    response_text = await self._call_task_sampling_model(messages, llm)
                else:
                    response_text, _ = await self._execute_agent(
                        agent,
                        environment.mcp_servers(None, state),
                        active_prompt,
                        Tracer(),
                    )
                processed = environment.process_task_sampling_result(
                    seed,
                    response_text,
                    state,
                    context,
                )
                if processed.get("accepted"):
                    break
                if attempt < environment.task_sampling_max_attempts():
                    retry_prompt = environment.build_task_sampling_retry_prompt(
                        original_prompt, processed
                    )
                    if retry_prompt is None:
                        break
                    active_prompt = retry_prompt
            record.result = processed
            if processed.get("accepted"):
                record.status = "succeeded"
                self._write_generated_task(environment, seed, processed)
            else:
                record.status = "rejected"
                reasons = processed.get("judgment", {}).get("hard_reject_reasons", [])
                record.error = "; ".join(str(reason) for reason in reasons)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            record.status = "failed"
            record.error = f"{exc}\n{traceback.format_exc()}"
        finally:
            if state is not None:
                await environment.cleanup(state, context)
            record.ended_at = datetime.now().isoformat()
            if self.config.output.task_results:
                self.artifacts.write_task_record(record)
        return record

    async def _call_task_sampling_model(
        self, messages: list[dict[str, str]], llm: BaseLLM
    ) -> str:
        """Generate one task with the configured LLM messages."""
        response = await llm.generate_async(
            messages=messages,
            callbacks=self.callbacks,
            _executor=self._task_sampling_executor,
        )
        if response is None:
            raise RuntimeError("Task sampling LLM returned no response")
        return str(response)

    async def _execute_agent(
        self,
        agent: AgentProtocol,
        mcp_servers: list[dict[str, Any]],
        question: str,
        tracer: Tracer,
        *,
        output_format: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        """Run one ReAct task, restarting MCP servers between worker items."""
        if getattr(agent, "_initialized", False):
            await agent.change_servers(mcp_servers)
        else:
            await agent.initialize(mcp_servers=mcp_servers)
        agent.reset()
        response = await agent.execute(
            question,
            output_format=output_format,
            callbacks=self.callbacks,
            tracer=tracer,
        )
        return self._normalize_response(response)

    async def _run_one_task_with_runtime(
        self,
        task: TaskSpec,
        agent: AgentProtocol,
        environment: Environment,
        context: Context,
    ) -> TaskRunRecord:
        started_at = datetime.now().isoformat()
        record = TaskRunRecord(
            task_id=task.id,
            task_source=task.source_path,
            domain=self.config.domain,
            mode=self.config.mode,
            status="running",
            started_at=started_at,
        )
        state: RuntimeState | None = None
        tracer = Tracer(include_prompts=self.config.output.include_prompts)
        try:
            state = await environment.prepare(task, context)
            response_text, trace_id = await self._execute_agent(
                agent,
                environment.mcp_servers(task, state),
                task.question,
                tracer,
                output_format=task.output_format or None,
            )
            record.result = response_text
            record.trace_id = trace_id
            if self.config.evaluation.enabled:
                eval_results = await environment.evaluate(task, response_text, state, context)
                record.evaluation_results = [item.model_dump(mode="json") for item in eval_results]
            record.status = "succeeded"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            record.status = "failed"
            record.error = f"{exc}\n{traceback.format_exc()}"
        finally:
            if state is not None:
                try:
                    await environment.after_execution(
                        task,
                        record.result,
                        state,
                        context,
                        tracer,
                    )
                finally:
                    await environment.cleanup(state, context)
            record.ended_at = datetime.now().isoformat()
            if self.config.output.traces:
                self._write_trajectory(task, record, tracer)
            if self.config.output.task_results:
                self.artifacts.write_task_record(record)
        return record

    def _build_worker_runtime(
        self,
        worker_id: int,
        items: list[TaskSpec | dict[str, Any]],
        *,
        mode: str,
    ) -> tuple[Context, Environment, AgentProtocol]:
        context, environment = self._build_worker_context_environment(worker_id, items, mode=mode)
        agent = self.agent_factory(self.config, environment, context)
        return context, environment, agent

    def _build_worker_context_environment(
        self,
        worker_id: int,
        items: list[TaskSpec | dict[str, Any]],
        *,
        mode: str,
    ) -> tuple[Context, Environment]:
        """Build worker-local context and environment without constructing an agent."""
        context = self.context.model_copy(deep=True)
        context.metadata["mode"] = mode
        environment = environment_registry.create(self.config.domain)
        context.env.update(environment.worker_env(worker_id, context, items, mode=mode))
        return context, environment

    def _partition_work_items(
        self,
        items: list[TaskSpec | dict[str, Any]],
        worker_count: int,
        *,
        mode: str,
    ) -> list[list[Any]]:
        if worker_count <= 0:
            raise ValueError("workers must be positive")
        worker_count = min(worker_count, len(items)) if items else 1
        groups: dict[str, list[Any]] = {}
        for item in items:
            key = self.environment.work_item_group_key(
                item,
                mode=mode,
                partition_by=self.config.execution.partition_by,
            )
            groups.setdefault(key, []).append(item)

        buckets: list[list[Any]] = [[] for _ in range(worker_count)]
        loads = [0 for _ in range(worker_count)]
        for key, group_items in sorted(groups.items(), key=lambda entry: (-len(entry[1]), entry[0])):
            idx = min(range(worker_count), key=lambda i: (loads[i], i))
            buckets[idx].extend(group_items)
            loads[idx] += len(group_items)
        return buckets

    def _build_agent(self) -> AgentProtocol:
        return self.agent_factory(self.config, self.environment, self.context)

    def _build_llm(self, context: Context) -> BaseLLM:
        """Build the configured model through the shared LLM registry."""
        llm = ModelManager().build_model(self.config.llm.type, self.config.llm.config)
        llm.set_context(context)
        return llm

    def _load_task_jobs(self) -> list[TaskJob]:
        source = self.config.task_source
        if source.type == "file":
            if not source.path:
                raise ValueError("file task_source requires path")
            return [TaskJob(TaskLoader.load_file(source.path))]
        if source.type == "directory":
            if not source.path:
                raise ValueError("directory task_source requires path")
            return [TaskJob(task) for task in TaskLoader.load_directory(source.path, source.glob)]
        if source.type == "inline":
            return [TaskJob(task) for task in TaskLoader.load_inline(source.tasks)]
        raise ValueError(f"Unsupported task_source type: {source.type}")

    def _load_sampling_seeds(self) -> list[dict[str, Any]]:
        return self.environment.build_task_sampling_seeds(self.config.task_sampling, self.context)

    def _write_generated_task(
        self,
        environment: Environment,
        seed: dict[str, Any],
        processed: dict[str, Any],
    ) -> None:
        output_dir = self.config.task_sampling.output_task_dir
        if not output_dir:
            return
        relative_path, payload = environment.task_sampling_output(seed, processed)
        self.artifacts.write_json(f"{output_dir}/{relative_path}", payload)

    def _write_trajectory(
        self,
        task: TaskSpec,
        record: TaskRunRecord,
        tracer: Tracer,
    ) -> None:
        """Save one execution trajectory per task."""
        task_id = task.id or Path(task.source_path or "task").stem
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in task_id)
        tracer.write_trajectory(
            self.artifacts.root / "trajectories" / f"{safe_id}.json",
            task_id=task_id,
            question=task.question,
            final_answer=str(record.result or record.error),
        )

    @staticmethod
    def _normalize_response(response: AgentResponseProtocol | str | dict[str, Any]) -> tuple[str, str | None]:
        if isinstance(response, str):
            return response, None
        if isinstance(response, dict):
            return str(response.get("response") or response.get("output") or response), response.get("trace_id")
        trace_id = getattr(response, "trace_id", None)
        if hasattr(response, "get_response_str"):
            return response.get_response_str(), trace_id
        return str(response), trace_id
