"""PostgreSQL environment plugin."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from agentbrew.core.context import Context
from agentbrew.core.environment import Environment, EvaluationResult, RuntimeState
from agentbrew.core.task import TaskSpec

from .benchmark.evaluate import run_verifiers
from .state_manager import PostgresConnection, PostgresStateManager
from .task_sampling.generation import (
    MAX_GENERATION_ATTEMPTS,
    FEW_SHOT_ROOT,
    build_prompt,
    build_task_config,
    is_valid_existing_output,
    load_fusion_prompt_specs,
    parse_existing_refs,
    parse_existing_tables,
    parse_table_stats,
    prompt_variants_for_task_type,
    try_parse_json_object,
    validate_generation_payload,
)
from .task_sampling.task_filter import filter_task_tree


class PostgresEnvironment(Environment):
    """MCPMark-compatible PostgreSQL environment using task-local databases."""

    name = "postgres"

    @staticmethod
    def _private_config() -> dict[str, Any]:
        path = Path(__file__).parent / "servers.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("_postgres", {}) or {}

    @staticmethod
    def _set_context_default(context: Context, name: str, value: Any) -> None:
        if value is not None and not context.get_env(name):
            context.env[name] = str(value).strip()

    @staticmethod
    def _resolve_path(path: str) -> Path:
        raw = Path(path).expanduser()
        if raw.is_absolute():
            return raw
        candidates = [
            Path.cwd() / raw,
            Path(__file__).resolve().parents[2] / raw,
            Path(__file__).parent / raw,
        ]
        return next((item for item in candidates if item.exists()), candidates[0])

    def apply_runtime_config(self, context: Context) -> None:
        config = self._private_config()
        connection = config.get("connection", {}) or {}
        assets = config.get("assets", {}) or {}
        timeouts = config.get("timeouts", {}) or {}
        defaults = {
            "POSTGRES_HOST": connection.get("host", "localhost"),
            "POSTGRES_PORT": connection.get("port", 5432),
            "POSTGRES_USERNAME": connection.get("username", "postgres"),
            "POSTGRES_PASSWORD": connection.get("password", "password"),
            "POSTGRES_ADMIN_DATABASE": connection.get("admin_database", "postgres"),
            "POSTGRES_DOCKER_CONTAINER": connection.get("docker_container", "mcpmark-postgres"),
            "POSTGRES_BACKUP_DIR": assets.get("backup_dir"),
            "POSTGRES_SQL_DIR": assets.get("sql_dir"),
            "POSTGRES_METADATA_ROOT": assets.get("metadata_root"),
            "POSTGRES_BENCHMARK_TASK_ROOT": assets.get("benchmark_task_root"),
            "POSTGRES_MAX_META_CHARS": config.get("sampling", {}).get(
                "max_meta_chars", 14000
            ),
            "POSTGRES_RESTORE_TIMEOUT_SECONDS": timeouts.get("restore_seconds", 1800),
            "POSTGRES_VERIFIER_TIMEOUT_SECONDS": timeouts.get("verifier_seconds", 600),
        }
        for name, value in defaults.items():
            self._set_context_default(context, name, value)

    @staticmethod
    def _connection(context: Context) -> PostgresConnection:
        return PostgresConnection(
            host=context.get_env("POSTGRES_HOST", "localhost"),
            port=int(context.get_env("POSTGRES_PORT", "5432")),
            username=context.get_env("POSTGRES_USERNAME", "postgres"),
            password=context.get_env("POSTGRES_PASSWORD", "password"),
            admin_database=context.get_env("POSTGRES_ADMIN_DATABASE", "postgres"),
            docker_container=context.get_env("POSTGRES_DOCKER_CONTAINER", "mcpmark-postgres"),
        )

    def _state_manager(self, context: Context) -> PostgresStateManager:
        backup_dir = context.get_env("POSTGRES_BACKUP_DIR")
        if not backup_dir:
            raise ValueError("POSTGRES_BACKUP_DIR is required")
        return PostgresStateManager(
            self._connection(context),
            backup_dir=self._resolve_path(backup_dir),
            sql_dir=(
                self._resolve_path(context.get_env("POSTGRES_SQL_DIR"))
                if context.get_env("POSTGRES_SQL_DIR")
                else None
            ),
            prepare_root=Path(__file__).parent / "benchmark" / "evaluator",
            restore_timeout_seconds=int(
                context.get_env("POSTGRES_RESTORE_TIMEOUT_SECONDS", "1800")
            ),
        )

    async def prepare(self, task: TaskSpec | None, context: Context) -> RuntimeState:
        if task is None:
            raise ValueError("PostgreSQL task execution requires a task")
        self.apply_runtime_config(context)
        manager = await asyncio.to_thread(self._state_manager, context)
        task_id = task.id or "task"
        task_state = await asyncio.to_thread(
            manager.create_task_state, task.category or "postgres", task_id
        )
        env = {
            "POSTGRES_HOST": manager.connection.host,
            "POSTGRES_PORT": str(manager.connection.port),
            "POSTGRES_USERNAME": manager.connection.username,
            "POSTGRES_PASSWORD": manager.connection.password,
            "POSTGRES_DATABASE": task_state.database_name,
            "POSTGRES_DATABASE_URL": task_state.database_url,
            "POSTGRES_ADDRESS": task_state.database_url,
        }
        context.env.update(env)
        return RuntimeState(
            domain=self.name,
            task_id=task.id,
            resources={"state_manager": manager, "postgres_task": task_state},
            env=env,
        )

    async def cleanup(self, state: RuntimeState, context: Context) -> None:
        manager = state.resources.get("state_manager")
        task_state = state.resources.get("postgres_task")
        if manager and task_state:
            await asyncio.to_thread(manager.cleanup_database, task_state.database_name)
        for name in (
            "POSTGRES_DATABASE",
            "POSTGRES_DATABASE_URL",
            "POSTGRES_ADDRESS",
        ):
            context.env.pop(name, None)

    def mcp_servers(
        self, task: TaskSpec | None, state: RuntimeState
    ) -> list[dict[str, Any]]:
        if task and task.mcp_servers:
            return [server.model_dump(mode="json") for server in task.mcp_servers]
        return [{"name": "postgres-pro"}]

    def work_item_group_key(
        self,
        item: TaskSpec | dict[str, Any],
        *,
        mode: str,
        partition_by: str = "category",
    ) -> str:
        """Keep tasks using one immutable template on the same serial worker."""
        del partition_by
        if mode == "task_sample" and isinstance(item, dict):
            return str(item.get("id") or item.get("db_id") or "postgres")
        if isinstance(item, TaskSpec):
            return item.category or "postgres"
        return str(item.get("category") or item.get("db_id") or "postgres")

    def worker_env(
        self,
        worker_id: int,
        base_context: Context,
        items: list[TaskSpec | dict[str, Any]],
        *,
        mode: str,
    ) -> dict[str, str]:
        del items, mode
        self.apply_runtime_config(base_context)
        return {"WORKER_ID": str(worker_id)}

    async def evaluate(
        self,
        task: TaskSpec,
        result: Any,
        state: RuntimeState,
        context: Context,
    ) -> list[EvaluationResult]:
        del result
        return await run_verifiers(
            task,
            env=state.env,
            timeout_seconds=float(
                context.get_env("POSTGRES_VERIFIER_TIMEOUT_SECONDS", "600")
            ),
        )

    def _metadata_files(self, path: str) -> list[Path]:
        root = self._resolve_path(path)
        if root.is_file():
            return [root]
        return sorted(root.rglob("meta.json"))

    def build_task_sampling_seeds(
        self, task_sampling: Any, context: Context
    ) -> list[dict[str, Any]]:
        items = task_sampling.seeds.get("items") if hasattr(task_sampling, "seeds") else None
        if isinstance(items, list):
            return items
        self.apply_runtime_config(context)
        seed_config = task_sampling.seeds if hasattr(task_sampling, "seeds") else {}
        metadata_path = seed_config.get("path") or context.get_env("POSTGRES_METADATA_ROOT")
        if not metadata_path:
            raise ValueError("PostgreSQL task sampling requires seeds.path or POSTGRES_METADATA_ROOT")
        metadata_files = self._metadata_files(metadata_path)
        specs = load_fusion_prompt_specs()
        task_types = seed_config.get("task_types") or sorted(specs)
        unknown_task_types = sorted(set(task_types) - set(specs))
        if unknown_task_types:
            raise ValueError(f"Unknown PostgreSQL fusion task types: {unknown_task_types}")
        requested_variants = seed_config.get("variants")
        seeds: list[dict[str, Any]] = []
        output_root = Path(str(context.metadata.get("output_root", "")))
        output_task_dir = Path(str(context.metadata.get("output_task_dir", ".")))
        overwrite = bool(context.metadata.get("overwrite", True))
        for meta_path in metadata_files:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            db_id = str(meta.get("db_id") or meta_path.parent.name)
            for task_type_id in task_types:
                variants = prompt_variants_for_task_type(task_type_id)
                if requested_variants:
                    variants = tuple(
                        item for item in variants if item in requested_variants
                    )
                for variant in variants:
                    output_path = (
                        output_root / output_task_dir / task_type_id / str(variant) / f"{db_id}.json"
                    )
                    if not overwrite and is_valid_existing_output(output_path):
                        continue
                    seeds.append(
                        {
                            "id": f"{task_type_id}__{variant}__{db_id}",
                            "category": db_id,
                            "db_id": db_id,
                            "meta_path": str(meta_path),
                            "fusion_task_type": task_type_id,
                            "prompt_variant": str(variant),
                        }
                    )
        limit = getattr(task_sampling, "num_tasks", None)
        return seeds[:limit] if limit else seeds

    async def prepare_task_sampling(
        self, seed: dict[str, Any], context: Context
    ) -> RuntimeState:
        self.apply_runtime_config(context)
        meta_path = self._resolve_path(str(seed["meta_path"]))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["_path"] = str(meta_path)
        metadata["_table_names"] = sorted(parse_existing_tables(metadata))
        metadata["_ref_entities"] = sorted(parse_existing_refs(metadata))
        metadata["_table_stats"] = parse_table_stats(metadata)
        return RuntimeState(
            domain=self.name,
            task_id=str(seed.get("id") or "postgres_sample"),
            resources={"seed": seed, "metadata": metadata},
        )

    async def build_task_sampling_prompt(
        self,
        seed: dict[str, Any],
        state: RuntimeState,
        context: Context,
    ) -> str:
        metadata = state.resources["metadata"]
        max_chars = int(context.get_env("POSTGRES_MAX_META_CHARS", "14000"))
        return build_prompt(
            meta=metadata,
            task_type_id=str(seed["fusion_task_type"]),
            prompt_variant=str(seed["prompt_variant"]),
            max_meta_chars=max_chars,
        )

    def process_task_sampling_result(
        self,
        seed: dict[str, Any],
        response: str,
        state: RuntimeState,
        context: Context,
    ) -> dict[str, Any]:
        del context
        try:
            payload = try_parse_json_object(response)
            payload = validate_generation_payload(
                payload,
                str(seed["fusion_task_type"]),
                state.resources.get("metadata") or {},
            )
            return {
                "accepted": True,
                "question": payload["question"],
                "generation": payload,
                "raw_response": response,
                "judgment": {"accepted": True, "hard_reject_reasons": []},
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return {
                "accepted": False,
                "question": None,
                "raw_response": response,
                "judgment": {
                    "accepted": False,
                    "hard_reject_reasons": [str(exc)],
                },
            }

    def task_sampling_messages(
        self,
        prompt: str,
        seed: dict[str, Any],
        state: RuntimeState,
        context: Context,
    ) -> list[dict[str, str]]:
        del seed, state, context
        return [
            {
                "role": "system",
                "content": "You are a precise PostgreSQL task writer. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ]

    def task_sampling_max_attempts(self) -> int:
        return MAX_GENERATION_ATTEMPTS

    def build_task_sampling_retry_prompt(
        self,
        original_prompt: str,
        processed: dict[str, Any],
    ) -> str:
        reasons = processed.get("judgment", {}).get("hard_reject_reasons", [])
        reason = reasons[0] if reasons else "invalid generation"
        return (
            original_prompt
            + "\n\n"
            + "Your previous output was rejected.\n"
            + f"Reason: {reason}\n"
            + "Return corrected JSON only. Be conservative and low-hallucination."
        )

    def task_sampling_output(
        self, seed: dict[str, Any], processed: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        db_id = str(seed["db_id"])
        task_type_id = str(seed["fusion_task_type"])
        prompt_variant = str(seed["prompt_variant"])
        payload = build_task_config(db_id, str(processed["question"]))
        return f"{task_type_id}/{prompt_variant}/{db_id}.json", payload

    async def after_task_sampling(
        self,
        *,
        output_root: str,
        output_task_dir: str,
        context: Context,
    ) -> None:
        self.apply_runtime_config(context)
        raw_root = (Path(output_root) / output_task_dir).resolve()
        filtered_root = raw_root.parent / f"{raw_root.name}_filtered"
        await asyncio.to_thread(
            filter_task_tree,
            raw_root,
            self._resolve_path(context.get_env("POSTGRES_METADATA_ROOT")),
            FEW_SHOT_ROOT,
            filtered_root,
        )
