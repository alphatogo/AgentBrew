"""Notion environment plugin."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from notion_client import Client
import yaml

from agentbrew.agents.utils import render_prompt_template
from agentbrew.core.context import Context
from agentbrew.core.environment import Environment, EvaluationResult, RuntimeState
from agentbrew.core.task import TaskSpec

from .cleanup import cleanup_notion_task
from .context_collector import TemplateContextCollector, build_task_conditioned_template_context
from .prepare import prepare_notion_task
from .task_sampling.quality import process_generation


class NotionEnvironment(Environment):
    """MCPMark-backed Notion environment."""

    name = "notion"

    @staticmethod
    def _clean_env_value(value: str) -> str:
        return value.split("#", 1)[0].strip().strip('"').strip("'")

    @staticmethod
    def _private_config() -> dict[str, Any]:
        config_path = Path(__file__).parent / "servers.yaml"
        if not config_path.exists():
            return {}
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return data.get("_notion", {}) or {}

    def _set_context_default(self, context: Context, name: str, value: Any) -> None:
        if value is None or name in context.env:
            return
        context.env[name] = self._clean_env_value(str(value))

    def apply_runtime_config(self, context: Context) -> None:
        """Load Notion-owned runtime values from the environment's server config."""
        config = self._private_config()
        source = config.get("source", {}) or {}
        benchmark = config.get("benchmark", {}) or {}
        sampling = config.get("sampling", {}) or {}
        sampling_workers = sampling.get("workers", {}) if isinstance(sampling, dict) else {}
        sampling_worker = sampling_workers.get("1", {}) or sampling_workers.get(1, {}) or {}
        playwright = config.get("playwright", {}) or {}
        timeouts = config.get("timeouts", {}) or {}

        mode = str(context.metadata.get("mode") or "")
        source_parent_page_title = source.get("parent_page_title")
        if mode in {"task_sample", "trajectory_sample"}:
            source_parent_page_title = (
                source.get("sampling_parent_page_title")
                or source.get("data_sampling_parent_page_title")
                or source_parent_page_title
            )
        elif mode == "benchmark":
            source_parent_page_title = (
                source.get("benchmark_parent_page_title")
                or source_parent_page_title
            )

        source_api_key = source.get("api_key")
        if mode in {"task_sample", "trajectory_sample"}:
            source_api_key = source.get("sampling_api_key") or source_api_key

        self._set_context_default(context, "NOTION_API_KEY", source_api_key)
        self._set_context_default(context, "SOURCE_NOTION_API_KEY", source_api_key)
        self._set_context_default(context, "SOURCE_PARENT_PAGE_TITLE", source_parent_page_title)
        self._set_context_default(context, "NOTION_STATE_FILE", source.get("state_file"))

        eval_api_key = benchmark.get("eval_api_key")
        eval_parent_page_title = benchmark.get("eval_parent_page_title")
        if mode in {"task_sample", "trajectory_sample"}:
            eval_api_key = sampling_worker.get("eval_api_key") or eval_api_key
            eval_parent_page_title = sampling_worker.get("eval_parent_page_title") or eval_parent_page_title

        self._set_context_default(context, "EVAL_NOTION_API_KEY", eval_api_key)
        self._set_context_default(context, "EVAL_PARENT_PAGE_TITLE", eval_parent_page_title)

        self._set_context_default(context, "PLAYWRIGHT_BROWSER", playwright.get("browser"))
        self._set_context_default(context, "PLAYWRIGHT_HEADLESS", playwright.get("headless"))

        self._set_context_default(context, "NOTION_DUPLICATE_TIMEOUT_MS", timeouts.get("duplicate_timeout_ms"))
        self._set_context_default(context, "NOTION_MOVE_TIMEOUT_MS", timeouts.get("move_timeout_ms"))
        self._set_context_default(
            context,
            "NOTION_DATABASE_READY_MAX_RETRIES",
            timeouts.get("database_ready_max_retries"),
        )
        self._set_context_default(
            context,
            "NOTION_DATABASE_READY_RETRY_DELAY",
            timeouts.get("database_ready_retry_delay"),
        )
        self._set_context_default(
            context,
            "NOTION_SEARCH_INDEX_WAIT_SECONDS",
            timeouts.get("search_index_wait_seconds"),
        )
        self._set_context_default(
            context,
            "NOTION_SEARCH_READY_MAX_RETRIES",
            timeouts.get("search_ready_max_retries"),
        )
        self._set_context_default(
            context,
            "NOTION_SEARCH_READY_RETRY_DELAY",
            timeouts.get("search_ready_retry_delay"),
        )
        self._set_context_default(
            context,
            "NOTION_SEARCH_READY_STABLE_ATTEMPTS",
            timeouts.get("search_ready_stable_attempts"),
        )
        self._set_context_default(
            context,
            "NOTION_EVALUATION_SETTLE_SECONDS",
            timeouts.get("evaluation_settle_seconds"),
        )

    async def prepare(self, task: TaskSpec | None, context: Context) -> RuntimeState:
        self.apply_runtime_config(context)
        state_manager, notion_task = await prepare_notion_task(task, context)
        return RuntimeState(
            domain=self.name,
            task_id=task.id if task else None,
            resources={
                "state_manager": state_manager,
                "task": notion_task,
                "duplicated_initial_state_url": notion_task.duplicated_initial_state_url,
                "duplicated_initial_state_id": notion_task.duplicated_initial_state_id,
            },
            env={
                "EVAL_NOTION_API_KEY": context.get_env("EVAL_NOTION_API_KEY"),
                "MCPMARK_NOTION_PAGE_URL": notion_task.duplicated_initial_state_url or "",
                "MCPMARK_NOTION_PAGE_ID": notion_task.duplicated_initial_state_id or "",
            },
        )

    async def cleanup(self, state: RuntimeState, context: Context) -> None:
        await cleanup_notion_task(
            state.resources.get("state_manager"),
            state.resources.get("task"),
            context,
        )

    def mcp_servers(self, task: TaskSpec | None, state: RuntimeState) -> list[dict[str, Any]]:
        if task and task.mcp_servers:
            return [server.model_dump(mode="json") for server in task.mcp_servers]
        return [{"name": "notion_mcpmark"}]

    def work_item_group_key(
        self,
        item: TaskSpec | dict[str, Any],
        *,
        mode: str,
        partition_by: str = "category",
    ) -> str:
        if isinstance(item, TaskSpec):
            return item.category or "default"
        if mode == "task_sample":
            return str(
                item.get("_source_page_id")
                or item.get("source_page_title")
                or item.get("target_page_title")
                or item.get("category")
                or "default"
            )
        return str(
            item.get("category")
            or item.get("target_page_title")
            or item.get(partition_by)
            or "default"
        )

    def worker_env(
        self,
        worker_id: int,
        base_context: Context,
        items: list[TaskSpec | dict[str, Any]],
        *,
        mode: str,
    ) -> dict[str, str]:
        self.apply_runtime_config(base_context)
        if mode == "task_sample":
            return {"WORKER_ID": str(worker_id)}

        private_config = self._private_config()
        sampling_workers = (
            private_config.get("sampling", {}).get("workers", {})
            if isinstance(private_config.get("sampling", {}), dict)
            else {}
        )
        worker_config = sampling_workers.get(str(worker_id), {}) or sampling_workers.get(worker_id, {}) or {}

        api_key_name = f"TRAJECTORY_EVAL_NOTION_API_KEY_{worker_id}"
        page_title_name = f"TRAJECTORY_EVAL_PARENT_PAGE_TITLE_{worker_id}"
        api_key = base_context.get_env(api_key_name) or worker_config.get("eval_api_key")
        page_title = base_context.get_env(page_title_name) or worker_config.get("eval_parent_page_title")

        if worker_id == 1:
            api_key = api_key or base_context.get_env("EVAL_NOTION_API_KEY")
            page_title = page_title or base_context.get_env("EVAL_PARENT_PAGE_TITLE")

        if not api_key:
            raise ValueError(f"Missing {api_key_name} in base env")
        if not page_title:
            raise ValueError(f"Missing {page_title_name} in base env")

        return {
            "WORKER_ID": str(worker_id),
            "EVAL_NOTION_API_KEY": self._clean_env_value(api_key),
            "EVAL_PARENT_PAGE_TITLE": self._clean_env_value(page_title),
        }

    @staticmethod
    def _extract_page_title(page: dict[str, Any]) -> str:
        props = page.get("properties", {}) or {}
        for prop in props.values():
            if prop.get("type") == "title":
                return "".join(
                    item.get("plain_text", "")
                    for item in (prop.get("title") or [])
                ).strip()
        return ""

    def _find_source_hub_id(self, context: Context) -> str:
        client = Client(auth=context.get_env("SOURCE_NOTION_API_KEY"))
        hub_title = context.get_env("SOURCE_PARENT_PAGE_TITLE")
        response = client.search(
            query=hub_title,
            filter={"property": "object", "value": "page"},
            page_size=20,
        )
        for result in response.get("results", []):
            if self._extract_page_title(result) == hub_title:
                return result["id"]
        raise RuntimeError(f"Source hub page '{hub_title}' not found for task sampling")

    def _sampling_source_templates(self, context: Context) -> list[dict[str, str]]:
        cached = context.metadata.get("_notion_sampling_source_templates")
        if isinstance(cached, list) and cached:
            return [
                {"id": str(item["id"]), "title": str(item["title"])}
                for item in cached
                if isinstance(item, dict) and item.get("id") and item.get("title")
            ]

        client = Client(auth=context.get_env("SOURCE_NOTION_API_KEY"))
        hub_id = self._find_source_hub_id(context)
        pages: list[dict[str, str]] = []
        next_cursor = None
        while True:
            kwargs: dict[str, Any] = {"block_id": hub_id, "page_size": 100}
            if next_cursor:
                kwargs["start_cursor"] = next_cursor
            response = client.blocks.children.list(**kwargs)
            for child in response.get("results", []):
                if child.get("type") != "child_page":
                    continue
                title = (child.get("child_page", {}) or {}).get("title", "").strip()
                if title:
                    pages.append({"id": str(child["id"]), "title": title})
            if not response.get("has_more"):
                break
            next_cursor = response.get("next_cursor")

        if not pages:
            raise RuntimeError(
                f"No child pages found under source hub '{context.get_env('SOURCE_PARENT_PAGE_TITLE')}'"
            )
        context.metadata["_notion_sampling_source_templates"] = pages
        return pages

    def _sampling_source_pages(self, context: Context) -> list[str]:
        return [item["title"] for item in self._sampling_source_templates(context)]

    def _resolve_sampling_path(self, path: str) -> Path:
        """Resolve sampling assets relative to cwd, package root, or this environment."""
        raw = Path(path)
        if raw.is_absolute():
            return raw
        candidates = [
            Path.cwd() / raw,
            Path(__file__).resolve().parents[2] / raw,
            Path(__file__).parent / raw,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_few_shot_examples(self, path: str) -> list[dict[str, Any]]:
        """Load few-shot examples from a JSON file or a directory of JSON files."""
        resolved = self._resolve_sampling_path(path)
        files = sorted(resolved.glob("*.json")) if resolved.is_dir() else [resolved]
        examples: list[dict[str, Any]] = []
        for file_path in files:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                examples.extend(item for item in data if isinstance(item, dict))
            elif isinstance(data, dict):
                examples.append(data)
        if not examples:
            raise RuntimeError(f"No few-shot examples found at {resolved}")
        return examples

    def build_task_sampling_seeds(
        self,
        task_sampling: Any,
        context: Context,
    ) -> list[dict[str, Any]]:
        """Combine Notion source pages and few-shot examples into concrete sampling seeds."""
        items = task_sampling.seeds.get("items") if hasattr(task_sampling, "seeds") else None
        if isinstance(items, list):
            return items

        self.apply_runtime_config(context)
        templates = self._sampling_source_templates(context)
        few_shot_path = getattr(task_sampling, "few_shot_path", None)
        if not few_shot_path:
            count = getattr(task_sampling, "num_tasks", None) or 1
            return [{"id": f"sample_{idx:04d}"} for idx in range(1, count + 1)]

        examples = self._load_few_shot_examples(few_shot_path)
        seeds: list[dict[str, Any]] = []
        for template in templates:
            for template_task_index, example in enumerate(examples, start=1):
                idx = len(seeds) + 1
                seeds.append(
                    {
                        "id": f"{self._slugify(template['title'])}__sample_{idx:04d}",
                        "sampling_index": idx,
                        "template_task_index": template_task_index,
                        "category": template["title"],
                        "target_page_title": template["title"],
                        "source_page_title": template["title"],
                        "_source_page_id": template["id"],
                        "task_family": example.get("title"),
                        "task_categories": example.get("capability") or example.get("capability_tags") or [],
                        "benchmark_exemplar": {"examples": [example]},
                        "metadata": {
                            "few_shot_titles": [str(example.get("title"))],
                            "available_source_pages": [item["title"] for item in templates],
                        },
                    },
                )
        limit = getattr(task_sampling, "num_tasks", None)
        return seeds[:limit] if limit else seeds

    @staticmethod
    def _slugify(value: str) -> str:
        """Create stable, readable ids for generated sampling seeds."""
        return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "notion"

    def _assign_sampling_page(self, item: dict[str, Any], context: Context) -> None:
        templates = self._sampling_source_templates(context)
        pages = [template["title"] for template in templates]
        requested = (
            item.get("target_page_title")
            or item.get("source_page_title")
            or item.get("category")
        )
        if requested in pages:
            selected_template = next(template for template in templates if template["title"] == requested)
        else:
            counter = int(context.metadata.get("_notion_sampling_page_counter", 0))
            selected_template = templates[counter % len(templates)]
            context.metadata["_notion_sampling_page_counter"] = counter + 1

        selected = selected_template["title"]
        item["target_page_title"] = selected
        item["source_page_title"] = selected
        item["category"] = selected
        item["_source_page_id"] = selected_template["id"]
        item.setdefault("metadata", {})["available_source_pages"] = pages

    @staticmethod
    def _normalize_task_sampling_exemplar(seed: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        raw = seed.get("benchmark_exemplar") or seed.get("few_shot") or {}
        examples = raw.get("examples") if isinstance(raw, dict) else None
        if not isinstance(examples, list):
            examples = [raw] if isinstance(raw, dict) and raw else []

        normalized_examples: list[dict[str, Any]] = []
        for example in examples:
            if not isinstance(example, dict):
                continue
            item = dict(example)
            if "capability_tags" not in item and "capability" in item:
                item["capability_tags"] = item["capability"]
            if "original_question" not in item and "question" in item:
                item["original_question"] = item["question"]
            normalized_examples.append(item)

        if not normalized_examples:
            return {}, []

        requested_title = seed.get("benchmark_task_title") or seed.get("benchmark_exemplar_title")
        if requested_title:
            for example in normalized_examples:
                if example.get("title") == requested_title or example.get("benchmark_task_title") == requested_title:
                    return example, normalized_examples
        return normalized_examples[0], normalized_examples

    @staticmethod
    def _prompt_few_shot_example(example: dict[str, Any]) -> dict[str, Any]:
        """Keep only the few-shot fields that are shown to the task generator."""
        return {
            "title": example.get("title"),
            "capability": example.get("capability") or example.get("capability_tags") or [],
            "question": example.get("question") or example.get("original_question") or "",
        }

    async def prepare_task_sampling(
        self,
        seed: dict[str, Any],
        context: Context,
    ) -> RuntimeState:
        self.apply_runtime_config(context)
        if not seed.get("_source_page_id"):
            self._assign_sampling_page(seed, context)
        source_page_title = (
            seed.get("target_page_title")
            or seed.get("source_page_title")
            or seed.get("category")
        )
        page_id = str(seed["_source_page_id"])
        cache = context.metadata.setdefault("_notion_task_sampling_contexts", {})
        template_item = cache.get(page_id)
        if template_item is None:
            collector = TemplateContextCollector(
                context.get_env("SOURCE_NOTION_API_KEY"),
                max_rows_per_ds=int(seed.get("max_rows_per_data_source") or 20),
                max_pages=int(seed.get("max_pages") or 500),
            )
            template_item = await asyncio.to_thread(
                collector.collect,
                page_id=page_id,
                title=str(source_page_title),
                parent_title=context.get_env("SOURCE_PARENT_PAGE_TITLE"),
                depth=int(seed.get("source_page_depth") or 1),
            )
            cache[page_id] = template_item
        return RuntimeState(
            domain=self.name,
            task_id=str(seed.get("id")) if seed.get("id") else None,
            resources={
                "seed": seed,
                "template_item": template_item,
                "taskgen_context": template_item["taskgen_context"],
                "benchmark_support": template_item["benchmark_support"],
            },
            env={
                "NOTION_TEMPLATE_PAGE_ID": page_id,
                "NOTION_TEMPLATE_TITLE": str(source_page_title),
            },
        )

    async def before_worker_item(
        self,
        item: TaskSpec | dict[str, Any],
        context: Context,
        *,
        mode: str,
    ) -> None:
        if mode == "task_sample" and isinstance(item, dict):
            self.apply_runtime_config(context)
            self._assign_sampling_page(item, context)

    async def evaluate(
        self,
        task: TaskSpec,
        result: Any,
        state: RuntimeState,
        context: Context,
    ) -> list[EvaluationResult]:
        from .benchmark.evaluate import run_verifier  # pylint: disable=import-outside-toplevel

        settle_seconds = float(
            context.get_env("NOTION_EVALUATION_SETTLE_SECONDS", "0") or 0
        )
        if settle_seconds > 0:
            await asyncio.sleep(settle_seconds)

        eval_api_key = (
            state.env.get("EVAL_NOTION_API_KEY")
            or context.get_env("EVAL_NOTION_API_KEY")
        )
        return await run_verifier(task, eval_api_key)

    async def build_task_sampling_prompt(
        self,
        seed: dict[str, Any],
        state: RuntimeState,
        context: Context,
    ) -> str:
        template_item = state.resources.get("template_item") or {}
        benchmark_exemplar, few_shot_examples = self._normalize_task_sampling_exemplar(seed)
        conditioned_context = build_task_conditioned_template_context(
            template_item,
            benchmark_exemplar,
        )
        payload = {
            "template_context": conditioned_context,
            "benchmark_exemplar": self._prompt_few_shot_example(benchmark_exemplar),
        }
        if len(few_shot_examples) > 1:
            payload["few_shot_examples"] = [
                self._prompt_few_shot_example(example)
                for example in few_shot_examples
            ]
        prompt_path = Path(__file__).parent / "task_sampling" / "prompts" / "prompt.j2"
        return render_prompt_template(
            str(prompt_path),
            QUESTION=json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def process_task_sampling_result(
        self,
        seed: dict[str, Any],
        response: str,
        state: RuntimeState,
        context: Context,
    ) -> dict[str, Any]:
        """Parse and filter a generated Notion task using the old sampler rules."""
        del context
        template_item = state.resources.get("template_item") or {}
        exemplar, _ = self._normalize_task_sampling_exemplar(seed)
        capabilities = exemplar.get("capability") or exemplar.get("capability_tags") or []
        return process_generation(response, list(capabilities), template_item)

    def task_sampling_output(
        self,
        seed: dict[str, Any],
        processed: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Serialize an accepted sample as a trajectory-ready Notion task."""
        template_title = str(seed.get("source_page_title") or seed.get("category") or "notion")
        category_dir = self._slugify(template_title)
        task_index = int(seed.get("sampling_index") or seed.get("template_task_index") or 0)
        filename = f"task_{task_index:04d}.json" if task_index else f"{seed.get('id', 'task')}.json"
        return f"{category_dir}/{filename}", {
            "category": seed.get("category"),
            "question": processed.get("question"),
            "use_specified_server": True,
            "mcp_servers": [{"name": "notion_mcpmark"}],
            "prepares": [
                {
                    "prepare_func": "mcpmark_notion_setup",
                    "prepare_args": {},
                }
            ],
            "cleanups": [
                {
                    "server": "mcpmark",
                    "tool": "",
                    "cleanup_func": "notion_cleanup",
                    "cleanup_args": {},
                }
            ],
            "evaluators": [],
        }
