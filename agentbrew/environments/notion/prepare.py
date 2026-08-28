"""Notion environment preparation."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from agentbrew.core.context import Context
from agentbrew.core.logger import get_logger
from agentbrew.core.task import TaskSpec

from .state_manager import NotionStateManager
from .task_manager import NotionTask

logger = get_logger(__name__)


def _int_env(context: Context, name: str, default: str) -> int:
    value = _str_env(context, name, default)
    return int(value or default)


def _str_env(context: Context, name: str, default: str) -> str:
    value = context.get_env(name, default).split("#", 1)[0].strip()
    return value.strip().strip('"').strip("'")


async def prepare_notion_task(task: TaskSpec | None, context: Context) -> tuple[NotionStateManager, NotionTask]:
    """Duplicate the source Notion template into the evaluation workspace."""
    if context is None:
        raise ValueError("Context required for Notion setup")

    source_key = context.get_env("SOURCE_NOTION_API_KEY")
    eval_key = context.get_env("EVAL_NOTION_API_KEY")
    if not source_key or not eval_key:
        raise ValueError("SOURCE_NOTION_API_KEY and EVAL_NOTION_API_KEY are required")

    category = (task.category if task else "") or context.get_env("NOTION_CATEGORY", "company_in_a_box")
    task_id = (task.id if task and task.id else None) or f"task_{int(time.time())}"
    dummy_path = Path("agentbrew-generated")

    notion_task = NotionTask(
        task_instruction_path=dummy_path,
        task_verification_path=dummy_path,
        service="notion",
        category_id=category,
        task_id=task_id,
        task_name=f"agentbrew_task_{task_id}",
        original_initial_state_url=None,
        duplicated_initial_state_url=None,
        duplicated_initial_state_id=None,
    )

    state_manager = NotionStateManager(
        source_notion_key=source_key,
        eval_notion_key=eval_key,
        headless=_str_env(
            context,
            "PLAYWRIGHT_HEADLESS",
            context.get_env("NOTION_HEADLESS", "true"),
        ).lower() == "true",
        browser=_str_env(
            context,
            "PLAYWRIGHT_BROWSER",
            context.get_env("NOTION_BROWSER", "chromium"),
        ),
        source_parent_page_title=_str_env(context, "SOURCE_PARENT_PAGE_TITLE", "MCPMark Source Hub"),
        eval_parent_page_title=_str_env(context, "EVAL_PARENT_PAGE_TITLE", "MCPMark Eval Hub"),
        initial_wait_ms=_int_env(context, "NOTION_DUPLICATE_TIMEOUT_MS", "180000"),
        move_wait_ms=_int_env(context, "NOTION_MOVE_TIMEOUT_MS", "60000"),
        database_ready_max_retries=_int_env(context, "NOTION_DATABASE_READY_MAX_RETRIES", "10"),
        database_ready_retry_delay=_int_env(context, "NOTION_DATABASE_READY_RETRY_DELAY", "2"),
        search_ready_max_retries=_int_env(context, "NOTION_SEARCH_READY_MAX_RETRIES", "30"),
        search_ready_retry_delay=_int_env(context, "NOTION_SEARCH_READY_RETRY_DELAY", "3"),
        search_ready_stable_attempts=_int_env(context, "NOTION_SEARCH_READY_STABLE_ATTEMPTS", "2"),
        state_file=_str_env(context, "NOTION_STATE_FILE", "notion_state.json"),
    )

    logger.info("Setting up Notion environment for task: %s", notion_task.name)
    success = await asyncio.to_thread(state_manager.set_up, notion_task)
    if not success:
        raise RuntimeError("Notion setup failed")

    search_wait_seconds = _int_env(context, "NOTION_SEARCH_INDEX_WAIT_SECONDS", "0")
    if search_wait_seconds > 0:
        logger.info(
            "Waiting %d seconds for Notion search indexing",
            search_wait_seconds,
        )
        await asyncio.sleep(search_wait_seconds)

    if notion_task.duplicated_initial_state_url:
        context.env["MCPMARK_NOTION_PAGE_URL"] = notion_task.duplicated_initial_state_url
    if notion_task.duplicated_initial_state_id:
        context.env["MCPMARK_NOTION_PAGE_ID"] = notion_task.duplicated_initial_state_id
    context.env["EVAL_NOTION_API_KEY"] = eval_key

    return state_manager, notion_task
