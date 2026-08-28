"""Notion environment cleanup."""

from __future__ import annotations

import asyncio

from agentbrew.core.context import Context
from agentbrew.core.logger import get_logger

from .state_manager import NotionStateManager
from .task_manager import NotionTask

logger = get_logger(__name__)


async def cleanup_notion_task(
    state_manager: NotionStateManager | None,
    task: NotionTask | None,
    context: Context,
) -> bool:
    """Clean duplicated Notion page and clear transient context values."""
    if not state_manager or not task:
        return True

    logger.info("Cleaning up Notion environment for task: %s", task.name)
    success = await asyncio.to_thread(state_manager.clean_up, task)
    context.env.pop("MCPMARK_NOTION_PAGE_URL", None)
    context.env.pop("MCPMARK_NOTION_PAGE_ID", None)
    return success

