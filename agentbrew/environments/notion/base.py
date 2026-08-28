"""Small MCPMark-compatible base classes for Notion state management."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentbrew.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InitialStateInfo:
    """Information about a prepared initial state."""

    state_id: str
    state_url: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class BaseTask:
    """Base task shape expected by MCPMark state managers."""

    task_instruction_path: Path
    task_verification_path: Path
    service: str
    category_id: str
    task_id: str

    @property
    def name(self) -> str:
        return f"{self.category_id}__{self.task_id}"


class BaseStateManager(ABC):
    """Minimal state manager base copied from MCPMark semantics."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracked_resources: list[dict[str, Any]] = []

    def set_up(self, task: BaseTask) -> bool:
        try:
            logger.info("| Setting up initial state for %s task: %s", self.service_name, task.name)
            initial_state_info = self._create_initial_state(task)
            if not initial_state_info:
                logger.error("| Failed to create initial state for %s", task.name)
                return False
            self._store_initial_state_info(task, initial_state_info)
            logger.info("| ✓ Initial state setup completed for %s", task.name)
            return True
        except Exception as exc:
            logger.error("| Setup failed for %s: %s", task.name, exc)
            return False

    def clean_up(self, task: BaseTask | None = None) -> bool:
        try:
            cleanup_success = True
            if task:
                logger.info("| ○ Cleaning up initial state for %s task: %s", self.service_name, task.name)
                if not self._cleanup_task_initial_state(task):
                    cleanup_success = False
            if not self._cleanup_tracked_resources():
                cleanup_success = False
            return cleanup_success
        except Exception as exc:
            logger.error("Cleanup failed for %s: %s", self.service_name, exc)
            return False

    def track_resource(
        self,
        resource_type: str,
        identifier: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.tracked_resources.append(
            {
                "type": resource_type,
                "id": identifier,
                "created_at": time.time(),
                "metadata": metadata or {},
            }
        )

    def _cleanup_tracked_resources(self) -> bool:
        cleanup_success = True
        for resource in self.tracked_resources:
            try:
                if not self._cleanup_single_resource(resource):
                    cleanup_success = False
            except Exception as exc:
                logger.error("Failed to cleanup resource %s: %s", resource, exc)
                cleanup_success = False
        self.tracked_resources.clear()
        return cleanup_success

    @abstractmethod
    def _create_initial_state(self, task: BaseTask) -> InitialStateInfo | None:
        pass

    @abstractmethod
    def _store_initial_state_info(self, task: BaseTask, state_info: InitialStateInfo) -> None:
        pass

    @abstractmethod
    def _cleanup_task_initial_state(self, task: BaseTask) -> bool:
        pass

    @abstractmethod
    def _cleanup_single_resource(self, resource: dict[str, Any]) -> bool:
        pass


class BaseTaskManager:
    """Minimal task manager base for imported NotionTaskManager."""

    def __init__(
        self,
        tasks_root: Path,
        mcp_service: str | None = None,
        task_class: type | None = None,
        task_organization: str | None = None,
        task_suite: str | None = "standard",
    ):
        self.tasks_root = tasks_root
        self.mcp_service = mcp_service
        self.task_class = task_class or BaseTask
        self.task_organization = task_organization
        self.task_suite = task_suite


class BaseLoginHelper(ABC):
    """Minimal login-helper base."""

    @abstractmethod
    def login(self, **kwargs):
        pass

