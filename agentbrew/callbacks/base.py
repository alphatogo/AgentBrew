"""Lightweight callback compatibility layer."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    LOG = "log"
    STATUS = "status"
    EVENT = "event"
    RESPONSE = "response"
    ERROR = "error"
    PROGRESS = "progress"


class Status(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Event(str, Enum):
    BEFORE_CALL = "before_call"
    AFTER_CALL = "after_call"
    START = "start"
    END = "end"


class CallbackMessage(BaseModel):
    source: str
    type: MessageType
    data: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_id: str = ""


class BaseCallback:
    """Base callback interface."""

    def on_message(self, message: CallbackMessage) -> None:
        """Handle a callback message."""


def _as_list(callbacks: BaseCallback | list[BaseCallback] | None) -> list[BaseCallback]:
    if callbacks is None:
        return []
    if isinstance(callbacks, list):
        return callbacks
    return [callbacks]


def send_message(callbacks: BaseCallback | list[BaseCallback] | None, message: CallbackMessage) -> None:
    for callback in _as_list(callbacks):
        callback.on_message(message)


async def send_message_async(callbacks: BaseCallback | list[BaseCallback] | None, message: CallbackMessage) -> None:
    send_message(callbacks, message)

