"""VerbosePrintCallback: prints agent loop output to stdout."""
from __future__ import annotations

import sys

from .base import BaseCallback, CallbackMessage, MessageType


class VerbosePrintCallback(BaseCallback):
    """Print plain-text agent iteration logs to stdout."""

    def on_message(self, message: CallbackMessage) -> None:
        if message.type == MessageType.LOG:
            plain_text = message.metadata.get("data")
            if plain_text:
                sys.stdout.write(plain_text)
                sys.stdout.flush()
