"""Short-term memory: sliding window of recent messages.

Maintains a fixed-size window of the most recent conversation messages.
Older messages are dropped when the window is full.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Sliding window memory that keeps the N most recent messages.

    When the window is full, oldest messages are dropped first.
    System messages are always preserved.
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._messages: list[BaseMessage] = []

    def add(self, message: BaseMessage) -> None:
        """Add a message to memory."""
        self._messages.append(message)
        self._trim()

    def add_many(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages."""
        self._messages.extend(messages)
        self._trim()

    def get_messages(self) -> list[BaseMessage]:
        """Get all messages in the window."""
        return list(self._messages)

    def get_context(self, max_tokens: int = 4000) -> list[BaseMessage]:
        """Get messages that fit within a token budget.

        Estimates tokens at ~4 chars per token.
        Returns messages from newest to oldest within budget.
        """
        result: list[BaseMessage] = []
        char_budget = max_tokens * 4
        used = 0

        for msg in reversed(self._messages):
            msg_chars = len(str(msg.content)) if msg.content else 0
            if used + msg_chars > char_budget:
                break
            result.insert(0, msg)
            used += msg_chars

        return result

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def count(self) -> int:
        """Return the number of messages."""
        return len(self._messages)

    def _trim(self) -> None:
        """Trim to window size, preserving system messages."""
        if len(self._messages) <= self._window_size:
            return

        # Separate system messages from others
        from langchain_core.messages import SystemMessage
        system_msgs = [m for m in self._messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in self._messages if not isinstance(m, SystemMessage)]

        # Keep the most recent non-system messages
        keep_others = self._window_size - len(system_msgs)
        if keep_others > 0:
            other_msgs = other_msgs[-keep_others:]
        else:
            other_msgs = []

        self._messages = system_msgs + other_msgs
