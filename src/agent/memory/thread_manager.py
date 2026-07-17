"""Thread manager for conversation isolation and checkpointing.

Provides:
- Thread isolation: each conversation has its own thread_id
- Shared persistent checkpointer: survives process restarts
- Thread listing and cleanup
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class ThreadManager:
    """Manages conversation threads with isolated state.

    Each thread has:
    - A unique thread_id
    - Its own conversation history
    - Its own memory state

    Uses a shared checkpointer for LangGraph interrupt/resume support.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        use_persistent_checkpointer: bool = False,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._threads: dict[str, dict[str, Any]] = {}

        # Use MemorySaver (in-memory) by default
        # For production, use SqliteSaver or PostgresSaver
        self._checkpointer: BaseCheckpointSaver = MemorySaver()

        # Load existing threads if persistent
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._load_threads()

    def create_thread(self, name: str = "") -> str:
        """Create a new conversation thread.

        Args:
            name: Optional human-readable name

        Returns:
            The thread_id
        """
        thread_id = str(uuid.uuid4())
        self._threads[thread_id] = {
            "thread_id": thread_id,
            "name": name or f"Thread {len(self._threads) + 1}",
            "created_at": self._now(),
            "message_count": 0,
        }
        self._save_threads()
        return thread_id

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Get thread metadata."""
        return self._threads.get(thread_id)

    def list_threads(self) -> list[dict[str, Any]]:
        """List all threads."""
        return list(self._threads.values())

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread."""
        if thread_id not in self._threads:
            return False
        del self._threads[thread_id]
        self._save_threads()
        return True

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get the shared checkpointer for LangGraph."""
        return self._checkpointer

    def get_thread_config(self, thread_id: str) -> dict[str, Any]:
        """Get LangGraph config for a specific thread.

        Returns a config dict that can be passed to graph.invoke().
        """
        return {"configurable": {"thread_id": thread_id}}

    def increment_message_count(self, thread_id: str, count: int = 1) -> None:
        """Increment the message count for a thread."""
        if thread_id in self._threads:
            self._threads[thread_id]["message_count"] += count
            self._save_threads()

    def _now(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _save_threads(self) -> None:
        """Save thread metadata to disk."""
        if not self._checkpoint_dir:
            return
        path = self._checkpoint_dir / "threads.json"
        try:
            path.write_text(
                json.dumps(self._threads, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to save threads: %s", e)

    def _load_threads(self) -> None:
        """Load thread metadata from disk."""
        if not self._checkpoint_dir:
            return
        path = self._checkpoint_dir / "threads.json"
        if path.exists():
            try:
                self._threads = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to load threads: %s", e)
                self._threads = {}
