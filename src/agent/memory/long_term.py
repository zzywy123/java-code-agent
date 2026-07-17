"""Long-term project memory: persistent key-value store.

Stores project-level knowledge that persists across sessions.
Each memory entry is a key-value dict with metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Persistent long-term memory for project knowledge.

    Stores memories as JSON files in a directory.
    Supports CRUD operations and text search.
    """

    def __init__(self, persist_dir: Path) -> None:
        self._dir = persist_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def store(self, key: str, value: dict[str, Any]) -> None:
        """Store a memory entry.

        Args:
            key: Unique identifier for the memory
            value: Memory content as a dict
        """
        entry = {
            "key": key,
            "content": value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Update existing
        if key in self._index:
            entry["created_at"] = self._index[key].get("created_at", entry["created_at"])

        self._index[key] = entry
        self._save_entry(key, entry)

    def recall(self, key: str) -> dict[str, Any] | None:
        """Recall a memory by key."""
        entry = self._index.get(key)
        if entry:
            return entry.get("content")
        return None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search memories by text similarity.

        Simple keyword matching (not semantic search).
        """
        query_lower = query.lower()
        scored: list[tuple[float, dict[str, Any]]] = []

        for entry in self._index.values():
            content = entry.get("content", {})
            # Search in all string values
            text = " ".join(str(v) for v in content.values() if isinstance(v, str))
            text_lower = text.lower()

            # Identifier words plus CJK bigrams keep Chinese project preferences searchable.
            query_words = self._search_tokens(query_lower)
            text_words = self._search_tokens(text_lower)
            if not query_words:
                continue

            overlap = len(query_words & text_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    @staticmethod
    def _search_tokens(text: str) -> set[str]:
        import re

        tokens = set(re.findall(r"[a-z0-9_$.-]+", text.lower()))
        for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(sequence) == 1:
                tokens.add(sequence)
            else:
                tokens.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
        return tokens

    def list_all(self) -> list[dict[str, Any]]:
        """List all memory entries."""
        return list(self._index.values())

    def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        if key not in self._index:
            return False

        del self._index[key]
        entry_path = self._dir / f"{self._safe_filename(key)}.json"
        if entry_path.exists():
            entry_path.unlink()
        return True

    def count(self) -> int:
        """Return the number of stored memories."""
        return len(self._index)

    def clear(self) -> None:
        """Remove all memories."""
        for entry_path in self._dir.glob("*.json"):
            if entry_path.name == "_index.json":
                continue
            entry_path.unlink()
        self._index.clear()
        self._save_index()

    def _load_index(self) -> None:
        """Load the memory index from disk."""
        index_path = self._dir / "_index.json"
        if index_path.exists():
            try:
                raw = json.loads(index_path.read_text(encoding="utf-8"))
                self._index = raw.get("entries", {})
            except Exception as e:
                logger.warning("Failed to load memory index: %s", e)
                self._index = {}

    def _save_index(self) -> None:
        """Save the memory index to disk."""
        index_path = self._dir / "_index.json"
        try:
            data = {"version": 1, "entries": self._index}
            index_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to save memory index: %s", e)

    def _save_entry(self, key: str, entry: dict[str, Any]) -> None:
        """Save a single entry to disk and update index."""
        entry_path = self._dir / f"{self._safe_filename(key)}.json"
        try:
            entry_path.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to save memory entry '%s': %s", key, e)
        self._save_index()

    @staticmethod
    def _safe_filename(key: str) -> str:
        """Convert a key to a safe filename."""
        import re
        safe = re.sub(r'[^\w\-.]', '_', key)
        return safe[:100]  # Limit length
