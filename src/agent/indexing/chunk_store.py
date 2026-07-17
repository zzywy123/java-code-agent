"""Chunk store for indexed Java code slices.

Stores CodeChunk objects with metadata for retrieval.
Supports CRUD operations, file-based queries, method-based queries,
and JSON persistence.

Chunk ID = SHA-256(file_path + symbol_signature + content_hash).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.models import CodeChunk, CodeSlice


def compute_chunk_id(file_path: str, symbol_signature: str, content: str) -> str:
    """Compute a deterministic chunk ID.

    chunk_id = SHA-256(file_path + symbol_signature + content_hash)
    where content_hash = SHA-256(content).
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    raw = f"{file_path}:{symbol_signature}:{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_file_hash(content: str) -> str:
    """Compute SHA-256 hash of entire file content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ChunkStore:
    """In-memory store for CodeChunk objects.

    Supports:
    - add: Insert or update chunks
    - get_by_id: Retrieve a single chunk by ID
    - get_by_file: Get all chunks from a specific file
    - get_by_method: Find chunks by class name and method name
    - remove_by_file: Remove all chunks from a specific file
    - list_all: List all stored chunks
    - save/load: JSON persistence
    """

    def __init__(self) -> None:
        self._chunks: dict[str, CodeChunk] = {}
        # Secondary index: file_path → set of chunk_ids
        self._file_index: dict[str, set[str]] = {}
        # Secondary index: (class_name, method_name) → set of chunk_ids
        self._method_index: dict[tuple[str, str], set[str]] = {}

    def add(self, chunks: list[CodeChunk]) -> int:
        """Add or update chunks in the store.

        Returns the number of chunks actually added (not updated).
        """
        added = 0
        for chunk in chunks:
            is_new = chunk.chunk_id not in self._chunks
            self._chunks[chunk.chunk_id] = chunk

            # Update file index
            fp = chunk.slice.file_path
            if fp not in self._file_index:
                self._file_index[fp] = set()
            self._file_index[fp].add(chunk.chunk_id)

            # Update method index
            key = (chunk.slice.class_name, chunk.slice.method_name)
            if key not in self._method_index:
                self._method_index[key] = set()
            self._method_index[key].add(chunk.chunk_id)

            if is_new:
                added += 1
        return added

    def get_by_id(self, chunk_id: str) -> CodeChunk | None:
        """Retrieve a single chunk by ID."""
        return self._chunks.get(chunk_id)

    def get_by_file(self, file_path: str) -> list[CodeChunk]:
        """Get all chunks from a specific file."""
        ids = self._file_index.get(file_path, set())
        return [self._chunks[cid] for cid in ids if cid in self._chunks]

    def get_by_method(self, class_name: str, method_name: str) -> list[CodeChunk]:
        """Find chunks by class name and method name."""
        key = (class_name, method_name)
        ids = self._method_index.get(key, set())
        return [self._chunks[cid] for cid in ids if cid in self._chunks]

    def remove_by_file(self, file_path: str) -> int:
        """Remove all chunks from a specific file.

        Returns the number of chunks removed.
        """
        ids = self._file_index.pop(file_path, set())
        removed = 0
        for cid in ids:
            if cid in self._chunks:
                chunk = self._chunks.pop(cid)
                # Clean up method index
                mkey = (chunk.slice.class_name, chunk.slice.method_name)
                if mkey in self._method_index:
                    self._method_index[mkey].discard(cid)
                    if not self._method_index[mkey]:
                        del self._method_index[mkey]
                removed += 1
        return removed

    def remove_by_id(self, chunk_id: str) -> bool:
        """Remove a single chunk by ID."""
        chunk = self._chunks.pop(chunk_id, None)
        if chunk is None:
            return False
        # Clean up file index
        fp = chunk.slice.file_path
        if fp in self._file_index:
            self._file_index[fp].discard(chunk_id)
            if not self._file_index[fp]:
                del self._file_index[fp]
        # Clean up method index
        mkey = (chunk.slice.class_name, chunk.slice.method_name)
        if mkey in self._method_index:
            self._method_index[mkey].discard(chunk_id)
            if not self._method_index[mkey]:
                del self._method_index[mkey]
        return True

    def list_all(self) -> list[CodeChunk]:
        """List all stored chunks."""
        return list(self._chunks.values())

    def list_files(self) -> list[str]:
        """List all indexed file paths."""
        return list(self._file_index.keys())

    def count(self) -> int:
        """Return total number of chunks."""
        return len(self._chunks)

    def file_count(self) -> int:
        """Return number of indexed files."""
        return len(self._file_index)

    def save(self, path: Path) -> None:
        """Persist the store to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks.values()],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Load the store from a JSON file."""
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = raw.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported chunk store version: {version}")

        self._chunks.clear()
        self._file_index.clear()
        self._method_index.clear()

        for chunk_data in raw.get("chunks", []):
            chunk = CodeChunk.model_validate(chunk_data)
            self.add([chunk])

    def clear(self) -> None:
        """Remove all chunks."""
        self._chunks.clear()
        self._file_index.clear()
        self._method_index.clear()
