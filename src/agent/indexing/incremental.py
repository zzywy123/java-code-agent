"""Incremental indexer for Java code chunks.

Tracks file hashes to detect changes. When a file changes:
1. Remove old chunks from all stores (ChunkStore, VectorStore, BM25Index)
2. Re-slice the file and compute new chunks
3. Generate embeddings for new chunks
4. Add new chunks to all stores

File change detection uses SHA-256 content hashes, not modification times.
Chunk ID = SHA-256(file_path + symbol_signature + content_hash).
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from agent.indexing.bm25_index import BM25Index
from agent.indexing.chunk_store import ChunkStore, compute_chunk_id, compute_file_hash
from agent.indexing.embedding import EmbeddingService
from agent.indexing.java_slicer import JavaSlicer
from agent.indexing.vector_store import VectorStore
from agent.models import CodeChunk, CodeSlice, IndexStats

logger = logging.getLogger(__name__)


class IncrementalIndexer:
    """Manages incremental indexing of Java code.

    Orchestrates the JavaSlicer, ChunkStore, EmbeddingService, VectorStore,
    and BM25Index to keep the index up to date with minimal work.
    """

    def __init__(
        self,
        slicer: JavaSlicer,
        chunk_store: ChunkStore,
        embedding_service: EmbeddingService | None,
        vector_store: VectorStore | None,
        bm25_index: BM25Index,
    ) -> None:
        self._slicer = slicer
        self._chunk_store = chunk_store
        self._embedding = embedding_service
        self._vector_store = vector_store
        self._bm25 = bm25_index
        # file_path → SHA-256 hash of file content
        self._file_hashes: dict[str, str] = {}

    def index_directory(
        self,
        dir_path: Path,
        module: str = "",
        force: bool = False,
    ) -> IndexStats:
        """Index all Java files in a directory.

        Only re-indexes files whose content hash has changed.

        Args:
            dir_path: Root directory to scan
            module: Module name

        Returns:
            IndexStats with counts of files/chunks processed
        """
        start = time.monotonic()
        stats = IndexStats()

        java_files = list(dir_path.rglob("*.java"))
        stats.files_scanned = len(java_files)

        for java_file in java_files:
            try:
                file_stats = self.update_file(java_file, module, force=force)
                stats.files_updated += file_stats.files_updated
                stats.chunks_added += file_stats.chunks_added
                stats.chunks_removed += file_stats.chunks_removed
            except Exception as e:
                stats.errors.append(f"{java_file}: {e}")
                logger.error("Failed to index %s: %s", java_file, e)

        # Detect deleted files
        current_files = {f.as_posix() for f in java_files}
        deleted_files = [
            fp for fp in list(self._file_hashes.keys())
            if fp not in current_files
        ]
        for fp in deleted_files:
            self.remove_file(fp)
            stats.files_removed += 1

        stats.duration_seconds = round(time.monotonic() - start, 3)
        return stats

    def update_file(
        self,
        file_path: Path,
        module: str = "",
        force: bool = False,
    ) -> IndexStats:
        """Update the index for a single file.

        Checks if the file has changed (by content hash) before re-indexing.

        Args:
            file_path: Absolute path to the Java file
            module: Module name

        Returns:
            IndexStats for this file
        """
        stats = IndexStats()

        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            stats.errors.append(f"Cannot read (not UTF-8): {file_path}")
            return stats

        # Compute file hash
        current_hash = compute_file_hash(content)
        rel_path = file_path.as_posix()

        # Check if file changed
        old_hash = self._file_hashes.get(rel_path)
        if not force and old_hash == current_hash:
            return stats  # No change

        # File changed or is new — remove old chunks first
        if old_hash is not None:
            stats.chunks_removed = self._remove_file_chunks(rel_path)

        # Slice the file
        try:
            slices = self._slicer.slice_file(file_path, module)
        except Exception as e:
            stats.errors.append(f"Slice failed: {file_path}: {e}")
            return stats

        # Build chunks
        chunks = self._build_chunks(slices, current_hash)

        # Generate embeddings
        if chunks and self._embedding is not None:
            try:
                self._embed_chunks(chunks)
            except Exception as e:
                stats.errors.append(f"Embedding failed: {file_path}: {e}")
                # Continue without embeddings — BM25 still works
                # Disable vector work for the remainder of this build so a
                # network/model failure is not retried once per Java file.
                self._embedding = None
                self._vector_store = None

        # Add to stores
        if chunks:
            added = self._chunk_store.add(chunks)
            if self._vector_store is not None:
                self._vector_store.add(chunks)
            self._bm25.add(chunks)
            stats.chunks_added = added

        # Update file hash
        self._file_hashes[rel_path] = current_hash
        stats.files_updated = 1

        return stats

    def remove_file(self, file_path: str) -> int:
        """Remove all index data for a file.

        Args:
            file_path: Relative path (as posix string)

        Returns:
            Number of chunks removed
        """
        removed = self._remove_file_chunks(file_path)
        self._file_hashes.pop(file_path, None)
        return removed

    def get_indexed_files(self) -> list[str]:
        """Return list of all indexed file paths."""
        return list(self._file_hashes.keys())

    def get_vector_store(self) -> VectorStore | None:
        """Return the active vector store, or None after a vector failure."""
        return self._vector_store

    def is_indexed(self, file_path: str) -> bool:
        """Check if a file is currently indexed."""
        return file_path in self._file_hashes

    def get_file_hash(self, file_path: str) -> str | None:
        """Get the stored hash for a file."""
        return self._file_hashes.get(file_path)

    def save_state(self, path: Path) -> None:
        """Save the indexer state (file hashes) to disk."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "file_hashes": self._file_hashes,
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load_state(self, path: Path) -> None:
        """Load the indexer state from disk."""
        import json
        if not path.exists():
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._file_hashes = raw.get("file_hashes", {})

    def _remove_file_chunks(self, file_path: str) -> int:
        """Remove all chunks for a file from all stores."""
        # Get chunk IDs before removing
        old_chunks = self._chunk_store.get_by_file(file_path)
        chunk_ids = {c.chunk_id for c in old_chunks}

        # Remove from stores
        self._chunk_store.remove_by_file(file_path)
        if self._vector_store is not None:
            self._vector_store.delete_by_file(file_path)
        self._bm25.remove_by_ids(chunk_ids)

        return len(chunk_ids)

    def _build_chunks(self, slices: list[CodeSlice], file_hash: str) -> list[CodeChunk]:
        """Build CodeChunk objects from CodeSlice objects."""
        chunks = []
        for s in slices:
            chunk_id = compute_chunk_id(s.file_path, s.symbol_signature, s.content)
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                slice=s,
                file_hash=file_hash,
            ))
        return chunks

    def _embed_chunks(self, chunks: list[CodeChunk]) -> None:
        """Generate embeddings for chunks in batch."""
        # Build text to embed for each chunk
        texts = []
        for chunk in chunks:
            text = (
                f"{chunk.slice.package}.{chunk.slice.class_name}.{chunk.slice.method_name} "
                f"{chunk.slice.docstring} {chunk.slice.content[:800]}"
            )
            texts.append(text)

        embeddings = self._embedding.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
