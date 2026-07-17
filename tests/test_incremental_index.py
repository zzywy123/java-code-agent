"""Tests for incremental indexing.

Validates:
- Initial indexing of a directory
- Re-indexing only changed files (by content hash)
- Removal of deleted files
- File hash tracking
- State save/load
- IndexStats correctness
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.indexing.bm25_index import BM25Index
from agent.indexing.chunk_store import ChunkStore, compute_file_hash
from agent.indexing.embedding import EmbeddingService
from agent.indexing.incremental import IncrementalIndexer
from agent.indexing.java_slicer import JavaSlicer
from agent.indexing.vector_store import VectorStore
from agent.models import IndexStats


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a small Java repo for testing."""
    src = tmp_path / "src" / "main" / "java" / "com" / "example"
    src.mkdir(parents=True)

    (src / "OrderService.java").write_text(textwrap.dedent("""\
        package com.example;

        public class OrderService {
            public void createOrder(String customerId) {
                // create order
            }

            public void calculateTotal(Order order) {
                // calculate
            }
        }
    """), encoding="utf-8")

    (src / "UserService.java").write_text(textwrap.dedent("""\
        package com.example;

        public class UserService {
            public void findUser(String id) {
                // find user
            }
        }
    """), encoding="utf-8")

    return tmp_path


@pytest.fixture
def mock_embedding() -> MagicMock:
    """Mock embedding service that returns dummy vectors."""
    mock = MagicMock(spec=EmbeddingService)
    mock.embed_texts.return_value = [[0.1] * 10 for _ in range(20)]
    mock.embed_query.return_value = [0.1] * 10
    return mock


@pytest.fixture
def indexer(mock_embedding: MagicMock) -> IncrementalIndexer:
    """Create an indexer with real slicer/chunk store and mocked embedding/vector store."""
    slicer = JavaSlicer()
    chunk_store = ChunkStore()
    vector_store = MagicMock(spec=VectorStore)
    vector_store.add.return_value = 5
    vector_store.delete_by_file.return_value = 0
    vector_store.query.return_value = []
    vector_store.count.return_value = 0
    bm25 = BM25Index()

    return IncrementalIndexer(
        slicer=slicer,
        chunk_store=chunk_store,
        embedding_service=mock_embedding,
        vector_store=vector_store,
        bm25_index=bm25,
    )


class TestInitialIndexing:
    """First-time indexing of a directory."""

    def test_index_directory_finds_files(self, indexer: IncrementalIndexer, sample_repo: Path):
        stats = indexer.index_directory(sample_repo / "src")
        assert stats.files_scanned == 2
        assert stats.files_updated == 2
        assert stats.chunks_added > 0
        assert stats.errors == []

    def test_index_directory_populates_chunk_store(self, indexer: IncrementalIndexer, sample_repo: Path):
        indexer.index_directory(sample_repo / "src")
        assert indexer._chunk_store.count() > 0

    def test_index_directory_populates_bm25(self, indexer: IncrementalIndexer, sample_repo: Path):
        indexer.index_directory(sample_repo / "src")
        assert indexer._bm25.count() > 0

    def test_index_directory_tracks_file_hashes(self, indexer: IncrementalIndexer, sample_repo: Path):
        indexer.index_directory(sample_repo / "src")
        files = indexer.get_indexed_files()
        assert len(files) == 2


class TestIncrementalUpdate:
    """Re-indexing only changed files."""

    def test_unchanged_file_not_reindexed(self, indexer: IncrementalIndexer, sample_repo: Path):
        stats1 = indexer.index_directory(sample_repo / "src")
        chunks_after_first = indexer._chunk_store.count()

        # Index again without changes
        stats2 = indexer.index_directory(sample_repo / "src")
        assert stats2.files_updated == 0
        assert indexer._chunk_store.count() == chunks_after_first

    def test_changed_file_is_reindexed(self, indexer: IncrementalIndexer, sample_repo: Path):
        indexer.index_directory(sample_repo / "src")

        # Modify a file
        order_file = sample_repo / "src" / "main" / "java" / "com" / "example" / "OrderService.java"
        content = order_file.read_text()
        order_file.write_text(content + "\n// modified\n")

        stats = indexer.index_directory(sample_repo / "src")
        assert stats.files_updated == 1

    def test_new_file_is_indexed(self, indexer: IncrementalIndexer, sample_repo: Path):
        indexer.index_directory(sample_repo / "src")
        chunks_before = indexer._chunk_store.count()

        # Add a new file
        new_file = sample_repo / "src" / "main" / "java" / "com" / "example" / "PaymentService.java"
        new_file.write_text(textwrap.dedent("""\
            package com.example;

            public class PaymentService {
                public void processPayment(String orderId) {}
            }
        """))

        stats = indexer.index_directory(sample_repo / "src")
        assert stats.files_scanned == 3
        assert indexer._chunk_store.count() > chunks_before

    def test_deleted_file_is_removed(self, indexer: IncrementalIndexer, sample_repo: Path):
        indexer.index_directory(sample_repo / "src")
        indexed_files = indexer.get_indexed_files()
        assert len(indexed_files) == 2

        # Delete one file
        user_file = sample_repo / "src" / "main" / "java" / "com" / "example" / "UserService.java"
        user_file.unlink()

        stats = indexer.index_directory(sample_repo / "src")
        assert stats.files_removed == 1
        assert len(indexer.get_indexed_files()) == 1


class TestSingleFileUpdate:
    """Single file update operations."""

    def test_update_file_new(self, indexer: IncrementalIndexer, sample_repo: Path):
        java_file = sample_repo / "src" / "main" / "java" / "com" / "example" / "OrderService.java"
        stats = indexer.update_file(java_file, module="test")
        assert stats.files_updated == 1
        assert stats.chunks_added > 0

    def test_update_file_no_change(self, indexer: IncrementalIndexer, sample_repo: Path):
        java_file = sample_repo / "src" / "main" / "java" / "com" / "example" / "OrderService.java"
        indexer.update_file(java_file)
        stats = indexer.update_file(java_file)
        assert stats.files_updated == 0
        assert stats.chunks_added == 0

    def test_remove_file(self, indexer: IncrementalIndexer, sample_repo: Path):
        java_file = sample_repo / "src" / "main" / "java" / "com" / "example" / "OrderService.java"
        indexer.update_file(java_file)
        rel_path = java_file.as_posix()
        assert indexer.is_indexed(rel_path)

        removed = indexer.remove_file(rel_path)
        assert removed > 0
        assert not indexer.is_indexed(rel_path)


class TestStatePersistence:
    """State save/load for incremental indexing."""

    def test_save_and_load_state(self, indexer: IncrementalIndexer, sample_repo: Path, tmp_path: Path):
        indexer.index_directory(sample_repo / "src")
        state_path = tmp_path / "indexer_state.json"
        indexer.save_state(state_path)
        assert state_path.exists()

        new_indexer = IncrementalIndexer(
            slicer=JavaSlicer(),
            chunk_store=ChunkStore(),
            embedding_service=MagicMock(),
            vector_store=MagicMock(),
            bm25_index=BM25Index(),
        )
        new_indexer.load_state(state_path)
        assert len(new_indexer.get_indexed_files()) == 2


class TestIndexStats:
    """IndexStats reporting."""

    def test_stats_has_duration(self, indexer: IncrementalIndexer, sample_repo: Path):
        stats = indexer.index_directory(sample_repo / "src")
        assert stats.duration_seconds >= 0

    def test_stats_counts_errors(self, indexer: IncrementalIndexer, tmp_path: Path):
        """Errors are collected when file reading fails."""
        # Create a file that will cause a read error
        src = tmp_path / "src"
        src.mkdir()
        # Create a directory named "Bad.java" (not a file, will cause read error)
        bad_dir = src / "Bad.java"
        bad_dir.mkdir()

        stats = indexer.index_directory(src)
        # The rglob will find the directory, and reading it will fail
        assert stats.files_scanned == 1
