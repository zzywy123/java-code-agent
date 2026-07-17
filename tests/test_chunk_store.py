"""Tests for ChunkStore: chunk storage, queries, and persistence.

Validates:
- Adding and retrieving chunks
- File-based and method-based queries
- Removal operations
- JSON persistence (save/load)
- Chunk ID determinism
- Index consistency
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.indexing.chunk_store import ChunkStore, compute_chunk_id, compute_file_hash
from agent.models import CodeChunk, CodeSlice


def _make_slice(
    file_path: str = "src/Main.java",
    class_name: str = "Main",
    method_name: str = "main",
    package: str = "com.example",
    start_line: int = 1,
    end_line: int = 10,
    content: str = "public static void main(String[] args) {}",
    symbol_signature: str = "com.example.Main.main(String[])",
) -> CodeSlice:
    return CodeSlice(
        module="test",
        package=package,
        class_name=class_name,
        method_name=method_name,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        symbol_signature=symbol_signature,
    )


def _make_chunk(
    file_path: str = "src/Main.java",
    class_name: str = "Main",
    method_name: str = "main",
    content: str = "public static void main(String[] args) {}",
    symbol_signature: str = "com.example.Main.main(String[])",
) -> CodeChunk:
    s = _make_slice(
        file_path=file_path,
        class_name=class_name,
        method_name=method_name,
        content=content,
        symbol_signature=symbol_signature,
    )
    chunk_id = compute_chunk_id(s.file_path, s.symbol_signature, s.content)
    return CodeChunk(
        chunk_id=chunk_id,
        slice=s,
        file_hash=compute_file_hash(content),
    )


class TestChunkStoreBasic:
    """Basic CRUD operations."""

    def test_add_and_count(self):
        store = ChunkStore()
        chunk = _make_chunk()
        added = store.add([chunk])
        assert added == 1
        assert store.count() == 1

    def test_add_duplicate_does_not_increase_count(self):
        store = ChunkStore()
        chunk = _make_chunk()
        store.add([chunk])
        store.add([chunk])  # Same chunk
        assert store.count() == 1

    def test_get_by_id(self):
        store = ChunkStore()
        chunk = _make_chunk()
        store.add([chunk])
        retrieved = store.get_by_id(chunk.chunk_id)
        assert retrieved is not None
        assert retrieved.chunk_id == chunk.chunk_id

    def test_get_by_id_missing(self):
        store = ChunkStore()
        assert store.get_by_id("nonexistent") is None

    def test_get_by_file(self):
        store = ChunkStore()
        c1 = _make_chunk(file_path="src/A.java", content="class A {}", symbol_signature="A")
        c2 = _make_chunk(file_path="src/A.java", class_name="A", method_name="foo",
                         content="void foo() {}", symbol_signature="A.foo()")
        c3 = _make_chunk(file_path="src/B.java", content="class B {}", symbol_signature="B")
        store.add([c1, c2, c3])

        a_chunks = store.get_by_file("src/A.java")
        assert len(a_chunks) == 2

        b_chunks = store.get_by_file("src/B.java")
        assert len(b_chunks) == 1

    def test_get_by_method(self):
        store = ChunkStore()
        c1 = _make_chunk(class_name="Service", method_name="save", content="void save() {}",
                         symbol_signature="Service.save()")
        c2 = _make_chunk(class_name="Service", method_name="delete", content="void delete() {}",
                         symbol_signature="Service.delete()")
        store.add([c1, c2])

        save_chunks = store.get_by_method("Service", "save")
        assert len(save_chunks) == 1
        assert save_chunks[0].slice.method_name == "save"


class TestChunkStoreRemoval:
    """Removal operations."""

    def test_remove_by_file(self):
        store = ChunkStore()
        c1 = _make_chunk(file_path="src/A.java", content="a", symbol_signature="A")
        c2 = _make_chunk(file_path="src/B.java", content="b", symbol_signature="B")
        store.add([c1, c2])

        removed = store.remove_by_file("src/A.java")
        assert removed == 1
        assert store.count() == 1
        assert store.get_by_file("src/A.java") == []

    def test_remove_by_id(self):
        store = ChunkStore()
        chunk = _make_chunk()
        store.add([chunk])
        assert store.remove_by_id(chunk.chunk_id) is True
        assert store.count() == 0

    def test_remove_nonexistent_id(self):
        store = ChunkStore()
        assert store.remove_by_id("nonexistent") is False

    def test_clear(self):
        store = ChunkStore()
        store.add([_make_chunk()])
        store.add([_make_chunk(file_path="src/B.java", content="b", symbol_signature="B")])
        store.clear()
        assert store.count() == 0
        assert store.file_count() == 0


class TestChunkStorePersistence:
    """JSON save/load persistence."""

    def test_save_and_load(self, tmp_path: Path):
        store = ChunkStore()
        c1 = _make_chunk(file_path="src/A.java", content="a", symbol_signature="A")
        c2 = _make_chunk(file_path="src/B.java", content="b", symbol_signature="B")
        store.add([c1, c2])

        save_path = tmp_path / "chunks.json"
        store.save(save_path)
        assert save_path.exists()

        new_store = ChunkStore()
        new_store.load(save_path)
        assert new_store.count() == 2

    def test_load_preserves_metadata(self, tmp_path: Path):
        store = ChunkStore()
        chunk = _make_chunk()
        store.add([chunk])

        save_path = tmp_path / "chunks.json"
        store.save(save_path)

        new_store = ChunkStore()
        new_store.load(save_path)
        loaded = new_store.get_by_id(chunk.chunk_id)
        assert loaded is not None
        assert loaded.slice.class_name == chunk.slice.class_name
        assert loaded.slice.method_name == chunk.slice.method_name

    def test_load_nonexistent_file_is_noop(self, tmp_path: Path):
        store = ChunkStore()
        store.load(tmp_path / "nonexistent.json")
        assert store.count() == 0


class TestChunkIdDeterminism:
    """Chunk ID computation."""

    def test_same_input_same_id(self):
        id1 = compute_chunk_id("src/A.java", "A.foo()", "void foo() {}")
        id2 = compute_chunk_id("src/A.java", "A.foo()", "void foo() {}")
        assert id1 == id2

    def test_different_content_different_id(self):
        id1 = compute_chunk_id("src/A.java", "A.foo()", "void foo() {}")
        id2 = compute_chunk_id("src/A.java", "A.foo()", "void bar() {}")
        assert id1 != id2

    def test_different_path_different_id(self):
        id1 = compute_chunk_id("src/A.java", "A.foo()", "x")
        id2 = compute_chunk_id("src/B.java", "A.foo()", "x")
        assert id1 != id2

    def test_different_signature_different_id(self):
        id1 = compute_chunk_id("src/A.java", "A.foo()", "x")
        id2 = compute_chunk_id("src/A.java", "A.bar()", "x")
        assert id1 != id2


class TestChunkStoreIndexConsistency:
    """Index consistency after operations."""

    def test_file_index_cleanup_after_removal(self):
        store = ChunkStore()
        chunk = _make_chunk(file_path="src/A.java")
        store.add([chunk])
        store.remove_by_file("src/A.java")
        assert store.list_files() == []

    def test_method_index_cleanup_after_removal(self):
        store = ChunkStore()
        chunk = _make_chunk(class_name="Svc", method_name="run", content="void run() {}",
                            symbol_signature="Svc.run()")
        store.add([chunk])
        store.remove_by_file(chunk.slice.file_path)
        assert store.get_by_method("Svc", "run") == []

    def test_list_all_returns_all_chunks(self):
        store = ChunkStore()
        chunks = [
            _make_chunk(file_path=f"src/{c}.java", content=c, symbol_signature=c)
            for c in ["A", "B", "C"]
        ]
        store.add(chunks)
        assert len(store.list_all()) == 3
