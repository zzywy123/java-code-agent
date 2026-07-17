"""CLI index lifecycle tests."""

from pathlib import Path

from agent.config import RAGConfig
from agent import runtime


def test_build_rag_index_reuses_persisted_bm25_cache(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    source = repo / "src" / "main" / "java" / "com" / "example"
    source.mkdir(parents=True)
    (source / "OrderService.java").write_text(
        """package com.example;

public class OrderService {
    public int calculateTotal(int value) {
        return value;
    }
}
""",
        encoding="utf-8",
    )
    config = RAGConfig(
        enable_vector=False,
        index_dir=str(tmp_path / "index"),
    )
    monkeypatch.setattr(runtime, "load_rag_config", lambda: config)

    first_engine, first_count = runtime.build_rag_index(repo)
    first_stats = first_engine.index_stats
    second_engine, second_count = runtime.build_rag_index(repo)
    second_stats = second_engine.index_stats

    assert first_count > 0
    assert first_stats.files_updated == 1
    assert second_count == first_count
    assert second_stats.files_updated == 0
    assert second_stats.files_removed == 0
    assert second_engine._vector_store is None
    assert second_engine.search("calculateTotal")

    cache_dir = second_engine.index_cache_dir
    assert (cache_dir / "chunks.json").exists()
    assert (cache_dir / "state.json").exists()
    assert (cache_dir / "manifest.json").exists()
