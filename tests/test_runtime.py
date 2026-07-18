"""Shared runtime bootstrap tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import AgentConfig, EmbeddingConfig, LLMConfig, MemoryConfig, RAGConfig
from agent import runtime as runtime_module


def test_create_app_runtime_wires_one_shared_service(monkeypatch, tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    registry = {"read_file": object(), "run_tests": object()}
    search_engine = SimpleNamespace(_vector_store=None)
    session_manager = SimpleNamespace(
        get_or_create_active_session=lambda: "session-1",
        get_storage_dir=lambda: tmp_path / "sessions",
    )
    workflow = object()
    workflow_args = {}

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: (
            LLMConfig(provider="ollama"),
            SimpleNamespace(repo_root=repo, require_approval=True),
            AgentConfig(),
        ),
    )
    monkeypatch.setattr(runtime_module, "create_tool_registry", lambda root: registry)
    monkeypatch.setattr(runtime_module, "build_rag_index", lambda root: (search_engine, 7))
    monkeypatch.setattr(runtime_module, "create_llm", lambda config: "llm")
    monkeypatch.setattr(runtime_module, "SessionManager", lambda config, llm: session_manager)
    memory_config = MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
    )
    monkeypatch.setattr(runtime_module, "load_memory_config", lambda: memory_config)
    monkeypatch.setattr(runtime_module, "load_workflow_config", lambda: object())
    monkeypatch.setattr(runtime_module, "load_rag_config", lambda: object())
    monkeypatch.setattr(runtime_module, "build_mcp_adapter", lambda root: "mcp")
    def create_workflow(**kwargs):
        workflow_args.update(kwargs)
        return workflow

    monkeypatch.setattr(runtime_module, "create_workflow", create_workflow)

    app_runtime = runtime_module.create_app_runtime(repo)

    assert app_runtime.repo_root == repo.resolve()
    assert app_runtime.session_id == "session-1"
    assert app_runtime.tool_count == 2
    assert app_runtime.chunk_count == 7
    assert app_runtime.search_type == "BM25"
    assert app_runtime.service._workflow is workflow
    assert app_runtime.service._sessions is session_manager
    assert workflow_args["require_approval"] is True


def test_create_app_runtime_namespaces_session_storage(monkeypatch, tmp_path: Path):
    captured = {}
    memory_config = MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: (
            LLMConfig(provider="ollama"),
            SimpleNamespace(repo_root=repo, require_approval=True),
            AgentConfig(),
        ),
    )
    monkeypatch.setattr(runtime_module, "create_tool_registry", lambda root: {})
    monkeypatch.setattr(
        runtime_module,
        "build_rag_index",
        lambda root: (SimpleNamespace(_vector_store=None), 0),
    )
    monkeypatch.setattr(runtime_module, "create_llm", lambda config: "llm")
    monkeypatch.setattr(runtime_module, "load_memory_config", lambda: memory_config)
    monkeypatch.setattr(runtime_module, "build_mcp_adapter", lambda root: None)
    monkeypatch.setattr(runtime_module, "load_workflow_config", lambda: object())
    monkeypatch.setattr(runtime_module, "load_rag_config", lambda: object())
    monkeypatch.setattr(runtime_module, "create_workflow", lambda **kwargs: object())

    class FakeSessionManager:
        def __init__(self, config, llm):
            captured["config"] = config

        def get_or_create_active_session(self):
            return "session-1"

        def get_storage_dir(self):
            return tmp_path / "sessions"

    monkeypatch.setattr(runtime_module, "SessionManager", FakeSessionManager)

    runtime_module.create_app_runtime(repo, storage_namespace="browser-1")

    assert Path(captured["config"].checkpoint_dir).parent == tmp_path / "checkpoints"
    assert Path(captured["config"].long_term_persist_dir).parent == tmp_path / "memory"
    assert Path(captured["config"].checkpoint_dir).name == Path(
        captured["config"].long_term_persist_dir
    ).name


def test_create_app_runtime_rejects_missing_repository(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        runtime_module.create_app_runtime(tmp_path / "missing")


def test_build_rag_index_degrades_to_bm25_when_local_model_is_unavailable(
    monkeypatch, tmp_path: Path
):
    repo = tmp_path / "repo"
    source_dir = repo / "src" / "main" / "java"
    source_dir.mkdir(parents=True)
    (source_dir / "OrderService.java").write_text(
        "class OrderService { int total() { return 1; } }",
        encoding="utf-8",
    )
    rag_config = RAGConfig(
        enable_vector=True,
        index_dir=str(tmp_path / "index"),
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    monkeypatch.setattr(runtime_module, "load_rag_config", lambda: rag_config)
    monkeypatch.setattr(
        runtime_module,
        "load_embedding_config",
        lambda: EmbeddingConfig(local_files_only=True),
    )

    from agent.indexing.embedding import EmbeddingService

    def fail_initialize(self):
        raise RuntimeError("local model is unavailable")

    monkeypatch.setattr(EmbeddingService, "initialize", fail_initialize)

    engine, chunk_count = runtime_module.build_rag_index(repo)

    assert engine._vector_store is None
    assert chunk_count > 0
    assert engine.search("OrderService")
