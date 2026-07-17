"""Shared runtime bootstrap tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import AgentConfig, LLMConfig
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

    monkeypatch.setattr(
        runtime_module,
        "load_config",
        lambda: (
            LLMConfig(provider="ollama"),
            SimpleNamespace(repo_root=repo),
            AgentConfig(),
        ),
    )
    monkeypatch.setattr(runtime_module, "create_tool_registry", lambda root: registry)
    monkeypatch.setattr(runtime_module, "build_rag_index", lambda root: (search_engine, 7))
    monkeypatch.setattr(runtime_module, "create_llm", lambda config: "llm")
    monkeypatch.setattr(runtime_module, "SessionManager", lambda config, llm: session_manager)
    monkeypatch.setattr(runtime_module, "load_memory_config", lambda: object())
    monkeypatch.setattr(runtime_module, "load_workflow_config", lambda: object())
    monkeypatch.setattr(runtime_module, "load_rag_config", lambda: object())
    monkeypatch.setattr(runtime_module, "build_mcp_adapter", lambda root: "mcp")
    monkeypatch.setattr(runtime_module, "create_workflow", lambda **kwargs: workflow)

    app_runtime = runtime_module.create_app_runtime(repo)

    assert app_runtime.repo_root == repo.resolve()
    assert app_runtime.session_id == "session-1"
    assert app_runtime.tool_count == 2
    assert app_runtime.chunk_count == 7
    assert app_runtime.search_type == "BM25"
    assert app_runtime.service._workflow is workflow
    assert app_runtime.service._sessions is session_manager


def test_create_app_runtime_rejects_missing_repository(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        runtime_module.create_app_runtime(tmp_path / "missing")
