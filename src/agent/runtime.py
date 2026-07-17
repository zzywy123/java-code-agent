"""Shared application bootstrap used by CLI, Streamlit and evaluations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent.app_service import AppService
from agent.config import (
    AgentConfig,
    LLMConfig,
    load_config,
    load_embedding_config,
    load_mcp_config,
    load_memory_config,
    load_rag_config,
    load_workflow_config,
)
from agent.indexing.bm25_index import BM25Index
from agent.indexing.chunk_store import ChunkStore
from agent.indexing.hybrid_search import HybridSearchEngine
from agent.indexing.incremental import IncrementalIndexer
from agent.indexing.java_slicer import JavaSlicer
from agent.llm_client import create_llm
from agent.session import SessionManager
from agent.tools.factory import create_tool_registry
from agent.workflow import create_workflow

logger = logging.getLogger(__name__)


@dataclass
class AppRuntime:
    service: AppService
    session_id: str
    repo_root: Path
    llm_config: LLMConfig
    tool_count: int
    chunk_count: int
    search_type: str
    index_duration_seconds: float


def create_app_runtime(
    repo_root: Path | str | None = None,
    *,
    llm_config: LLMConfig | None = None,
    agent_config: AgentConfig | None = None,
) -> AppRuntime:
    """Construct one fully wired application runtime."""
    loaded_llm, security_config, loaded_agent = load_config()
    llm_config = llm_config or loaded_llm
    agent_config = agent_config or loaded_agent
    if repo_root is not None:
        resolved_repo = Path(repo_root).expanduser().resolve()
        if not resolved_repo.is_dir():
            raise ValueError(f"Repository root does not exist: {resolved_repo}")
        security_config.repo_root = resolved_repo
    resolved_repo = security_config.repo_root.resolve()

    registry = create_tool_registry(resolved_repo)
    started = time.perf_counter()
    search_engine, chunk_count = build_rag_index(resolved_repo)
    index_duration = time.perf_counter() - started
    search_type = (
        "Hybrid (BM25 + Vector)"
        if search_engine._vector_store is not None
        else "BM25"
    )

    llm = create_llm(llm_config)
    session_manager = SessionManager(load_memory_config(), llm=llm)
    session_id = session_manager.get_or_create_active_session()
    try:
        mcp_adapter = build_mcp_adapter(resolved_repo)
    except Exception as exc:
        logger.warning("MCP initialization failed; using direct tools: %s", exc)
        mcp_adapter = None

    workflow = create_workflow(
        llm=llm,
        llm_config=llm_config,
        agent_config=agent_config,
        workflow_config=load_workflow_config(),
        rag_config=load_rag_config(),
        tool_registry=registry,
        search_engine=search_engine,
        session_manager=session_manager,
        repo_root=resolved_repo,
        mcp_adapter=mcp_adapter,
        require_approval=security_config.require_approval,
    )
    return AppRuntime(
        service=AppService(workflow, session_manager, project_root=resolved_repo),
        session_id=session_id,
        repo_root=resolved_repo,
        llm_config=llm_config,
        tool_count=len(registry),
        chunk_count=chunk_count,
        search_type=search_type,
        index_duration_seconds=index_duration,
    )


def build_rag_index(repo_root: Path):
    """Load the persisted index and update only changed Java files."""
    rag_config = load_rag_config()
    embedding_config = load_embedding_config() if rag_config.enable_vector else None
    repo_key = hashlib.sha256(
        str(repo_root.resolve()).casefold().encode("utf-8")
    ).hexdigest()[:16]
    index_base = Path(rag_config.index_dir).expanduser()
    if not index_base.is_absolute():
        index_base = Path.cwd() / index_base
    cache_dir = index_base / repo_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = cache_dir / "chunks.json"
    state_path = cache_dir / "state.json"
    manifest_path = cache_dir / "manifest.json"

    vector_signature = "disabled"
    if embedding_config is not None:
        model_name = (
            embedding_config.model_name
            if embedding_config.provider.value == "local"
            else embedding_config.openai_model
        )
        vector_signature = (
            f"{embedding_config.provider.value}:{model_name}:"
            f"{embedding_config.dimension}"
        )

    force_reindex = rag_config.force_reindex
    if manifest_path.exists() and rag_config.enable_vector:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            force_reindex = force_reindex or (
                manifest.get("vector_signature") != vector_signature
            )
        except (OSError, json.JSONDecodeError):
            force_reindex = True

    chunk_store = ChunkStore()
    bm25_index = BM25Index()
    if not force_reindex and chunk_path.exists():
        try:
            chunk_store.load(chunk_path)
            bm25_index.add(chunk_store.list_all())
        except Exception as exc:
            logger.warning("Index cache is invalid; rebuilding: %s", exc)
            chunk_store.clear()
            bm25_index.clear()
            force_reindex = True

    java_dir = repo_root / "src" / "main" / "java"
    if not java_dir.exists():
        java_dir = repo_root / "src"
    if not java_dir.exists():
        java_dir = repo_root

    embedding_service = None
    vector_store = None
    if rag_config.enable_vector and embedding_config is not None:
        try:
            from agent.indexing.embedding import EmbeddingService
            from agent.indexing.vector_store import VectorStore

            embedding_service = EmbeddingService(embedding_config)
            # Resolve model availability during startup. Missing local files
            # degrade here instead of blocking the first user query.
            embedding_service.initialize()
            chroma_dir = Path(rag_config.chroma_persist_dir).expanduser()
            if not chroma_dir.is_absolute():
                chroma_dir = cache_dir / chroma_dir
            else:
                chroma_dir = chroma_dir / repo_key
            vector_store = VectorStore(
                rag_config,
                embedding_service,
                persist_dir=chroma_dir,
            )
            if force_reindex:
                vector_store.clear()
        except Exception as exc:
            logger.warning("Chroma unavailable; using BM25: %s", exc)
            embedding_service = None
            vector_store = None

    indexer = IncrementalIndexer(
        slicer=JavaSlicer(),
        chunk_store=chunk_store,
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_index=bm25_index,
    )
    if not force_reindex and state_path.exists():
        try:
            indexer.load_state(state_path)
        except Exception as exc:
            logger.warning("Index state is invalid; checking all files: %s", exc)

    missing_embeddings = bool(
        vector_store is not None
        and any(chunk.embedding is None for chunk in chunk_store.list_all())
    )
    stats = indexer.index_directory(
        java_dir,
        force=force_reindex or missing_embeddings,
    )
    vector_store = indexer.get_vector_store()
    if vector_store is not None and chunk_store.count() > 0:
        try:
            if vector_store.count() != chunk_store.count():
                vector_store.add(chunk_store.list_all())
        except Exception as exc:
            logger.warning("Vector index restore failed; using BM25: %s", exc)
            vector_store = None

    chunk_store.save(chunk_path)
    indexer.save_state(state_path)
    manifest_path.write_text(
        json.dumps({
            "version": 1,
            "repo_root": str(repo_root.resolve()),
            "vector_signature": vector_signature,
            "vector_ready": vector_store is not None,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if stats.errors:
        logger.warning("Index completed with %d errors: %s", len(stats.errors), stats.errors[0])
    search_engine = HybridSearchEngine(rag_config, vector_store, bm25_index)
    search_engine.index_stats = stats
    search_engine.index_cache_dir = cache_dir
    return search_engine, chunk_store.count()


def build_mcp_adapter(repo_root: Path):
    """Create the read-only stdio MCP adapter used by Researcher."""
    from agent.agents.permission import AgentRole, PermissionManager
    from agent.mcp.client import MCPToolAdapter, create_mcp_client

    config = load_mcp_config()
    if not config.enabled:
        return None
    if config.transport != "stdio":
        raise ValueError("Only MCP stdio transport is supported")
    env = os.environ.copy()
    env["AGENT_REPO_ROOT"] = str(repo_root)
    env["MCP_AGENT_ROLE"] = AgentRole.RESEARCHER.value
    client = create_mcp_client(
        [
            sys.executable,
            "-m",
            "agent.mcp.server",
            "--repo-root",
            str(repo_root),
            "--role",
            "researcher",
        ],
        env=env,
        cwd=Path.cwd(),
    )
    adapter = MCPToolAdapter(client, PermissionManager(), AgentRole.RESEARCHER)
    available_tools = adapter.initialize_sync()
    logger.info("MCP stdio ready with %d read-only tools", len(available_tools))
    return adapter
