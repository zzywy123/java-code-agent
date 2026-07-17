"""CLI entry point for the Java Coding Agent.

Phase 1 + Phase 2 integrated:
- Rich-formatted output with syntax highlighting
- Continuous conversation loop
- Approval flow with diff display via LangGraph interrupt/resume
- Agentic RAG for code search (Hybrid Search = BM25 + Vector)
- Multi-Agent routing (Supervisor → Researcher/Coder/Tester/Verifier)
- Agent permission enforcement
- Streaming display: retrieval, agent switch, tool call, approval events
"""

from __future__ import annotations

import os
import logging
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from agent.agent_graph import create_agent_graph
from agent.agent_state import AgentState
from agent.app_service import AppService
from agent.config import (
    load_config,
    load_mcp_config,
    load_memory_config,
    load_rag_config,
    load_workflow_config,
)
from agent.llm_client import create_llm
from agent.models import (
    ApprovalDecision,
    ReviewArtifact,
    SearchArtifact,
    TestResultArtifact,
    ToolCallRequest,
    ToolStatus,
)
from agent.observability.logger import configure_logging
from agent.runtime import create_app_runtime
from agent.tools.base import ToolRegistry
from agent.tools.build_tools import RunTestsTool
from agent.tools.file_tools import ListFilesTool, ReadFileTool
from agent.tools.git_tools import GitDiffTool, GitLogTool, GitStatusTool
from agent.tools.patch_tools import ApplyPatchTool, UndoPatchTool
from agent.tools.search_tools import SearchCodeTool
from agent.session import SessionManager
from agent.workflow import create_workflow, initial_workflow_state

logger = logging.getLogger(__name__)

# Suppress HuggingFace symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

console = Console(force_terminal=True)


# ============================================================
# Tool Registry
# ============================================================

def create_tool_registry(repo_root: Path) -> ToolRegistry:
    """Create and configure the tool registry."""
    from agent.tools.factory import create_tool_registry as factory

    return factory(repo_root)


# ============================================================
# Phase 2: RAG Index Builder
# ============================================================

def build_rag_index(repo_root: Path):
    """Load the persisted index and update only changed Java files."""
    import hashlib
    import json

    from agent.indexing.java_slicer import JavaSlicer
    from agent.indexing.chunk_store import ChunkStore
    from agent.indexing.bm25_index import BM25Index
    from agent.indexing.hybrid_search import HybridSearchEngine
    from agent.indexing.incremental import IncrementalIndexer

    rag_config = load_rag_config()
    embedding_config = None
    if rag_config.enable_vector:
        from agent.config import load_embedding_config
        embedding_config = load_embedding_config()

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

    slicer = JavaSlicer()
    chunk_store = ChunkStore()
    bm25_index = BM25Index()
    if not force_reindex and chunk_path.exists():
        try:
            chunk_store.load(chunk_path)
            bm25_index.add(chunk_store.list_all())
        except Exception as exc:
            logger.warning("索引缓存损坏，将执行全量重建: %s", exc)
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
            logger.warning("Chroma不可用，降级为BM25: %s", exc)
            embedding_service = None
            vector_store = None

    indexer = IncrementalIndexer(
        slicer=slicer,
        chunk_store=chunk_store,
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_index=bm25_index,
    )
    if not force_reindex and state_path.exists():
        try:
            indexer.load_state(state_path)
        except Exception as exc:
            logger.warning("索引状态损坏，将重新检查全部文件: %s", exc)

    missing_embeddings = bool(
        vector_store is not None
        and any(chunk.embedding is None for chunk in chunk_store.list_all())
    )
    stats = indexer.index_directory(
        java_dir,
        force=force_reindex or missing_embeddings,
    )
    vector_store = indexer.get_vector_store()

    # Restore Chroma from cached embeddings if its persistent directory was
    # removed independently from the chunk cache.
    if vector_store is not None and chunk_store.count() > 0:
        try:
            if vector_store.count() != chunk_store.count():
                vector_store.add(chunk_store.list_all())
        except Exception as exc:
            logger.warning("向量索引恢复失败，当前会话降级为BM25: %s", exc)
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
        logger.warning("索引完成但有 %d 个错误: %s", len(stats.errors), stats.errors[0])
    logger.info(
        "索引更新完成: scanned=%d updated=%d removed=%d chunks=%d duration=%.3fs cache=%s",
        stats.files_scanned,
        stats.files_updated,
        stats.files_removed,
        chunk_store.count(),
        stats.duration_seconds,
        cache_dir,
    )

    search_engine = HybridSearchEngine(rag_config, vector_store, bm25_index)
    search_engine.index_stats = stats
    search_engine.index_cache_dir = cache_dir
    return search_engine, chunk_store.count()


# ============================================================
# Phase 2: Multi-Agent System
# ============================================================

def build_multi_agent_system(tool_registry: ToolRegistry, search_engine):
    """Build the Multi-Agent system with Supervisor and sub-agents."""
    from agent.agents.permission import PermissionManager, AgentRole
    from agent.agents.supervisor import SupervisorAgent
    from agent.agents.researcher import ResearcherAgent
    from agent.agents.coder import CoderAgent
    from agent.agents.tester import TesterAgent
    from agent.agents.verifier import VerifierAgent
    from agent.rag.query_rewriter import QueryRewriter
    from agent.rag.evidence_judge import EvidenceJudge
    from agent.rag.agentic_rag import AgenticRAG

    rag_config = load_rag_config()
    pm = PermissionManager()

    # Build Agentic RAG
    query_rewriter = QueryRewriter(llm=None)
    evidence_judge = EvidenceJudge(threshold=0.2)
    agentic_rag = AgenticRAG(rag_config, search_engine, query_rewriter, evidence_judge)

    supervisor = SupervisorAgent(llm=None)
    researcher = ResearcherAgent(tool_registry, pm, agentic_rag)
    coder = CoderAgent(tool_registry, pm)
    tester = TesterAgent(tool_registry, pm)
    verifier = VerifierAgent(tool_registry, pm)

    return {
        "supervisor": supervisor,
        "researcher": researcher,
        "coder": coder,
        "tester": tester,
        "verifier": verifier,
        "permission_manager": pm,
    }


# ============================================================
# Display Functions
# ============================================================

def display_welcome():
    """Display welcome message."""
    console.print(Panel(
        "[bold green]Java Coding Agent[/bold green]\n"
        "类似 Claude Code 的 Java 仓库代码助手\n\n"
        "[dim]输入你的问题或指令，Agent 会搜索代码、修改文件、运行测试。[/dim]\n"
        "[dim]输入 'quit' 或 'exit' 退出。[/dim]",
        title="Welcome",
        border_style="green",
    ))


def display_tool_call(name: str, args: dict):
    """Display a tool call being executed."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    for k, v in args.items():
        table.add_row(k, str(v)[:100])
    console.print(Panel(table, title=f"工具调用: {name}", border_style="blue"))


def display_tool_result(name: str, status: ToolStatus, output: str):
    """Display a tool execution result."""
    status_colors = {
        ToolStatus.SUCCESS: "green",
        ToolStatus.ERROR: "red",
        ToolStatus.DENIED: "yellow",
        ToolStatus.TIMEOUT: "red",
        ToolStatus.NOT_FOUND: "yellow",
        ToolStatus.INVALID_ARGUMENT: "red",
        ToolStatus.EXECUTION_ERROR: "red",
    }
    color = status_colors.get(status, "white")
    console.print(Panel(
        output[:2000],  # Truncate display
        title=f"{name} [{status.value}]",
        border_style=color,
    ))


def display_approval_request(approval_data: dict) -> bool:
    """Display approval request and get user decision."""
    summary = approval_data.get("summary", "未知操作")
    tool_calls = approval_data.get("tool_calls", [])

    console.print(Panel(
        f"[bold yellow]需要审批[/bold yellow]\n\n"
        f"操作: {summary}\n"
        f"工具调用: {len(tool_calls)} 个",
        title="审批请求",
        border_style="yellow",
    ))

    for tc in tool_calls:
        name = tc.get("name", "unknown")
        args = tc.get("arguments", {})
        console.print(f"  - {name}: {args}")

    while True:
        response = console.input("\n[bold]批准执行？(y/n): [/bold]").strip().lower()
        if response in ("y", "yes", "是"):
            return True
        if response in ("n", "no", "否"):
            return False
        console.print("[yellow]请输入 y 或 n[/yellow]")


def display_search_results(sources, max_show=5):
    """Display RAG search results as a table."""
    if not sources:
        return
    table = Table(title="检索结果", show_header=True, header_style="bold cyan")
    table.add_column("#", width=3)
    table.add_column("文件", width=30)
    table.add_column("方法", width=25)
    table.add_column("行号", width=10)
    table.add_column("分数", width=8)

    for i, r in enumerate(sources[:max_show], 1):
        s = r.chunk.slice
        file_name = Path(s.file_path).name
        table.add_row(
            str(i), file_name,
            f"{s.class_name}.{s.method_name}",
            f"{s.start_line}-{s.end_line}",
            f"{r.score:.2f}",
        )
    console.print(table)


def display_agent_switch(role: str, reason: str = ""):
    """Display agent switching event."""
    colors = {
        "researcher": "blue",
        "coder": "yellow",
        "tester": "green",
        "verifier": "magenta",
    }
    color = colors.get(role, "white")
    msg = f"[bold {color}]→ Agent: {role}[/bold {color}]"
    if reason:
        msg += f" [dim]({reason})[/dim]"
    console.print(msg)


def display_final_answer(answer: str | None, error: str | None, patches: list):
    """Display the final answer and summary."""
    if answer:
        console.print(Panel(
            Markdown(answer),
            title="回答",
            border_style="green",
        ))

    if error:
        console.print(Panel(
            f"[red]{error}[/red]",
            title="错误",
            border_style="red",
        ))

    if patches:
        console.print(Panel(
            f"本次会话修改了 [bold]{len(patches)}[/bold] 个文件:\n"
            + "\n".join(f"  - {p.file_path}" for p in patches),
            title="变更记录",
            border_style="cyan",
        ))


def display_stream_events(events) -> None:
    """Render UI-independent AppService events in the terminal."""
    for event in events:
        data = event.data
        if event.event_type == "agent_switch":
            display_agent_switch(str(data.get("agent", "unknown")))
        elif event.event_type == "rag_retrieval":
            artifact = SearchArtifact.model_validate(data)
            display_search_results(artifact.results)
            if artifact.analysis:
                console.print(f"[dim]{artifact.analysis}[/dim]")
        elif event.event_type == "tool_call":
            display_tool_call(
                str(data.get("name", "unknown")),
                dict(data.get("arguments", {})),
            )
        elif event.event_type == "tool_result":
            content = str(data.get("content", ""))
            content_lower = content.lower()
            explicit_status = str(data.get("status", ""))
            try:
                status = ToolStatus(explicit_status) if explicit_status else ToolStatus.SUCCESS
            except ValueError:
                status = ToolStatus.SUCCESS
            for candidate in (
                ToolStatus.DENIED,
                ToolStatus.TIMEOUT,
                ToolStatus.EXECUTION_ERROR,
                ToolStatus.INVALID_ARGUMENT,
                ToolStatus.NOT_FOUND,
                ToolStatus.ERROR,
            ):
                if f"[{candidate.value}]" in content_lower:
                    status = candidate
                    break
            display_tool_result(str(data.get("name", "tool")), status, content)
        elif event.event_type == "test_result":
            result = TestResultArtifact.model_validate(data)
            display_tool_result(
                "run_tests",
                ToolStatus.SUCCESS if result.success else ToolStatus.EXECUTION_ERROR,
                result.stdout or result.stderr,
            )
        elif event.event_type == "review_result":
            review = ReviewArtifact.model_validate(data)
            console.print(Panel(
                review.summary + ("\n" + "\n".join(review.issues) if review.issues else ""),
                title="Verifier",
                border_style="green" if review.approved else "red",
            ))
        elif event.event_type == "rework":
            console.print(
                f"[yellow]Verifier 要求返工，第 {data.get('rework_count', 0)} 次[/yellow]"
            )
        elif event.event_type == "token_usage":
            cost = data.get("cost")
            cost_text = f", cost={cost:.6f}" if isinstance(cost, (int, float)) else ""
            estimated = " estimated" if data.get("estimated") else ""
            console.print(
                "[dim]Token: "
                f"in={data.get('input_tokens', 0)}, "
                f"out={data.get('output_tokens', 0)}, "
                f"total={data.get('total_tokens', 0)}"
                f"{cost_text}{estimated}[/dim]"
            )
        elif event.event_type == "error":
            console.print(Panel(
                f"[red]{data.get('message', '未知错误')}[/red]",
                title="错误",
                border_style="red",
            ))


def run_app_service_with_approval(
    service: AppService,
    session_id: str,
    query: str,
):
    """Drive AppService until completion while collecting user approvals."""
    result = service.submit(session_id, query)
    while True:
        display_stream_events(result.events)
        if not result.needs_approval:
            display_final_answer(result.final_answer, result.error, result.patches)
            return result

        approved = display_approval_request(result.approval_data or {})
        result = service.resume(
            session_id,
            ApprovalDecision(
                approved=approved,
                reason="" if approved else "用户拒绝操作",
            ),
        )


# ============================================================
# Phase 2: Query Processor
# ============================================================

def process_query_with_agents(
    query: str,
    agents: dict,
    llm_config=None,
    agent_config=None,
    tool_registry=None,
    repo_root=None,
) -> None:
    """Process a user query through the Multi-Agent system.

    Routes to the appropriate agent and displays results.
    For code questions: RAG retrieval → LLM answer generation.
    For code changes: Phase 1 agent graph with approval flow.
    """
    supervisor = agents["supervisor"]
    researcher = agents["researcher"]
    coder = agents["coder"]
    tester = agents["tester"]
    verifier = agents["verifier"]

    # Supervisor routes the query
    role = supervisor.route(query)
    display_agent_switch(role.value)

    if role.value == "researcher":
        # Researcher uses Agentic RAG
        console.print("[dim]正在检索...[/dim]")
        artifact = researcher.run(query)

        # Display search results
        display_search_results(artifact.results)

        if not artifact.results:
            console.print("[yellow]未找到相关代码。请尝试更具体的查询。[/yellow]")
            return

        # Use LLM to generate answer from RAG results
        if llm_config and tool_registry and repo_root:
            _llm_answer_with_rag(query, artifact, llm_config, agent_config, tool_registry, repo_root)
        else:
            _display_citation_answer(query, artifact)

    elif role.value == "coder":
        # Route to Phase 1 agent graph for code modification
        if llm_config and agent_config and tool_registry and repo_root:
            console.print("[yellow]Coder Agent: 正在修改代码...[/yellow]")
            graph = create_agent_graph(llm_config, agent_config, tool_registry, repo_root)
            state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "iteration": 0,
                "consecutive_failures": 0,
                "pending_tool_calls": [],
                "patches": [],
                "final_answer": None,
                "error": None,
            }
            run_agent_with_approval(graph, state)
        else:
            console.print("[yellow]Coder 需要 LLM 配置才能修改代码[/yellow]")

    elif role.value == "tester":
        console.print("[green]Tester Agent: 运行测试...[/green]")
        artifact = tester.run(query)
        console.print(f"[green]命令: {artifact.command}[/green]")
        console.print(f"[green]结果: {'通过' if artifact.success else '失败'} "
                      f"({artifact.tests_passed} 通过, {artifact.tests_failed} 失败)[/green]")

    elif role.value == "verifier":
        console.print("[magenta]Verifier Agent: 审查代码...[/magenta]")
        artifact = verifier.run(query)
        status = "[green]通过[/green]" if artifact.approved else "[red]未通过[/red]"
        console.print(f"审查结果: {status}")
        if artifact.issues:
            for issue in artifact.issues:
                console.print(f"  [red]- {issue}[/red]")


def _llm_answer_with_rag(query, artifact, llm_config, agent_config, tool_registry, repo_root):
    """Use LLM to generate answer based on RAG retrieval results.

    Builds a context from RAG results and sends to LLM with a system prompt.
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from agent.llm_client import create_llm

    # Build context from RAG results
    context_parts = []
    for i, r in enumerate(artifact.results[:8], 1):
        s = r.chunk.slice
        context_parts.append(
            f"[{i}] {s.file_path}:{s.start_line}-{s.end_line} "
            f"{s.class_name}.{s.method_name}\n{s.content}"
        )
    context = "\n\n".join(context_parts)

    rag_prompt = f"""\
你是一个 Java 代码助手。根据检索到的代码片段回答用户问题。

要求：
- 基于下面的代码片段回答，不要编造不存在的代码
- 回答必须引用真实的文件路径和行号，格式：`文件路径:行号`
- 用简体中文回答
- 代码标识符保留英文

检索到的代码：

{context}
"""

    llm = create_llm(llm_config)
    messages = [
        SystemMessage(content=rag_prompt),
        HumanMessage(content=query),
    ]

    console.print("[dim]正在生成回答...[/dim]")
    try:
        response = llm.invoke(messages)
        console.print(Panel(
            Markdown(response.content),
            title="回答",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[red]LLM 调用失败: {e}[/red]")
        # Fallback to citation answer
        _display_citation_answer(query, artifact)


def _display_citation_answer(query: str, artifact):
    """Build and display a citation-based answer from RAG results (fallback)."""
    lines = []
    lines.append(f"**问题**: {query}\n")
    lines.append(f"**检索**: {artifact.analysis}\n")
    lines.append("**相关代码**:\n")

    for i, r in enumerate(artifact.results[:5], 1):
        s = r.chunk.slice
        lines.append(f"{i}. `{s.file_path}:{s.start_line}` — **{s.class_name}.{s.method_name}**")
        preview = s.content.strip().split("\n")[0][:120]
        lines.append(f"   `{preview}`\n")

    console.print(Panel(
        "\n".join(lines),
        title="回答",
        border_style="green",
    ))


# ============================================================
# Phase 1: Agent Graph Runner (for code modification tasks)
# ============================================================

def run_agent_with_approval(graph, state: AgentState):
    """Run agent graph with proper interrupt/resume handling."""
    current_input = state
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        result = graph.invoke(current_input, config=config)

        messages = result.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        display_tool_call(tc["name"], tc.get("args", {}))
            elif isinstance(msg, ToolMessage):
                content = msg.content
                status = ToolStatus.SUCCESS
                if "[error]" in content.lower():
                    status = ToolStatus.ERROR
                elif "[denied]" in content.lower():
                    status = ToolStatus.DENIED
                elif "[execution_error]" in content.lower():
                    status = ToolStatus.EXECUTION_ERROR
                display_tool_result(msg.name, status, content)

        if result.get("final_answer") or result.get("error"):
            display_final_answer(
                result.get("final_answer"),
                result.get("error"),
                result.get("patches", []),
            )
            return result

        pending = result.get("pending_tool_calls", [])
        if pending:
            from agent.security.approval import needs_approval, build_approval_request
            if needs_approval(pending):
                approval_req = build_approval_request(pending)
                approved = display_approval_request({
                    "summary": approval_req.summary,
                    "tool_calls": [tc.model_dump() for tc in pending],
                })
                if approved:
                    current_input = Command(resume={"approved": True})
                else:
                    current_input = Command(resume={"approved": False, "reason": "用户拒绝操作"})
                continue
            else:
                current_input = Command(resume={"approved": True})
                continue

        if messages:
            last_ai = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    last_ai = msg
                    break
            if last_ai:
                console.print(Panel(
                    Markdown(last_ai.content),
                    title="回答",
                    border_style="green",
                ))

        return result


def build_mcp_adapter(repo_root: Path):
    """Create the stdio MCP adapter used by the Researcher main path."""
    from agent.agents.permission import AgentRole, PermissionManager
    from agent.mcp.client import MCPToolAdapter, create_mcp_client

    config = load_mcp_config()
    if not config.enabled:
        return None
    if config.transport != "stdio":
        raise ValueError("Phase 3A仅支持MCP stdio transport")

    env = os.environ.copy()
    env["AGENT_REPO_ROOT"] = str(repo_root)
    env["MCP_AGENT_ROLE"] = AgentRole.RESEARCHER.value
    client = create_mcp_client(
        [sys.executable, "-m", "agent.mcp.server", "--repo-root", str(repo_root), "--role", "researcher"],
        env=env,
        cwd=Path.cwd(),
    )
    return MCPToolAdapter(client, PermissionManager(), AgentRole.RESEARCHER)


def run_workflow_with_approval(workflow, state, session_manager: SessionManager, session_id: str):
    """Run the integrated parent graph through completion or approval interrupts."""
    config = session_manager.get_thread_config(session_id)
    current_input = state

    while True:
        result = workflow.invoke(current_input, config=config)
        interrupts = result.get("__interrupt__", [])
        if interrupts:
            payload = interrupts[0].value
            approved = display_approval_request(payload)
            current_input = Command(resume={
                "approved": approved,
                "reason": "" if approved else "用户拒绝操作",
            })
            continue

        search_artifact = result.get("search_artifact")
        if search_artifact:
            display_search_results(search_artifact.results)
            console.print(f"[dim]{search_artifact.analysis}[/dim]")

        test_result = result.get("test_result")
        if test_result:
            display_tool_result(
                "run_tests",
                ToolStatus.SUCCESS if test_result.success else ToolStatus.EXECUTION_ERROR,
                test_result.stdout or test_result.stderr,
            )

        review = result.get("review_result")
        if review:
            color = "green" if review.approved else "red"
            console.print(Panel(
                review.summary + ("\n" + "\n".join(review.issues) if review.issues else ""),
                title="Verifier",
                border_style=color,
            ))

        display_final_answer(
            result.get("final_answer"),
            result.get("error"),
            result.get("patches", []),
        )
        return result


# ============================================================
# Main
# ============================================================

def main():
    """Main CLI entry point."""
    configure_logging()
    display_welcome()

    console.print("[dim]正在初始化运行时和代码索引...[/dim]")
    try:
        runtime = create_app_runtime()
    except Exception as e:
        console.print(f"[red]运行时初始化失败: {e}[/red]")
        sys.exit(1)

    app_service = runtime.service
    session_id = runtime.session_id
    console.print(f"[dim]仓库: {runtime.repo_root}[/dim]")
    console.print(
        f"[dim]模型: {runtime.llm_config.provider.value} / {runtime.llm_config.model}[/dim]"
    )
    console.print(f"[dim]已注册 {runtime.tool_count} 个工具[/dim]")
    console.print(
        f"[dim]索引完成: {runtime.chunk_count} chunks, {runtime.search_type}, "
        f"{runtime.index_duration_seconds:.1f}s[/dim]"
    )
    console.print(f"[dim]Multi-Agent主工作流已就绪，Session: {session_id}[/dim]")

    console.print()

    # Conversation loop
    while True:
        try:
            user_input = console.input("\n[bold green]You> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]再见！[/dim]")
            break
        if user_input.lower() in ("new", "新会话", "新任务"):
            session_id = app_service.create_session()
            console.print(f"[dim]已创建新Session: {session_id}[/dim]")
            continue

        try:
            run_app_service_with_approval(app_service, session_id, user_input)
        except Exception as e:
            console.print(f"[red]Agent执行错误: {e}[/red]")
            import traceback
            traceback.print_exc()

    app_service.close()


if __name__ == "__main__":
    main()
