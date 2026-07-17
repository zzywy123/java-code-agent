"""Researcher Agent: read-only code search and analysis.

The Researcher can:
- Search code (search_code, list_files, read_file)
- View git status and history (git_status, git_diff, git_log)
- Use Agentic RAG for deep code understanding

Cannot modify any files or execute commands.
"""

from __future__ import annotations

import logging
import re
import shlex
from typing import Any

from agent.agents.artifacts import ArtifactFactory
from agent.agents.permission import AgentRole, PermissionManager
from agent.models import AgentArtifact, SearchArtifact, SearchResult
from agent.rag.agentic_rag import AgenticRAG
from agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class ResearcherAgent:
    """Read-only agent for code search and analysis.

    Role: RESEARCHER (read-only)
    Capabilities: search, read, git, Agentic RAG
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        permission_manager: PermissionManager,
        agentic_rag: AgenticRAG | None = None,
        mcp_adapter: Any | None = None,
    ) -> None:
        self._tools = tool_registry
        self._permissions = permission_manager
        self._rag = agentic_rag
        self._mcp = mcp_adapter
        self._role = AgentRole.RESEARCHER

    def run(self, task: str, context: dict[str, Any] | None = None) -> SearchArtifact:
        """Execute a research task.

        Args:
            task: The research question
            context: Optional context from previous agents

        Returns:
            SearchArtifact with results and analysis
        """
        logger.info("Researcher: %s", task[:100])

        direct_request = self._parse_direct_git_request(task)
        if direct_request is not None:
            tool_name, arguments = direct_request
            self._permissions.assert_tool_allowed(self._role, tool_name)
            result = self._tools.execute(
                name=tool_name,
                tool_call_id=f"researcher_{tool_name}",
                **arguments,
            )
            return ArtifactFactory.create_search_artifact(
                query=task,
                analysis=f"直接调用只读工具 {tool_name}",
                direct_answer=result.output,
                render_hint="diff" if tool_name == "git_diff" else "text",
            )

        mcp_note = ""
        if self._mcp is not None:
            try:
                identifier = self._extract_identifier(task)
                mcp_result = self._mcp.call_tool_sync(
                    "search_code",
                    {"query": identifier, "path": ".", "file_pattern": "*.java"},
                )
                mcp_note = f"MCP search_code: {mcp_result.output[:200]}"
            except Exception as exc:
                logger.warning("MCP search_code failed; continuing with RAG: %s", exc)
                mcp_note = f"MCP降级: {exc}"

        # Use Agentic RAG if available
        if self._rag:
            rag_result = self._rag.retrieve(task)
            return ArtifactFactory.create_search_artifact(
                query=task,
                results=rag_result.sources,
                analysis=f"{mcp_note}\nRAG 检索 {rag_result.rounds_used} 轮，"
                         f"证据{'充分' if rag_result.evidence_sufficient else '不足'}"
                         f"{'（已降级）' if rag_result.degraded else ''}",
                relevant_files=[
                    r.chunk.slice.file_path for r in rag_result.sources[:5]
                ],
            )

        # Fallback: basic search using search_code tool
        return self._basic_search(task)

    @staticmethod
    def _extract_identifier(task: str) -> str:
        """Extract a likely Java identifier for exact MCP search."""
        import re

        candidates = re.findall(r"[A-Za-z_$][A-Za-z0-9_$.]*", task)
        return max(candidates, key=len) if candidates else task

    @staticmethod
    def _parse_direct_git_request(
        task: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """Map explicit command-like Git reads to the real repository tools."""
        match = re.search(r"(?<![\w-])git\s+(diff|status|log)(?=\s|$)", task, re.IGNORECASE)
        if match is None:
            return None

        normalized = task.strip().lower()
        question_markers = ("为什么", "如何", "怎么", "是什么", "？", "?")
        action_markers = ("查看", "显示", "执行", "运行", "给我", "看看", "请")
        if any(marker in normalized for marker in question_markers) and not any(
            marker in normalized for marker in action_markers
        ):
            return None

        command = match.group(1).lower()
        try:
            tokens = shlex.split(task[match.start():], posix=True)
        except ValueError:
            tokens = task[match.start():].split()
        arguments: dict[str, Any] = {}
        trailing = tokens[2:] if len(tokens) >= 2 else []

        if command == "diff":
            arguments["staged"] = any(
                token in {"--staged", "--cached"} for token in trailing
            ) or any(term in normalized for term in ("暂存区", "已暂存"))
            path = next(
                (
                    token for token in trailing
                    if not token.startswith("-")
                    and (
                        "/" in token
                        or "\\" in token
                        or token.lower().endswith(".java")
                    )
                ),
                "",
            )
            if path:
                arguments["path"] = path
            return "git_diff", arguments

        if command == "log":
            count = ResearcherAgent._parse_log_count(trailing)
            if count is not None:
                arguments["count"] = count
            return "git_log", arguments

        return "git_status", arguments

    @staticmethod
    def _parse_log_count(tokens: list[str]) -> int | None:
        for index, token in enumerate(tokens):
            match = re.fullmatch(r"-(?:n)?(\d+)", token)
            if match:
                return max(1, min(50, int(match.group(1))))
            match = re.fullmatch(r"--max-count=(\d+)", token)
            if match:
                return max(1, min(50, int(match.group(1))))
            if token == "-n" and index + 1 < len(tokens) and tokens[index + 1].isdigit():
                return max(1, min(50, int(tokens[index + 1])))
        return None

    def search_code(self, query: str) -> list[SearchResult]:
        """Search code using the search_code tool."""
        result = self._tools.execute(
            name="search_code",
            tool_call_id="researcher_search",
            query=query,
        )
        # Parse results into SearchResult objects
        # (simplified — in production would parse the tool output)
        return []

    def _basic_search(self, task: str) -> SearchArtifact:
        """Basic search without Agentic RAG."""
        # Extract key terms from the task
        terms = task.split()[:5]
        query = " ".join(terms)

        result = self._tools.execute(
            name="search_code",
            tool_call_id="researcher_basic",
            query=query,
        )

        return ArtifactFactory.create_search_artifact(
            query=task,
            analysis=f"基础搜索: {result.output[:200]}",
        )
