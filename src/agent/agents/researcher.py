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
from agent.models import AgentArtifact, SearchArtifact, SearchResult, ToolResult, ToolStatus
from agent.rag.agentic_rag import AgenticRAG
from agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

REPOSITORY_AUDIT_SCOPES = (
    "这个项目",
    "整个项目",
    "全项目",
    "项目中",
    "项目里",
    "这个工程",
    "整个工程",
    "工程中",
    "工程里",
    "代码库",
    "仓库",
    "全部代码",
    "所有代码",
)
REPOSITORY_AUDIT_INTENTS = (
    "bug",
    "问题",
    "缺陷",
    "风险",
    "审查",
    "检查",
    "review",
)
AUDIT_FILE_LIMIT = 5


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
            result, channel = self._execute_read_tool(
                tool_name,
                arguments,
                tool_call_id=f"researcher_{tool_name}",
            )
            return ArtifactFactory.create_search_artifact(
                query=task,
                analysis=f"通过 {channel} 调用只读工具 {tool_name}",
                direct_answer=result.output,
                render_hint="diff" if tool_name == "git_diff" else "text",
            )

        if self._is_repository_audit(task):
            return self._audit_repository(task)

        # Use Agentic RAG if available
        if self._rag:
            identifier = self._extract_identifier(task)
            tool_result, channel = self._execute_read_tool(
                "search_code",
                {"query": identifier, "path": ".", "file_pattern": "*.java"},
                tool_call_id="researcher_evidence",
            )
            tool_evidence = []
            if tool_result.status == ToolStatus.SUCCESS and tool_result.output:
                tool_evidence.append(
                    f"[{channel}:search_code query={identifier}]\n{tool_result.output[:6000]}"
                )
            if self._has_usable_search_output(tool_result):
                tool_evidence.extend(self._read_hit_context(tool_result.output))
                return ArtifactFactory.create_search_artifact(
                    query=task,
                    analysis=f"{channel} 一轮只读检索已命中，跳过多轮 RAG",
                    relevant_files=[],
                    tool_evidence=tool_evidence,
                )
            rag_result = self._rag.retrieve(task)
            return ArtifactFactory.create_search_artifact(
                query=task,
                results=rag_result.sources,
                analysis=f"{channel} 只读检索 + RAG {rag_result.rounds_used} 轮，"
                         f"证据{'充分' if rag_result.evidence_sufficient else '不足'}"
                         f"{'（已降级）' if rag_result.degraded else ''}",
                relevant_files=[
                    r.chunk.slice.file_path for r in rag_result.sources[:5]
                ],
                tool_evidence=tool_evidence,
            )

        # Fallback: basic search using search_code tool
        return self._basic_search(task)

    def _audit_repository(self, task: str) -> SearchArtifact:
        """Collect representative source evidence for a repository-wide review."""
        listing, list_channel = self._execute_read_tool(
            "list_files",
            {"path": ".", "pattern": "**/*.java"},
            tool_call_id="researcher_audit_files",
        )
        java_paths = self._parse_java_paths(listing.output)
        selected_paths = self._select_audit_files(java_paths)
        tool_evidence: list[str] = []

        for index, path in enumerate(selected_paths, 1):
            result, channel = self._execute_read_tool(
                "read_file",
                {"path": path, "start_line": 1, "end_line": 500},
                tool_call_id=f"researcher_audit_source_{index}",
            )
            if result.status == ToolStatus.SUCCESS and result.output:
                tool_evidence.append(
                    f"[{channel}:read_file repository-audit]\n{result.output[:8000]}"
                )

        rag_results: list[SearchResult] = []
        rag_analysis = "未启用 RAG"
        if self._rag is not None:
            audit_query = (
                f"{task} Java null exception validation calculation state transition "
                "transaction boundary concurrency authorization resource leak"
            )
            rag_result = self._rag.retrieve(audit_query)
            rag_results = rag_result.sources
            rag_analysis = (
                f"RAG {rag_result.rounds_used} 轮，"
                f"证据{'充分' if rag_result.evidence_sufficient else '不足'}"
                f"{'（已降级）' if rag_result.degraded else ''}"
            )

        return ArtifactFactory.create_search_artifact(
            query=task,
            results=rag_results,
            analysis=(
                f"{list_channel} 扫描到 {len(java_paths)} 个 Java 文件，"
                f"读取 {len(tool_evidence)} 个关键实现；{rag_analysis}"
            ),
            relevant_files=selected_paths,
            tool_evidence=tool_evidence,
        )

    @staticmethod
    def _is_repository_audit(task: str) -> bool:
        normalized = task.strip().lower()
        return any(scope in normalized for scope in REPOSITORY_AUDIT_SCOPES) and any(
            intent in normalized for intent in REPOSITORY_AUDIT_INTENTS
        )

    @staticmethod
    def _parse_java_paths(listing: str) -> list[str]:
        paths: list[str] = []
        for line in listing.splitlines():
            match = re.match(r"^\s*(?:📄\s+)?([^\r\n]+\.java)\s*$", line)
            if match:
                paths.append(match.group(1).strip().replace("\\", "/"))
        return list(dict.fromkeys(paths))

    @staticmethod
    def _select_audit_files(paths: list[str]) -> list[str]:
        def priority(path: str) -> tuple[int, str]:
            lowered = path.lower()
            name = lowered.rsplit("/", 1)[-1]
            score = 50 if "/src/main/" in f"/{lowered}" else 0
            if "/src/test/" in f"/{lowered}" or name.endswith("test.java"):
                score -= 50
            weights = (
                ("service", 60),
                ("controller", 50),
                ("security", 45),
                ("auth", 45),
                ("repository", 30),
                ("handler", 25),
                ("manager", 25),
                ("config", 10),
                ("application", -10),
            )
            score += sum(weight for marker, weight in weights if marker in name)
            return -score, lowered

        return sorted(paths, key=priority)[:AUDIT_FILE_LIMIT]

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
        self._execute_read_tool(
            "search_code",
            {"query": query},
            tool_call_id="researcher_search",
        )
        # Parse results into SearchResult objects
        # (simplified — in production would parse the tool output)
        return []

    def _basic_search(self, task: str) -> SearchArtifact:
        """Basic search without Agentic RAG."""
        # Extract key terms from the task
        terms = task.split()[:5]
        query = " ".join(terms)

        result, channel = self._execute_read_tool(
            "search_code",
            {"query": query},
            tool_call_id="researcher_basic",
        )

        return ArtifactFactory.create_search_artifact(
            query=task,
            analysis=f"通过 {channel} 完成基础搜索",
            tool_evidence=[result.output[:6000]] if result.output else [],
        )

    def _execute_read_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tool_call_id: str,
    ) -> tuple[ToolResult, str]:
        """Prefer MCP for read tools and fall back to the local registry."""
        self._permissions.assert_tool_allowed(self._role, name)
        if self._mcp is not None:
            try:
                result = self._mcp.call_tool_sync(name, arguments)
                if result.status == ToolStatus.SUCCESS:
                    return result, "MCP"
                logger.warning(
                    "MCP %s returned %s; falling back to local tools",
                    name,
                    result.status.value,
                )
            except Exception as exc:
                logger.warning("MCP %s failed; falling back to local tools: %s", name, exc)

        return (
            self._tools.execute(
                name=name,
                tool_call_id=tool_call_id,
                **arguments,
            ),
            "Local ToolRegistry",
        )

    @staticmethod
    def _has_usable_search_output(result: ToolResult) -> bool:
        """Recognize a real search hit without another model call."""
        if result.status != ToolStatus.SUCCESS:
            return False
        match_count = result.metadata.get("match_count")
        if isinstance(match_count, int):
            return match_count > 0
        output = result.output.strip().lower()
        if not output:
            return False
        no_match_markers = ("未找到匹配", "没有找到", "no matches", "no match", "not found")
        return not any(marker in output for marker in no_match_markers)

    def _read_hit_context(self, search_output: str) -> list[str]:
        """Read bounded source context for up to two exact search hits."""
        evidence: list[str] = []
        seen_paths: set[str] = set()
        for match in re.finditer(
            r"^\s*([^:\r\n]+\.java):(\d+):",
            search_output,
            flags=re.MULTILINE | re.IGNORECASE,
        ):
            path = match.group(1).strip()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            line_number = int(match.group(2))
            result, channel = self._execute_read_tool(
                "read_file",
                {
                    "path": path,
                    "start_line": max(1, line_number - 25),
                    "end_line": line_number + 50,
                },
                tool_call_id=f"researcher_context_{len(seen_paths)}",
            )
            if result.status == ToolStatus.SUCCESS and result.output:
                evidence.append(f"[{channel}:read_file]\n{result.output[:8000]}")
            if len(seen_paths) >= 2:
                break
        return evidence
