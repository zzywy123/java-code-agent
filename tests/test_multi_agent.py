"""Tests for Multi-Agent system: Supervisor, Researcher, Coder, Tester, Verifier.

Validates:
- Supervisor routing
- Agent permission enforcement
- Artifact creation and handoff
- Verifier approval/rejection
- Multi-step workflow
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.agents.artifacts import ArtifactFactory
from agent.agents.coder import CoderAgent
from agent.agents.permission import AgentRole, PermissionManager, PermissionViolationError
from agent.agents.researcher import ResearcherAgent
from agent.agents.supervisor import SupervisorAgent
from agent.agents.tester import TesterAgent
from agent.agents.verifier import VerifierAgent
from agent.models import (
    CodeChangeArtifact,
    ReviewArtifact,
    RAGResult,
    SearchArtifact,
    TestResultArtifact,
    ToolResult,
    ToolStatus,
)
from agent.tools.factory import create_tool_registry


@pytest.fixture
def permission_manager() -> PermissionManager:
    return PermissionManager()


@pytest.fixture
def mock_tool_registry() -> MagicMock:
    registry = MagicMock()
    registry.execute.return_value = MagicMock(
        status=MagicMock(value="success"),
        output="test output",
        metadata={"exit_code": 0},
    )
    return registry


# ============================================================
# Supervisor Tests
# ============================================================

class TestSupervisor:
    """Supervisor routing tests."""

    def test_route_code_question_to_researcher(self):
        supervisor = SupervisorAgent(llm=None)
        role = supervisor.route("OrderService 的 calculateTotal 有什么 bug？")
        assert role == AgentRole.RESEARCHER

    def test_route_modify_request_to_coder(self):
        supervisor = SupervisorAgent(llm=None)
        role = supervisor.route("修复 calculateTotal 方法的 bug")
        assert role == AgentRole.CODER

    def test_route_test_request_to_tester(self):
        supervisor = SupervisorAgent(llm=None)
        role = supervisor.route("运行测试")
        assert role == AgentRole.TESTER

    def test_route_review_to_verifier(self):
        supervisor = SupervisorAgent(llm=None)
        role = supervisor.route("审查代码修改")
        assert role == AgentRole.VERIFIER

    def test_llm_routing(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = '{"agent": "coder", "reason": "needs fix"}'
        supervisor = SupervisorAgent(llm=mock_llm)
        role = supervisor.route("fix the bug")
        assert role == AgentRole.CODER

    def test_llm_routing_fallback(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        supervisor = SupervisorAgent(llm=mock_llm)
        role = supervisor.route("run tests")
        assert role == AgentRole.TESTER

    def test_explicit_fix_and_test_cannot_be_downgraded_to_researcher(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = '{"agent":"researcher","reason":"analyze first"}'
        supervisor = SupervisorAgent(llm=mock_llm)

        role = supervisor.route("请修复 OrderService.calculateTotal 的 Bug，并运行测试")

        assert role == AgentRole.CODER
        mock_llm.invoke.assert_not_called()

    def test_question_about_a_fix_remains_read_only(self):
        supervisor = SupervisorAgent(llm=None)
        role = supervisor.route("这个修复为什么要使用 getSubtotal？")
        assert role == AgentRole.RESEARCHER

    def test_explicit_git_read_routes_without_llm(self):
        mock_llm = MagicMock()
        supervisor = SupervisorAgent(llm=mock_llm)

        role = supervisor.route("git diff")

        assert role == AgentRole.RESEARCHER
        mock_llm.invoke.assert_not_called()


# ============================================================
# Permission Tests
# ============================================================

class TestPermissions:
    """Agent permission enforcement."""

    def test_researcher_can_read(self, permission_manager: PermissionManager):
        assert permission_manager.can_use_tool(AgentRole.RESEARCHER, "read_file")
        assert permission_manager.can_use_tool(AgentRole.RESEARCHER, "search_code")

    def test_researcher_cannot_write(self, permission_manager: PermissionManager):
        assert not permission_manager.can_use_tool(AgentRole.RESEARCHER, "apply_patch")

    def test_coder_can_write(self, permission_manager: PermissionManager):
        assert permission_manager.can_use_tool(AgentRole.CODER, "apply_patch")
        assert permission_manager.can_use_tool(AgentRole.CODER, "undo_patch")

    def test_tester_can_execute(self, permission_manager: PermissionManager):
        assert permission_manager.can_use_tool(AgentRole.TESTER, "run_tests")

    def test_tester_cannot_write(self, permission_manager: PermissionManager):
        assert not permission_manager.can_use_tool(AgentRole.TESTER, "apply_patch")

    def test_verifier_cannot_write(self, permission_manager: PermissionManager):
        assert not permission_manager.can_use_tool(AgentRole.VERIFIER, "apply_patch")

    def test_verifier_cannot_execute(self, permission_manager: PermissionManager):
        assert not permission_manager.can_use_tool(AgentRole.VERIFIER, "run_tests")

    def test_assert_tool_allowed_raises(self, permission_manager: PermissionManager):
        with pytest.raises(PermissionViolationError):
            permission_manager.assert_tool_allowed(AgentRole.RESEARCHER, "apply_patch")

    def test_filter_tool_calls(self, permission_manager: PermissionManager):
        from agent.models import ToolCallRequest
        calls = [
            ToolCallRequest(id="1", name="read_file", arguments={}),
            ToolCallRequest(id="2", name="apply_patch", arguments={}),
        ]
        allowed, denied = permission_manager.filter_tool_calls(AgentRole.RESEARCHER, calls)
        assert len(allowed) == 1
        assert len(denied) == 1


# ============================================================
# Artifact Tests
# ============================================================

class TestArtifacts:
    """Artifact creation and parsing."""

    def test_create_search_artifact(self):
        art = ArtifactFactory.create_search_artifact(
            query="test",
            analysis="found results",
            direct_answer="git output",
            render_hint="diff",
        )
        assert art.artifact_type == "search_results"
        assert art.query == "test"
        assert art.direct_answer == "git output"
        assert art.render_hint == "diff"

    def test_create_code_change_artifact(self):
        art = ArtifactFactory.create_code_change_artifact(
            description="fix bug", affected_files=["A.java"],
        )
        assert art.artifact_type == "code_change"

    def test_create_test_result_artifact(self):
        art = ArtifactFactory.create_test_result_artifact(
            command="mvn test", exit_code=0, tests_passed=5,
        )
        assert art.artifact_type == "test_result"
        assert art.success is True

    def test_create_review_artifact(self):
        art = ArtifactFactory.create_review_artifact(
            approved=True, summary="looks good",
        )
        assert art.artifact_type == "review"
        assert art.approved is True

    def test_parse_artifact(self):
        data = {"artifact_type": "review", "approved": False, "issues": ["bug"]}
        art = ArtifactFactory.parse(data)
        assert isinstance(art, ReviewArtifact)
        assert art.approved is False

    def test_to_dict(self):
        art = ArtifactFactory.create_review_artifact(approved=True)
        d = ArtifactFactory.to_dict(art)
        assert d["artifact_type"] == "review"
        assert d["approved"] is True


# ============================================================
# Verifier Tests
# ============================================================

class TestVerifier:
    """Verifier approval/rejection."""

    def test_verifier_approves_good_code(self, mock_tool_registry, permission_manager):
        verifier = VerifierAgent(mock_tool_registry, permission_manager, llm=None)
        context = {
            "agent_artifacts": [
                ArtifactFactory.create_code_change_artifact("fix"),
                ArtifactFactory.create_test_result_artifact("mvn test", 0, tests_passed=5),
            ]
        }
        result = verifier.run("review", context)
        assert result.approved is True

    def test_verifier_rejects_failing_tests(self, mock_tool_registry, permission_manager):
        verifier = VerifierAgent(mock_tool_registry, permission_manager, llm=None)
        context = {
            "agent_artifacts": [
                ArtifactFactory.create_test_result_artifact("mvn test", 1, tests_failed=2),
            ]
        }
        result = verifier.run("review", context)
        assert result.approved is False
        assert len(result.issues) > 0

    def test_verifier_rejects_no_context(self, mock_tool_registry, permission_manager):
        mock_tool_registry.execute.return_value.output = "没有变更"
        verifier = VerifierAgent(mock_tool_registry, permission_manager, llm=None)
        result = verifier.run("review")
        assert result.approved is False

    def test_verifier_can_review_real_diff_without_code_artifact(
        self, mock_tool_registry, permission_manager
    ):
        mock_tool_registry.execute.return_value.output = (
            "Git Diff:\ndiff --git a/A.java b/A.java\n-old\n+new"
        )
        verifier = VerifierAgent(mock_tool_registry, permission_manager, llm=None)

        result = verifier.run("review current workspace")

        assert result.approved is True


# ============================================================
# Researcher Tests
# ============================================================

class TestResearcher:
    """Researcher agent tests."""

    def test_researcher_returns_artifact(self, mock_tool_registry, permission_manager):
        researcher = ResearcherAgent(mock_tool_registry, permission_manager, agentic_rag=None)
        result = researcher.run("what does OrderService do?")
        assert isinstance(result, SearchArtifact)

    def test_repository_bug_audit_reads_key_sources_instead_of_searching_bug(
        self, permission_manager
    ):
        registry = MagicMock()

        def execute(name, tool_call_id, **arguments):
            if name == "list_files":
                return ToolResult(
                    tool_call_id=tool_call_id,
                    name=name,
                    status=ToolStatus.SUCCESS,
                    output=(
                        "目录 . 内容 (4 项):\n"
                        "📄 src/main/java/demo/Order.java\n"
                        "📄 src/main/java/demo/OrderController.java\n"
                        "📄 src/main/java/demo/OrderRepository.java\n"
                        "📄 src/main/java/demo/OrderService.java"
                    ),
                )
            if name == "read_file":
                path = arguments["path"]
                return ToolResult(
                    tool_call_id=tool_call_id,
                    name=name,
                    status=ToolStatus.SUCCESS,
                    output=f"文件: {path} (第 1-2 行)\n1 | class Example {{}}",
                )
            raise AssertionError(f"仓库审查不应调用 {name}")

        registry.execute.side_effect = execute
        rag = MagicMock()
        rag.retrieve.return_value = RAGResult(
            rounds_used=2,
            evidence_sufficient=False,
            degraded=True,
        )
        researcher = ResearcherAgent(registry, permission_manager, agentic_rag=rag)

        result = researcher.run("这个项目有什么bug？")

        called_tools = [call.kwargs["name"] for call in registry.execute.call_args_list]
        assert called_tools[0] == "list_files"
        assert "search_code" not in called_tools
        assert called_tools.count("read_file") == 4
        assert result.relevant_files[0].endswith("OrderService.java")
        assert any("OrderService.java" in item for item in result.tool_evidence)
        assert "扫描到 4 个 Java 文件" in result.analysis
        rag.retrieve.assert_called_once()

    def test_specific_bug_question_keeps_identifier_search(
        self, mock_tool_registry, permission_manager
    ):
        rag = MagicMock()
        rag.retrieve.return_value = RAGResult(rounds_used=1, degraded=True)
        researcher = ResearcherAgent(
            mock_tool_registry,
            permission_manager,
            agentic_rag=rag,
        )

        researcher.run("OrderService 有什么 bug？")

        mock_tool_registry.execute.assert_called_once_with(
            name="search_code",
            tool_call_id="researcher_evidence",
            query="OrderService",
            path=".",
            file_pattern="*.java",
        )

    def test_repository_bug_audit_reads_a_real_java_repository(
        self, tmp_repo, permission_manager
    ):
        rag = MagicMock()
        rag.retrieve.return_value = RAGResult(rounds_used=1, degraded=True)
        researcher = ResearcherAgent(
            create_tool_registry(tmp_repo),
            permission_manager,
            agentic_rag=rag,
        )

        result = researcher.run("检查这个工程中可能存在的问题")

        assert any(path.endswith("Hello.java") for path in result.relevant_files)
        assert any("class Hello" in evidence for evidence in result.tool_evidence)
        assert "读取 2 个关键实现" in result.analysis

    def test_mcp_is_primary_channel_for_direct_git_reads(
        self, mock_tool_registry, permission_manager
    ):
        mcp = MagicMock()
        mcp.call_tool_sync.return_value = ToolResult(
            tool_call_id="mcp_git_status",
            name="git_status",
            status=ToolStatus.SUCCESS,
            output="MCP Git Status",
        )
        researcher = ResearcherAgent(
            mock_tool_registry,
            permission_manager,
            agentic_rag=None,
            mcp_adapter=mcp,
        )

        result = researcher.run("git status")

        mcp.call_tool_sync.assert_called_once_with("git_status", {})
        mock_tool_registry.execute.assert_not_called()
        assert result.direct_answer == "MCP Git Status"
        assert "MCP" in result.analysis

    def test_mcp_search_output_becomes_downstream_tool_evidence(
        self, mock_tool_registry, permission_manager
    ):
        mcp = MagicMock()
        mcp.call_tool_sync.return_value = ToolResult(
            tool_call_id="mcp_search_code",
            name="search_code",
            status=ToolStatus.SUCCESS,
            output="src/OrderService.java:42: calculateTotal",
        )
        researcher = ResearcherAgent(
            mock_tool_registry,
            permission_manager,
            agentic_rag=None,
            mcp_adapter=mcp,
        )

        result = researcher.run("解释 OrderService.calculateTotal")

        mock_tool_registry.execute.assert_not_called()
        assert result.tool_evidence == ["src/OrderService.java:42: calculateTotal"]
        assert "MCP" in result.analysis

    def test_mcp_search_hit_skips_agentic_rag(
        self, mock_tool_registry, permission_manager
    ):
        mcp = MagicMock()
        mcp.call_tool_sync.return_value = ToolResult(
            tool_call_id="mcp_search_code",
            name="search_code",
            status=ToolStatus.SUCCESS,
            output="src/OrderService.java:42: calculateTotal",
            metadata={"match_count": 1},
        )
        rag = MagicMock()
        researcher = ResearcherAgent(
            mock_tool_registry,
            permission_manager,
            agentic_rag=rag,
            mcp_adapter=mcp,
        )

        result = researcher.run("解释 OrderService.calculateTotal")

        rag.retrieve.assert_not_called()
        assert "跳过多轮 RAG" in result.analysis
        assert result.tool_evidence
        assert any(
            call.args[0] == "read_file"
            for call in mcp.call_tool_sync.call_args_list
        )

    @pytest.mark.parametrize(
        ("request_text", "tool_name", "arguments"),
        [
            ("git diff", "git_diff", {"staged": False}),
            ("查看 git diff --staged", "git_diff", {"staged": True}),
            ("git diff -- src/Main.java", "git_diff", {"staged": False, "path": "src/Main.java"}),
            ("git status", "git_status", {}),
            ("git log -n 5", "git_log", {"count": 5}),
        ],
    )
    def test_explicit_git_read_calls_real_tool(
        self,
        request_text,
        tool_name,
        arguments,
        mock_tool_registry,
        permission_manager,
    ):
        rag = MagicMock()
        researcher = ResearcherAgent(mock_tool_registry, permission_manager, agentic_rag=rag)

        result = researcher.run(request_text)

        mock_tool_registry.execute.assert_called_once_with(
            name=tool_name,
            tool_call_id=f"researcher_{tool_name}",
            **arguments,
        )
        rag.retrieve.assert_not_called()
        assert result.direct_answer == "test output"
        assert result.render_hint == ("diff" if tool_name == "git_diff" else "text")

    def test_git_knowledge_question_is_not_executed(
        self,
        mock_tool_registry,
        permission_manager,
    ):
        researcher = ResearcherAgent(mock_tool_registry, permission_manager, agentic_rag=None)

        result = researcher.run("git diff 是什么？")

        assert result.direct_answer is None
        mock_tool_registry.execute.assert_called_once_with(
            name="search_code",
            tool_call_id="researcher_basic",
            query="git diff 是什么？",
        )


# ============================================================
# Coder Tests
# ============================================================

class TestCoder:
    """Coder agent tests."""

    def test_coder_returns_artifact(self, mock_tool_registry, permission_manager):
        coder = CoderAgent(mock_tool_registry, permission_manager)
        result = coder.run("fix the bug")
        assert isinstance(result, CodeChangeArtifact)


# ============================================================
# Tester Tests
# ============================================================

class TestTester:
    """Tester agent tests."""

    def test_tester_returns_artifact(self, mock_tool_registry, permission_manager):
        tester = TesterAgent(mock_tool_registry, permission_manager)
        result = tester.run("run tests")
        assert isinstance(result, TestResultArtifact)

    def test_tester_dispatches_gradle_from_build_file(
        self, tmp_path, mock_tool_registry, permission_manager
    ):
        (tmp_path / "build.gradle.kts").write_text("plugins {}", encoding="utf-8")
        mock_tool_registry.get.return_value.repo_root = tmp_path
        tester = TesterAgent(mock_tool_registry, permission_manager)

        tester.run("run tests")

        mock_tool_registry.execute.assert_called_once_with(
            name="run_tests",
            tool_call_id="tester_gradle",
            tool="gradle",
            goals=["test"],
        )
