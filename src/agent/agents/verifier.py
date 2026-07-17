"""Verifier Agent: code review and quality verification.

The Verifier can:
- Read code (read_file, search_code, list_files)
- View git diff (git_diff)
- Review changes and produce a verdict

Cannot modify any files or execute commands.
Can approve or reject changes — rejection triggers rework.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.agents.artifacts import ArtifactFactory
from agent.agents.permission import AgentRole, PermissionManager
from agent.models import (
    AgentArtifact,
    CodeChangeArtifact,
    ReviewArtifact,
    TestResultArtifact,
)
from agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

VERIFIER_PROMPT = """\
你是代码审查专家。审查代码修改的质量，判断是否应该批准。

审查标准：
1. 修改是否正确解决了问题
2. 是否引入了新的 bug 或安全风险
3. 代码风格是否一致
4. 测试是否充分

输出格式（严格 JSON）：
{"approved": true/false, "issues": ["issue1"], "suggestions": ["suggestion1"], "summary": "总结"}

不要输出其他内容，只输出 JSON。
"""


class VerifierAgent:
    """Agent for code review and verification.

    Role: VERIFIER (read-only, can approve/reject)
    Capabilities: search, read, review
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        permission_manager: PermissionManager,
        llm: ChatOpenAI | None = None,
    ) -> None:
        self._tools = tool_registry
        self._permissions = permission_manager
        self._role = AgentRole.VERIFIER
        self._llm = llm

    def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> ReviewArtifact:
        """Execute a review task.

        Args:
            task: What to review
            context: Should contain code_change and test_result artifacts

        Returns:
            ReviewArtifact with approval/rejection verdict
        """
        logger.info("Verifier: %s", task[:100])

        self._permissions.assert_tool_allowed(self._role, "git_diff")
        diff_result = self._tools.execute(
            name="git_diff",
            tool_call_id="verifier_diff",
            path="",
        )
        real_diff = ""
        if diff_result.status.value == "success" and diff_result.output != "没有变更":
            real_diff = diff_result.output

        # Extract artifacts from context
        code_change: CodeChangeArtifact | None = None
        test_result: TestResultArtifact | None = None

        if context:
            artifacts = context.get("agent_artifacts", [])
            for art in artifacts:
                if hasattr(art, "artifact_type"):
                    if art.artifact_type == "code_change":
                        code_change = art
                    elif art.artifact_type == "test_result":
                        test_result = art

        if not real_diff and code_change:
            # A non-Git target still has an auditable PatchRecord.
            real_diff = "\n".join(p.unified_diff for p in code_change.patches)

        # If tests failed, automatically reject
        if test_result and not test_result.success:
            return ArtifactFactory.create_review_artifact(
                approved=False,
                issues=[f"测试失败: {test_result.tests_failed} 个测试未通过"],
                suggestions=["修复失败的测试后重新提交"],
                summary=f"测试未通过（{test_result.tests_passed} 通过, {test_result.tests_failed} 失败）",
            )

        # Use LLM for code review if available
        if self._llm and (code_change or real_diff):
            return self._llm_review(task, code_change, test_result, real_diff)

        # Rule-based review
        return self._rule_based_review(code_change, test_result, real_diff)

    def approve(self, artifact: ReviewArtifact) -> bool:
        """Check if a review artifact represents approval."""
        return artifact.approved

    def _llm_review(
        self,
        task: str,
        code_change: CodeChangeArtifact | None,
        test_result: TestResultArtifact | None,
        real_diff: str = "",
    ) -> ReviewArtifact:
        """Use LLM for code review."""
        context_parts = [f"审查任务：{task}"]
        if code_change:
            context_parts.append(f"修改描述：{code_change.description}")
        if code_change and code_change.affected_files:
            context_parts.append(f"影响文件：{', '.join(code_change.affected_files)}")
        if test_result:
            context_parts.append(f"测试结果：{'通过' if test_result.success else '失败'}")
        if real_diff:
            context_parts.append(f"真实 Git Diff：\n{real_diff[:12000]}")

        context = "\n".join(context_parts)

        try:
            response = self._llm.invoke([
                SystemMessage(content=VERIFIER_PROMPT),
                HumanMessage(content=f"请审查以下修改：\n{context}"),
            ])

            import json
            content = response.content.strip()
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(content[start:end + 1])
                return ArtifactFactory.create_review_artifact(
                    approved=data.get("approved", False),
                    issues=data.get("issues", []),
                    suggestions=data.get("suggestions", []),
                    summary=data.get("summary", "LLM 审查完成"),
                )
        except Exception as e:
            logger.warning("LLM review failed: %s", e)

        return self._rule_based_review(code_change, test_result, real_diff)

    def _rule_based_review(
        self,
        code_change: CodeChangeArtifact | None,
        test_result: TestResultArtifact | None,
        real_diff: str = "",
    ) -> ReviewArtifact:
        """Simple rule-based review."""
        issues: list[str] = []
        suggestions: list[str] = []

        if not code_change and not real_diff:
            issues.append("工作区没有可审查的 Git Diff")
            return ArtifactFactory.create_review_artifact(
                approved=False, issues=issues,
                summary="缺少真实代码变更",
            )

        # Check if tests passed
        if test_result and test_result.success:
            suggestions.append("测试全部通过")
        elif test_result:
            issues.append(f"测试失败: {test_result.tests_failed} 个")

        # Check file count
        if code_change and len(code_change.affected_files) > 5:
            suggestions.append("修改涉及较多文件，建议分步提交")

        approved = len(issues) == 0
        summary = "审查通过" if approved else f"审查未通过: {'; '.join(issues)}"

        return ArtifactFactory.create_review_artifact(
            approved=approved,
            issues=issues,
            suggestions=suggestions,
            summary=summary,
        )
