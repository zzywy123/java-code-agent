"""Supervisor Agent: routes tasks to appropriate sub-agents.

The Supervisor analyzes the user's request and decides which sub-agent
should handle it. It can also orchestrate multi-step workflows by
sequencing multiple agents.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.agents.permission import AgentRole

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """\
你是一个任务路由专家。根据用户的请求，决定应该由哪个 Agent 来处理。

可用的 Agent：
- researcher: 搜索和分析代码，回答代码问题（只读）
- coder: 修改代码、创建文件、应用补丁（需要审批）
- tester: 运行测试和构建命令（受限命令）
- verifier: 审查代码质量，验证修改是否合格（只读，可驳回）

路由规则：
1. 代码问答 → researcher
2. 代码修改 → coder（完成后 → tester → verifier）
3. 运行测试 → tester
4. 代码审查 → verifier
5. 复合任务 → 先 researcher 分析，再 coder 修改

输出格式（严格 JSON）：
{"agent": "researcher|coder|tester|verifier", "reason": "简短原因"}

不要输出其他内容，只输出 JSON。
"""


class SupervisorAgent:
    """Supervisor that routes tasks to sub-agents.

    Uses LLM to analyze user requests and select the appropriate agent.
    Falls back to researcher for read-only queries.
    """

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm

    def route(self, task: str, context: dict[str, Any] | None = None) -> AgentRole:
        """Route a task to the appropriate agent.

        Args:
            task: The user's request
            context: Optional context (e.g., previous agent results)

        Returns:
            The AgentRole to handle the task
        """
        explicit_route = self._explicit_action_route(task)
        if explicit_route is not None:
            return explicit_route

        if not self._llm:
            return self._rule_based_route(task)

        try:
            return self._llm_route(task, context)
        except Exception as e:
            logger.warning("LLM routing failed, using rule-based: %s", e)
            return self._rule_based_route(task)

    def should_continue(self, state: dict[str, Any]) -> bool:
        """Decide whether to continue the multi-agent workflow.

        Returns True if more agents need to run, False if done.
        """
        # Check if verifier approved
        artifacts = state.get("agent_artifacts", [])
        for art in artifacts:
            if hasattr(art, "artifact_type") and art.artifact_type == "review":
                return not art.approved  # Stop if approved, continue if rejected

        # Check if we have a code change but no review yet
        has_code_change = any(
            hasattr(a, "artifact_type") and a.artifact_type == "code_change"
            for a in artifacts
        )
        has_review = any(
            hasattr(a, "artifact_type") and a.artifact_type == "review"
            for a in artifacts
        )

        if has_code_change and not has_review:
            return True  # Need to test and verify

        return False

    def _llm_route(self, task: str, context: dict[str, Any] | None) -> AgentRole:
        """Use LLM to route the task."""
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"用户请求：{task}"),
        ]

        response = self._llm.invoke(messages)
        content = response.content.strip()

        import json
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(content[start:end + 1])
            agent_name = data.get("agent", "researcher")
            role_map = {
                "researcher": AgentRole.RESEARCHER,
                "coder": AgentRole.CODER,
                "tester": AgentRole.TESTER,
                "verifier": AgentRole.VERIFIER,
            }
            return role_map.get(agent_name, AgentRole.RESEARCHER)

        return AgentRole.RESEARCHER

    def _rule_based_route(self, task: str) -> AgentRole:
        """Simple keyword-based routing."""
        task_lower = task.lower()

        if any(marker in task_lower for marker in ["为什么", "怎么", "如何", "是什么", "哪里", "是否", "吗", "？", "?", " why ", " how ", " what "]):
            return AgentRole.RESEARCHER

        # Review intent must win over incidental words such as "修改".
        if any(kw in task_lower for kw in ["审查", "review", "代码评审", "verify", "验证修改"]):
            return AgentRole.VERIFIER

        # Write intent takes precedence for compound requests such as
        # "fix the bug and run tests".
        if any(kw in task_lower for kw in ["修改", "修复", "创建", "写入", "新增", "删除", "重构", "patch", "fix", "create", "update", "implement", "refactor"]):
            return AgentRole.CODER

        # Review operations (check first — "审查修改" should route to verifier)
        if any(kw in task_lower for kw in ["审查", "review", "检查代码", "verify", "验证"]):
            return AgentRole.VERIFIER

        # Test operations
        if any(kw in task_lower for kw in ["测试", "运行测试", "构建", "test", "run test", "build", "maven", "gradle"]):
            return AgentRole.TESTER

        # Default: researcher
        return AgentRole.RESEARCHER

    def _explicit_action_route(self, task: str) -> AgentRole | None:
        """Protect explicit side-effect requests from ambiguous LLM routing."""
        import re

        normalized = task.strip().lower()
        if re.search(r"(?<![\w-])git\s+(?:diff|status|log)(?=\s|$)", normalized):
            return AgentRole.RESEARCHER
        question_markers = (
            "为什么", "怎么", "如何", "是什么", "哪里", "是否", "吗", "？", "?",
            " why ", " how ", " what ",
        )
        write_terms = (
            "修改", "修复", "创建", "写入", "新增", "删除", "重构", "实现",
            "fix", "modify", "update", "create", "write", "add", "remove",
            "refactor", "implement", "patch",
        )
        imperative_prefixes = (
            "请", "帮我", "替我", "需要你", "给我", "直接",
            "fix ", "modify ", "update ", "create ", "write ", "add ",
            "remove ", "refactor ", "implement ", "patch ",
        )

        has_write_term = any(term in normalized for term in write_terms)
        looks_imperative = normalized.startswith(imperative_prefixes) or bool(
            re.search(r"(?:请|帮我|替我|需要你).*(?:修改|修复|创建|新增|删除|重构|实现)", normalized)
        )
        looks_like_question = any(marker in f" {normalized} " for marker in question_markers)
        review_terms = ("审查", "代码评审", "review", "verify", "验证修改")
        if any(term in normalized for term in review_terms) and not looks_like_question:
            return AgentRole.VERIFIER

        if has_write_term and (looks_imperative or not looks_like_question):
            return AgentRole.CODER

        test_terms = ("运行测试", "执行测试", "跑测试", "构建项目", "run tests", "run test", "build")
        if any(term in normalized for term in test_terms) and not looks_like_question:
            return AgentRole.TESTER

        return None
