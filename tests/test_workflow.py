"""Integrated parent-graph tests for coding, testing and verification."""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent.config import AgentConfig, LLMConfig, MemoryConfig, RAGConfig, WorkflowConfig
from agent.models import PatchRecord, ToolResult, ToolStatus
from agent.session import SessionManager
from agent.tools.base import BaseTool, ToolRegistry
from agent.workflow import create_workflow, initial_workflow_state


class ScriptedLLM:
    def __init__(self) -> None:
        self.review_calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        text = "\n".join(str(message.content) for message in messages)
        if "任务路由专家" in text:
            return AIMessage(content='{"agent":"coder","reason":"需要修改"}')
        if "检索查询改写专家" in text:
            return AIMessage(content='["calculateTotal", "OrderService quantity"]')
        if "证据评估专家" in text:
            return AIMessage(content='{"sufficient":false,"confidence":0.2,"reason":"无索引"}')
        if "代码审查专家" in text:
            self.review_calls += 1
            return AIMessage(content='{"approved":true,"issues":[],"suggestions":[],"summary":"真实Diff和测试通过"}')
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="Patch已应用")
        return AIMessage(
            content="",
            tool_calls=[{
                "id": "patch-1",
                "name": "apply_patch",
                "args": {"path": "A.java", "unified_diff": "@@ -1 +1 @@\n-old\n+new"},
            }],
        )


class RejectOnceLLM(ScriptedLLM):
    def invoke(self, messages):
        text = "\n".join(str(message.content) for message in messages)
        if "代码审查专家" in text:
            self.review_calls += 1
            if self.review_calls == 1:
                return AIMessage(content='{"approved":false,"issues":["缺少边界处理"],"suggestions":[],"summary":"需要返工"}')
            return AIMessage(content='{"approved":true,"issues":[],"suggestions":[],"summary":"返工通过"}')
        if any(
            isinstance(message, HumanMessage) and "Verifier意见" in str(message.content)
            for message in messages
        ) and not isinstance(messages[-1], ToolMessage):
            return AIMessage(
                content="",
                tool_calls=[{
                    "id": "patch-rework",
                    "name": "apply_patch",
                    "args": {"path": "A.java", "unified_diff": "@@ -1 +1 @@\n-old\n+new"},
                }],
            )
        return super().invoke(messages)


class AlreadyFixedLLM(ScriptedLLM):
    def invoke(self, messages):
        text = "\n".join(str(message.content) for message in messages)
        if "检索查询改写专家" in text:
            return AIMessage(content='["OrderService calculateTotal", "OrderItem getSubtotal"]')
        if "证据评估专家" in text:
            return AIMessage(content='{"sufficient":true,"confidence":0.9,"reason":"symbols found"}')
        if "代码审查专家" in text:
            self.review_calls += 1
            return AIMessage(content='{"approved":true,"issues":[],"suggestions":[],"summary":"代码已正确且测试通过"}')
        return AIMessage(content="代码已经使用 getSubtotal，无需修改")


class EmptySearch:
    def search(self, query, top_k=10):
        return []


class DummyTool(BaseTool):
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self, repo_root: Path, name: str, result_factory):
        super().__init__(repo_root)
        self.name = name
        self.description = name
        self._result_factory = result_factory

    def execute(self, **kwargs):
        return self._result_factory(kwargs["tool_call_id"])


def build_registry(tmp_path: Path) -> ToolRegistry:
    registry = ToolRegistry()
    patch = PatchRecord(
        file_path=str(tmp_path / "A.java"),
        content_hash_before="0" * 64,
        content_hash_after="1" * 64,
        unified_diff="@@ -1 +1 @@\n-old\n+new",
    )
    for name in ("list_files", "read_file", "search_code", "git_status", "git_log"):
        registry.register(DummyTool(tmp_path, name, lambda call_id, n=name: ToolResult(
            tool_call_id=call_id, name=n, status=ToolStatus.SUCCESS, output="ok",
        )))
    registry.register(DummyTool(tmp_path, "apply_patch", lambda call_id: ToolResult(
        tool_call_id=call_id,
        name="apply_patch",
        status=ToolStatus.SUCCESS,
        output="patched",
        metadata={"patch_record": patch.model_dump(mode="json")},
    )))
    registry.register(DummyTool(tmp_path, "undo_patch", lambda call_id: ToolResult(
        tool_call_id=call_id, name="undo_patch", status=ToolStatus.SUCCESS, output="undone",
    )))
    registry.register(DummyTool(tmp_path, "run_tests", lambda call_id: ToolResult(
        tool_call_id=call_id,
        name="run_tests",
        status=ToolStatus.SUCCESS,
        output="Tests run: 1, Failures: 0",
        metadata={"exit_code": 0},
    )))
    registry.register(DummyTool(tmp_path, "git_diff", lambda call_id: ToolResult(
        tool_call_id=call_id,
        name="git_diff",
        status=ToolStatus.SUCCESS,
        output="diff --git a/A.java b/A.java\n-old\n+new",
    )))
    return registry


def test_coding_workflow_interrupts_then_tests_and_verifies(tmp_path):
    llm = ScriptedLLM()
    session = SessionManager(MemoryConfig(
        checkpoint_dir=str(tmp_path / "cp"),
        long_term_persist_dir=str(tmp_path / "memory"),
    ), llm=llm)
    session_id = session.create_session()
    graph = create_workflow(
        llm=llm,
        llm_config=LLMConfig(provider="ollama"),
        agent_config=AgentConfig(),
        workflow_config=WorkflowConfig(max_rework=1),
        rag_config=RAGConfig(),
        tool_registry=build_registry(tmp_path),
        search_engine=EmptySearch(),
        session_manager=session,
        repo_root=tmp_path,
    )
    config = session.get_thread_config(session_id)

    interrupted = graph.invoke(initial_workflow_state("修复calculateTotal", session_id), config)
    assert interrupted["__interrupt__"]
    result = graph.invoke(Command(resume={"approved": True}), config)

    assert result["test_result"].success is True
    assert result["review_result"].approved is True
    assert len(result["patches"]) == 1
    assert "审查通过" in result["final_answer"]
    session.close()


def test_verifier_rejection_returns_to_coder(tmp_path):
    llm = RejectOnceLLM()
    session = SessionManager(MemoryConfig(
        checkpoint_dir=str(tmp_path / "cp"),
        long_term_persist_dir=str(tmp_path / "memory"),
    ), llm=llm)
    session_id = session.create_session()
    graph = create_workflow(
        llm=llm,
        llm_config=LLMConfig(provider="ollama"),
        agent_config=AgentConfig(),
        workflow_config=WorkflowConfig(max_rework=1),
        rag_config=RAGConfig(),
        tool_registry=build_registry(tmp_path),
        search_engine=EmptySearch(),
        session_manager=session,
        repo_root=tmp_path,
    )
    config = session.get_thread_config(session_id)
    first = graph.invoke(initial_workflow_state("修复calculateTotal", session_id), config)
    second = graph.invoke(Command(resume={"approved": True}), config)
    assert second["__interrupt__"]
    result = graph.invoke(Command(resume={"approved": True}), config)

    assert result["rework_count"] == 1
    assert llm.review_calls == 2
    assert result["review_result"].approved is True
    session.close()


def test_already_fixed_request_still_runs_tests_without_patch(tmp_path):
    llm = AlreadyFixedLLM()
    session = SessionManager(MemoryConfig(
        checkpoint_dir=str(tmp_path / "cp"),
        long_term_persist_dir=str(tmp_path / "memory"),
    ), llm=llm)
    session_id = session.create_session()
    graph = create_workflow(
        llm=llm,
        llm_config=LLMConfig(provider="ollama"),
        agent_config=AgentConfig(),
        workflow_config=WorkflowConfig(max_rework=1),
        rag_config=RAGConfig(),
        tool_registry=build_registry(tmp_path),
        search_engine=EmptySearch(),
        session_manager=session,
        repo_root=tmp_path,
    )

    result = graph.invoke(
        initial_workflow_state(
            "请修复 OrderService.calculateTotal 的 Bug，并运行测试",
            session_id,
        ),
        session.get_thread_config(session_id),
    )

    assert result["patches"] == []
    assert result["test_result"].success is True
    assert result["review_result"].approved is True
    assert "测试通过" in result["final_answer"]
    session.close()


def test_git_diff_returns_real_tool_output_without_rag_answer(tmp_path):
    llm = ScriptedLLM()
    session = SessionManager(MemoryConfig(
        checkpoint_dir=str(tmp_path / "cp"),
        long_term_persist_dir=str(tmp_path / "memory"),
    ), llm=llm)
    session_id = session.create_session()
    graph = create_workflow(
        llm=llm,
        llm_config=LLMConfig(provider="ollama"),
        agent_config=AgentConfig(),
        workflow_config=WorkflowConfig(max_rework=1),
        rag_config=RAGConfig(),
        tool_registry=build_registry(tmp_path),
        search_engine=EmptySearch(),
        session_manager=session,
        repo_root=tmp_path,
    )

    result = graph.invoke(
        initial_workflow_state("git diff", session_id),
        session.get_thread_config(session_id),
    )

    assert result["route"] == "researcher"
    assert result["final_answer"].startswith("diff --git")
    assert "当前快照" not in result["final_answer"]
    assert result["search_artifact"].direct_answer == result["final_answer"]
    assert result["messages"][-1].additional_kwargs["render_hint"] == "diff"
    session.close()
