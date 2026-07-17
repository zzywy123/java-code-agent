"""Integrated Multi-Agent LangGraph workflow for Phase 3A."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.agent_graph import create_agent_graph
from agent.agents.artifacts import ArtifactFactory
from agent.agents.permission import AgentRole, PermissionManager
from agent.agents.researcher import ResearcherAgent
from agent.agents.supervisor import SupervisorAgent
from agent.agents.tester import TesterAgent
from agent.agents.verifier import VerifierAgent
from agent.config import AgentConfig, LLMConfig, RAGConfig, WorkflowConfig
from agent.models import (
    AgentArtifact,
    CodeChangeArtifact,
    PatchRecord,
    ReviewArtifact,
    SearchArtifact,
    TestResultArtifact,
    ToolCallRequest,
)
from agent.observability.tracer import observe_span
from agent.rag.agentic_rag import AgenticRAG
from agent.rag.evidence_judge import EvidenceJudge
from agent.rag.query_rewriter import QueryRewriter
from agent.session import SessionManager
from agent.tools.base import ToolRegistry


class WorkflowState(TypedDict, total=False):
    """Shared state for routing, coding, testing and verifier rework."""

    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    route: str
    session_id: str
    trace_id: str
    iteration: int
    consecutive_failures: int
    pending_tool_calls: list[ToolCallRequest]
    patches: list[PatchRecord]
    agent_artifacts: list[AgentArtifact]
    search_artifact: SearchArtifact | None
    test_result: TestResultArtifact | None
    review_result: ReviewArtifact | None
    rework_count: int
    final_answer: str | None
    error: str | None


READ_TOOLS = {"list_files", "read_file", "search_code", "git_status", "git_diff", "git_log"}
CODER_TOOLS = READ_TOOLS | {"apply_patch", "undo_patch"}
TEST_TOOLS = READ_TOOLS | {"run_tests"}


def create_workflow(
    *,
    llm: Any,
    llm_config: LLMConfig,
    agent_config: AgentConfig,
    workflow_config: WorkflowConfig,
    rag_config: RAGConfig,
    tool_registry: ToolRegistry,
    search_engine: Any,
    session_manager: SessionManager,
    repo_root: Any,
    mcp_adapter: Any | None = None,
):
    """Create the persisted parent graph with a restricted Coder subgraph."""
    permissions = PermissionManager()
    researcher = ResearcherAgent(
        tool_registry.restricted(READ_TOOLS),
        permissions,
        AgenticRAG(
            rag_config,
            search_engine,
            QueryRewriter(llm=llm),
            EvidenceJudge(llm=llm, threshold=rag_config.evidence_threshold),
        ),
        mcp_adapter=mcp_adapter,
    )
    supervisor = SupervisorAgent(llm=llm)
    tester = TesterAgent(tool_registry.restricted(TEST_TOOLS), permissions)
    verifier = VerifierAgent(tool_registry.restricted(READ_TOOLS), permissions, llm=llm)

    def traced_node(name: str, node):
        def wrapped(state):
            with observe_span(name):
                return node(state)
        return wrapped

    def context_provider(state) -> list[BaseMessage]:
        return session_manager.build_context(
            state.get("session_id", "default"),
            list(state.get("messages", [])),
            state.get("task", ""),
        )

    coding_graph = create_agent_graph(
        llm_config,
        agent_config,
        tool_registry.restricted(CODER_TOOLS),
        repo_root,
        llm=llm,
        checkpointer=False,
        context_provider=context_provider,
    )

    def supervisor_node(state: WorkflowState) -> dict:
        task = state.get("task") or _latest_human_text(state.get("messages", []))
        role = supervisor.route(task, {"rework_count": state.get("rework_count", 0)})
        route = "coding_workflow" if role == AgentRole.CODER else role.value
        return {
            "task": task,
            "route": route,
            "trace_id": state.get("trace_id") or str(uuid.uuid4()),
            "iteration": state.get("iteration", 0),
            "consecutive_failures": state.get("consecutive_failures", 0),
            "pending_tool_calls": state.get("pending_tool_calls", []),
            "patches": state.get("patches", []),
            "agent_artifacts": state.get("agent_artifacts", []),
            "rework_count": state.get("rework_count", 0),
            "final_answer": None,
            "error": None,
        }

    def route_from_supervisor(state: WorkflowState) -> str:
        return state.get("route", "researcher")

    def researcher_node(state: WorkflowState) -> dict:
        artifact = researcher.run(state["task"])
        artifacts = list(state.get("agent_artifacts", [])) + [artifact]
        if state.get("route") == "researcher":
            answer = (
                artifact.direct_answer
                if artifact.direct_answer is not None
                else _answer_from_sources(llm, state["task"], artifact)
            )
            return {
                "search_artifact": artifact,
                "agent_artifacts": artifacts,
                "messages": [AIMessage(
                    content=answer,
                    additional_kwargs=(
                        {"render_hint": artifact.render_hint}
                        if artifact.render_hint is not None
                        else {}
                    ),
                )],
                "final_answer": answer,
            }

        evidence = _format_research_context(artifact)
        return {
            "search_artifact": artifact,
            "agent_artifacts": artifacts,
            "messages": [SystemMessage(content=evidence)],
        }

    def collect_change_node(state: WorkflowState) -> dict:
        patches = list(state.get("patches", []))
        artifact = ArtifactFactory.create_code_change_artifact(
            description=state.get("task", ""),
            patches=patches,
            affected_files=sorted({patch.file_path for patch in patches}),
            rationale="Coder子图已应用真实Patch",
        )
        artifacts = [
            item for item in state.get("agent_artifacts", [])
            if getattr(item, "artifact_type", "") != "code_change"
        ]
        artifacts.append(artifact)
        return {"agent_artifacts": artifacts, "final_answer": None}

    def tester_node(state: WorkflowState) -> dict:
        result = tester.run(
            state.get("task", "运行测试"),
            {"agent_artifacts": state.get("agent_artifacts", [])},
        )
        artifacts = [
            item for item in state.get("agent_artifacts", [])
            if getattr(item, "artifact_type", "") != "test_result"
        ]
        artifacts.append(result)
        return {"test_result": result, "agent_artifacts": artifacts}

    def verifier_node(state: WorkflowState) -> dict:
        review = verifier.run(
            state.get("task", "审查修改"),
            {"agent_artifacts": state.get("agent_artifacts", [])},
        )
        artifacts = [
            item for item in state.get("agent_artifacts", [])
            if getattr(item, "artifact_type", "") != "review"
        ]
        artifacts.append(review)
        return {"review_result": review, "agent_artifacts": artifacts}

    def route_after_verifier(state: WorkflowState) -> str:
        review = state.get("review_result")
        if review and review.approved:
            return "finish"
        if state.get("route") != "coding_workflow":
            return "finish"
        if state.get("rework_count", 0) >= workflow_config.max_rework:
            return "finish"
        return "rework"

    def route_after_tester(state: WorkflowState) -> str:
        return "verifier" if state.get("route") == "coding_workflow" else "finish"

    def rework_node(state: WorkflowState) -> dict:
        review = state.get("review_result")
        issues = review.issues if review else ["Verifier未提供具体原因"]
        return {
            "messages": [HumanMessage(content="请根据Verifier意见返工：\n- " + "\n- ".join(issues))],
            "rework_count": state.get("rework_count", 0) + 1,
            "iteration": 0,
            "consecutive_failures": 0,
            "pending_tool_calls": [],
            "final_answer": None,
            "error": None,
        }

    def finish_node(state: WorkflowState) -> dict:
        if state.get("final_answer"):
            return {}
        review = state.get("review_result")
        test_result = state.get("test_result")
        if review:
            status = "通过" if review.approved else "未通过"
            answer = f"Verifier审查{status}：{review.summary}"
            if test_result:
                answer += f"\n测试退出码：{test_result.exit_code}"
            if not review.approved and state.get("rework_count", 0) >= workflow_config.max_rework:
                answer += f"\n已达到最大返工次数 {workflow_config.max_rework}。"
            return {"final_answer": answer}
        if test_result:
            return {"final_answer": f"测试{'通过' if test_result.success else '失败'}：{test_result.command}"}
        return {"final_answer": "工作流已结束。"}

    graph = StateGraph(WorkflowState)
    graph.add_node("supervisor", traced_node("supervisor.route", supervisor_node))
    graph.add_node("researcher", traced_node("researcher.retrieve", researcher_node))
    graph.add_node("coder", coding_graph)
    graph.add_node("collect_change", traced_node("coder.collect_change", collect_change_node))
    graph.add_node("tester", traced_node("tester.run", tester_node))
    graph.add_node("verifier", traced_node("verifier.review", verifier_node))
    graph.add_node("rework", traced_node("workflow.rework", rework_node))
    graph.add_node("finish", traced_node("workflow.finish", finish_node))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "coding_workflow": "researcher",
            "tester": "tester",
            "verifier": "verifier",
        },
    )
    graph.add_conditional_edges(
        "researcher",
        lambda state: "finish" if state.get("route") == "researcher" else "coder",
        {"finish": "finish", "coder": "coder"},
    )
    graph.add_edge("coder", "collect_change")
    graph.add_edge("collect_change", "tester")
    graph.add_conditional_edges("tester", route_after_tester, {"verifier": "verifier", "finish": "finish"})
    graph.add_conditional_edges("verifier", route_after_verifier, {"rework": "rework", "finish": "finish"})
    graph.add_edge("rework", "coder")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=session_manager.get_checkpointer())


def initial_workflow_state(query: str, session_id: str) -> WorkflowState:
    return {
        "messages": [HumanMessage(content=query)],
        "task": query,
        "session_id": session_id,
        "iteration": 0,
        "consecutive_failures": 0,
        "pending_tool_calls": [],
        "patches": [],
        "agent_artifacts": [],
        "rework_count": 0,
        "final_answer": None,
        "error": None,
    }


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _format_research_context(artifact: SearchArtifact) -> str:
    lines = ["Researcher检索到以下真实代码证据："]
    for result in artifact.results[:8]:
        source = result.chunk.slice
        lines.append(
            f"- {source.file_path}:{source.start_line}-{source.end_line} "
            f"{source.symbol_signature}\n{source.content[:1200]}"
        )
    if not artifact.results:
        lines.append("- 未检索到充分证据，Coder必须使用只读工具继续定位，禁止猜测。")
    return "\n".join(lines)


def _answer_from_sources(llm: Any, task: str, artifact: SearchArtifact) -> str:
    context = _format_research_context(artifact)
    response = llm.invoke([
        SystemMessage(content=(
            "你是Java代码研究员。只能根据给出的代码证据回答，并引用真实文件路径和行号。"
            "证据不足时明确说明，不得编造。\n\n" + context
        )),
        HumanMessage(content=task),
    ])
    return str(response.content)
