"""LangGraph agent graph for the Java Coding Agent.

Architecture:
- agent_node: Calls LLM, parses tool calls into pending_tool_calls
- check_approval: Routes to approval or tool_executor (READ-ONLY, never modifies state)
- request_approval: Uses interrupt() to pause for user approval
- tool_executor: Executes all pending tool calls, writes results to state

Key design decisions:
- Router functions NEVER modify state
- AgentState uses add_messages reducer for automatic message accumulation
- Multiple tool calls are processed as a batch
- interrupt() is used for approval flow with resume support
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from agent.agent_state import AgentState  # noqa: E402
from agent.config import AgentConfig, LLMConfig
from agent.llm_client import create_llm_with_tools
from agent.models import PatchRecord, ToolCallRequest, ToolResult, ToolStatus
from agent.observability.tracer import observe_span
from agent.prompts import SYSTEM_PROMPT
from agent.security.approval import (
    build_approval_request,
    create_denied_results,
    needs_approval,
)
from agent.tools.base import ToolRegistry


def create_agent_graph(
    llm_config: LLMConfig,
    agent_config: AgentConfig,
    tool_registry: ToolRegistry,
    repo_root: Any,
    *,
    llm: Any | None = None,
    checkpointer: BaseCheckpointSaver | None | bool = None,
    context_provider: Callable[[AgentState], list[BaseMessage]] | None = None,
    require_approval: bool = True,
) -> Any:
    """Create the LangGraph agent graph.

    Args:
        llm_config: LLM configuration
        agent_config: Agent configuration
        tool_registry: Registry of available tools
        repo_root: Repository root path

    Returns:
        Compiled LangGraph StateGraph
    """
    # Get tools in OpenAI format
    openai_tools = tool_registry.get_openai_tools()

    # Create LLM with tools bound
    if llm is None:
        llm = create_llm_with_tools(llm_config, openai_tools)
    else:
        llm = llm.bind_tools(openai_tools)

    def traced_node(name: str, node):
        def wrapped(state):
            with observe_span(name):
                return node(state)
        return wrapped

    # --- Agent Node ---
    def agent_node(state: AgentState) -> dict:
        """Call LLM and parse tool calls.

        Adds system prompt on first iteration.
        Returns partial state update with new messages and pending tool calls.
        """
        messages = (
            context_provider(state)
            if context_provider is not None
            else list(state["messages"])
        )

        # Add system prompt if this is the first message
        if state["iteration"] == 0:
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Call LLM
        response = llm.invoke(messages)

        # Parse tool calls from response
        pending = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                pending.append(ToolCallRequest(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc.get("args", {}),
                ))

        if any(_is_duplicate_patch(call, state.get("patches", []), repo_root) for call in pending):
            response = AIMessage(
                content="补丁已经成功应用，修改阶段结束，后续由 Tester 运行测试。"
            )
            pending = []

        return {
            "messages": [response],
            "pending_tool_calls": pending,
            "iteration": state["iteration"] + 1,
        }

    # --- Router (READ-ONLY, never modifies state) ---
    def router(state: AgentState) -> str:
        """Route based on whether there are pending tool calls.

        Returns:
            "check_approval" if there are tool calls to process
            "end" if no tool calls (final answer)
        """
        # Check termination conditions first
        if state.get("error"):
            return "end"
        if state["iteration"] >= agent_config.max_iterations:
            return "end"
        if state["consecutive_failures"] >= agent_config.max_consecutive_failures:
            return "end"
        # Check if there are pending tool calls
        if state.get("pending_tool_calls"):
            return "check_approval"
        return "end"

    # --- Check Approval Node ---
    def check_approval(state: AgentState) -> str:
        """Check if any pending tool calls need approval.

        Returns routing decision. Does NOT modify state.
        """
        pending = state.get("pending_tool_calls", [])
        if require_approval and needs_approval(pending):
            return "request_approval"
        return "tool_executor"

    # --- Request Approval Node ---
    def request_approval(state: AgentState) -> dict:
        """Request user approval via interrupt().

        Pauses execution and waits for user response.
        Returns state update with approval result.
        """
        pending = state.get("pending_tool_calls", [])
        approval_req = build_approval_request(pending)

        # Use interrupt to pause for user approval
        decision = interrupt({
            "type": "approval_request",
            "summary": approval_req.summary,
            "tool_calls": [tc.model_dump() for tc in pending],
            "diffs": approval_req.diffs,
            "commands": approval_req.commands,
        })

        # Check user's decision
        if isinstance(decision, dict):
            approved = decision.get("approved", False)
            reason = decision.get("reason", "")
        elif isinstance(decision, bool):
            approved = decision
            reason = ""
        else:
            approved = bool(decision)
            reason = ""

        if not approved:
            # Create denied results and inject back as messages
            denied_results = create_denied_results(pending, reason or "用户拒绝操作")
            messages = []
            for dr in denied_results:
                messages.append(ToolMessage(
                    content=dr["output"],
                    tool_call_id=dr["tool_call_id"],
                    name=dr["name"],
                ))
            return {
                "messages": messages,
                "pending_tool_calls": [],
                "consecutive_failures": state["consecutive_failures"] + 1,
                "approval_rejected": True,
                "final_answer": "操作已被用户拒绝，未修改文件。",
            }

        # Approved - continue to tool_executor (empty update, flow continues)
        return {"approval_rejected": False}

    # --- Tool Executor Node ---
    def tool_executor(state: AgentState) -> dict:
        """Execute all pending tool calls and collect results.

        Returns state update with tool results as messages.
        """
        pending = state.get("pending_tool_calls", [])
        messages = []
        patches = list(state.get("patches", []))
        consecutive_failures = state["consecutive_failures"]

        for tc in pending:
            result = tool_registry.execute(
                name=tc.name,
                tool_call_id=tc.id,
                **tc.arguments,
            )

            # Create ToolMessage for the result
            msg_content = result.output
            if result.status == ToolStatus.SUCCESS:
                msg_content = f"[{result.status.value}] {result.output}"
            else:
                msg_content = f"[{result.status.value}] {result.output}"
                consecutive_failures += 1

            messages.append(ToolMessage(
                content=msg_content,
                tool_call_id=tc.id,
                name=tc.name,
            ))

            # Track patches
            if tc.name == "apply_patch" and result.status == ToolStatus.SUCCESS:
                patch_data = result.metadata.get("patch_record")
                if patch_data:
                    patches.append(PatchRecord.model_validate(patch_data))

            # Reset consecutive failures on success
            if result.status == ToolStatus.SUCCESS:
                consecutive_failures = 0

        return {
            "messages": messages,
            "pending_tool_calls": [],
            "patches": patches,
            "consecutive_failures": consecutive_failures,
        }

    # --- End Node ---
    def end_node(state: AgentState) -> dict:
        """Extract final answer from the last AI message."""
        messages = state.get("messages", [])
        final_answer = state.get("final_answer")

        if not final_answer:
            # Find the last AIMessage content
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content
                    break

        error = state.get("error")
        if state["iteration"] >= agent_config.max_iterations:
            error = f"达到最大迭代次数 ({agent_config.max_iterations})"
        elif state["consecutive_failures"] >= agent_config.max_consecutive_failures:
            error = f"连续失败次数超限 ({agent_config.max_consecutive_failures})"

        return {
            "final_answer": final_answer,
            "error": error,
        }

    # --- Build Graph ---
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", traced_node("coder.agent", agent_node))
    graph.add_node("check_approval", lambda s: {})  # No-op, routing only
    graph.add_node("request_approval", traced_node("coder.approval", request_approval))
    graph.add_node("tool_executor", traced_node("coder.tools", tool_executor))
    graph.add_node("end_node", traced_node("coder.finish", end_node))

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        router,
        {
            "check_approval": "check_approval",
            "end": "end_node",
        },
    )
    graph.add_conditional_edges(
        "check_approval",
        check_approval,
        {
            "request_approval": "request_approval",
            "tool_executor": "tool_executor",
        },
    )
    graph.add_conditional_edges(
        "request_approval",
        lambda state: "end" if state.get("approval_rejected") else "execute",
        {"end": "end_node", "execute": "tool_executor"},
    )
    graph.add_edge("tool_executor", "agent")
    graph.add_edge("end_node", END)

    # Compile with MemorySaver for interrupt/resume support
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def _is_duplicate_patch(
    call: ToolCallRequest,
    patches: list[PatchRecord],
    repo_root: Any,
) -> bool:
    """Treat an identical patch for the same file as already completed."""
    if call.name != "apply_patch":
        return False
    path = str(call.arguments.get("path") or "")
    unified_diff = str(call.arguments.get("unified_diff") or "")
    if not path or not unified_diff:
        return False
    try:
        requested = (Path(repo_root).resolve() / path).resolve()
        return any(
            Path(patch.file_path).resolve() == requested
            and patch.unified_diff == unified_diff
            for patch in patches
        )
    except (OSError, RuntimeError):
        return False
