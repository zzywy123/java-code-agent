"""Persistent session and memory-context tests."""

from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, StateGraph

from agent.config import MemoryConfig
from agent.session import SessionManager


class CounterState(TypedDict):
    value: int


def test_sqlite_checkpoint_survives_manager_restart(tmp_path):
    config = MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
    )
    manager = SessionManager(config)
    session_id = manager.create_session("persistent")

    builder = StateGraph(CounterState)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    graph = builder.compile(checkpointer=manager.get_checkpointer())
    graph.invoke({"value": 1}, manager.get_thread_config(session_id))
    manager.close()

    reopened = SessionManager(config)
    rebuilt = builder.compile(checkpointer=reopened.get_checkpointer())
    snapshot = rebuilt.get_state(reopened.get_thread_config(session_id))
    assert snapshot.values["value"] == 2
    assert reopened.get_or_create_active_session() == session_id
    reopened.close()


def test_memory_context_injects_only_validated_project_memory(tmp_path):
    config = MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
    )
    manager = SessionManager(config)
    session_id = manager.create_session()
    manager.long_term.store("style", {"type": "convention", "content": "测试命名使用should前缀"})
    manager.long_term.store("stale_bug", {"type": "bug", "content": "calculateTotal仍有Bug"})

    context = manager.build_context(
        session_id,
        [HumanMessage(content="请补充测试")],
        "测试规范",
    )
    rendered = "\n".join(str(message.content) for message in context)
    assert "should前缀" in rendered
    assert "calculateTotal仍有Bug" not in rendered
    manager.close()


def test_context_keeps_tool_call_and_tool_result_adjacent(tmp_path):
    config = MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
    )
    manager = SessionManager(config)
    session_id = manager.create_session()
    ai = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "read_file", "args": {}}],
    )
    tool = ToolMessage(content="file content", tool_call_id="call-1", name="read_file")
    context = manager.build_context(session_id, [HumanMessage(content="读取文件"), ai, tool], "读取文件")
    ai_index = next(index for index, item in enumerate(context) if isinstance(item, AIMessage))
    assert isinstance(context[ai_index + 1], ToolMessage)
    assert context[ai_index + 1].tool_call_id == "call-1"
    manager.close()


def test_context_converts_tool_result_orphaned_by_window(tmp_path):
    config = MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
        short_term_window=20,
    )
    manager = SessionManager(config)
    session_id = manager.create_session()
    ai = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "read_file", "args": {}}],
    )
    tool = ToolMessage(content="file content", tool_call_id="call-1", name="read_file")
    messages = [ai, tool] + [HumanMessage(content=f"message-{index}") for index in range(19)]

    context = manager.build_context(session_id, messages, "read file")

    assert not any(isinstance(message, ToolMessage) for message in context)
    assert any(
        isinstance(message, SystemMessage) and "file content" in str(message.content)
        for message in context
    )
    manager.close()


def test_delete_session_removes_checkpoint_and_memory_caches(tmp_path):
    config = MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
    )
    manager = SessionManager(config)
    session_id = manager.create_session("delete me")
    builder = StateGraph(CounterState)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    graph = builder.compile(checkpointer=manager.get_checkpointer())
    graph.invoke({"value": 0}, manager.get_thread_config(session_id))
    manager.build_context(session_id, [HumanMessage(content="remember")], "task")

    manager.delete_session(session_id)

    raw_config = {"configurable": {"thread_id": session_id}}
    assert manager.get_checkpointer().get_tuple(raw_config) is None
    assert session_id not in manager._short_term
    assert session_id not in manager._summaries
    assert session_id not in manager._seen_messages
    assert all(item["session_id"] != session_id for item in manager.list_sessions())
    with pytest.raises(KeyError, match="会话不存在"):
        manager.get_thread_config(session_id)
    manager.close()
