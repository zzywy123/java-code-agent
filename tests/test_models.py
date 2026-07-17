"""Tests for agent.models module."""

from __future__ import annotations

from datetime import datetime

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent.agent_state import AgentState
from agent.models import (
    PatchRecord,
    ToolCallRequest,
    ToolResult,
    ToolStatus,
)


class TestToolStatus:
    """Tests for ToolStatus enum."""

    def test_all_statuses_exist(self):
        expected = {
            "success", "error", "denied", "timeout",
            "not_found", "invalid_argument", "execution_error",
            "pending_approval",
        }
        actual = {s.value for s in ToolStatus}
        assert actual == expected

    def test_string_values(self):
        assert ToolStatus.SUCCESS == "success"
        assert ToolStatus.NOT_FOUND == "not_found"
        assert ToolStatus.INVALID_ARGUMENT == "invalid_argument"
        assert ToolStatus.EXECUTION_ERROR == "execution_error"


class TestToolCallRequest:
    """Tests for ToolCallRequest model."""

    def test_create_minimal(self):
        req = ToolCallRequest(id="call_1", name="read_file")
        assert req.id == "call_1"
        assert req.name == "read_file"
        assert req.arguments == {}

    def test_create_with_arguments(self):
        req = ToolCallRequest(
            id="call_2",
            name="search_code",
            arguments={"query": "calculateTotal", "path": "."},
        )
        assert req.arguments["query"] == "calculateTotal"

    def test_serialization(self):
        req = ToolCallRequest(id="c1", name="test", arguments={"a": 1})
        data = req.model_dump()
        assert data["id"] == "c1"
        assert data["arguments"] == {"a": 1}
        restored = ToolCallRequest.model_validate(data)
        assert restored == req


class TestToolResult:
    """Tests for ToolResult model."""

    def test_success_result(self):
        result = ToolResult(
            tool_call_id="call_1",
            name="read_file",
            status=ToolStatus.SUCCESS,
            output="file content here",
        )
        assert result.status == ToolStatus.SUCCESS
        assert result.metadata == {}

    def test_error_result(self):
        result = ToolResult(
            tool_call_id="call_2",
            name="run_tests",
            status=ToolStatus.EXECUTION_ERROR,
            output="Tests run: 3, Failures: 1",
            metadata={"exit_code": 1},
        )
        assert result.status == ToolStatus.EXECUTION_ERROR
        assert result.metadata["exit_code"] == 1

    def test_denied_result(self):
        result = ToolResult(
            tool_call_id="call_3",
            name="apply_patch",
            status=ToolStatus.DENIED,
            output="安全层拒绝：路径穿越",
        )
        assert result.status == ToolStatus.DENIED

    def test_not_found_result(self):
        result = ToolResult(
            tool_call_id="call_4",
            name="read_file",
            status=ToolStatus.NOT_FOUND,
            output="File not found: /missing.java",
        )
        assert result.status == ToolStatus.NOT_FOUND


class TestPatchRecord:
    """Tests for PatchRecord model."""

    def test_create_patch_record(self):
        record = PatchRecord(
            file_path="/repo/src/Main.java",
            content_hash_before="abc123",
            content_hash_after="def456",
            unified_diff="@@ -1,3 +1,4 @@\n+new line",
        )
        assert record.is_new_file is False
        assert isinstance(record.timestamp, datetime)

    def test_new_file_patch(self):
        record = PatchRecord(
            file_path="/repo/src/New.java",
            content_hash_before="",
            content_hash_after="abc123",
            unified_diff="+++ New.java\n+public class New {}",
            is_new_file=True,
        )
        assert record.is_new_file is True

    def test_no_full_content_stored(self):
        """Verify that PatchRecord never stores full file content."""
        record = PatchRecord(
            file_path="/repo/src/Main.java",
            content_hash_before="a" * 64,
            content_hash_after="b" * 64,
            unified_diff="small diff",
        )
        data = record.model_dump()
        # No field should contain the actual file content
        assert "file_content" not in data
        assert "original_content" not in data
        assert "patched_content" not in data
        # Only hashes and diff
        assert len(data["content_hash_before"]) == 64
        assert len(data["content_hash_after"]) == 64


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_state_has_required_keys(self):
        required = {
            "messages", "iteration", "consecutive_failures",
            "pending_tool_calls", "patches", "final_answer", "error",
        }
        actual = set(AgentState.__annotations__.keys())
        assert required <= actual

    def test_messages_uses_add_messages_reducer(self):
        """Verify messages field uses add_messages annotation."""
        import typing
        ann = AgentState.__annotations__["messages"]
        # Resolve forward refs from __future__ annotations
        resolved = typing.get_type_hints(AgentState, include_extras=True)["messages"]
        assert hasattr(resolved, "__metadata__")
        from langgraph.graph.message import add_messages
        assert add_messages in resolved.__metadata__

    def test_state_with_messages(self):
        """Verify state can hold LangChain messages."""
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "iteration": 0,
            "consecutive_failures": 0,
            "pending_tool_calls": [],
            "patches": [],
            "final_answer": None,
            "error": None,
        }
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Hello"
