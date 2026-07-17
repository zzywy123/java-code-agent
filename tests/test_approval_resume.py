"""Tests for approval flow and resume after interrupt.

Verifies that:
- Approval request is properly built
- Denied operations create correct ToolMessages
- State is preserved across approval flow
"""

from __future__ import annotations

import pytest

from agent.models import ToolCallRequest, ToolStatus
from agent.security.approval import (
    build_approval_request,
    create_denied_results,
    needs_approval,
)


class TestNeedsApproval:
    """Tests for needs_approval function."""

    def test_read_only_no_approval(self):
        calls = [
            ToolCallRequest(id="c1", name="search_code", arguments={"query": "test"}),
            ToolCallRequest(id="c2", name="read_file", arguments={"path": "Main.java"}),
            ToolCallRequest(id="c3", name="git_status", arguments={}),
        ]
        assert needs_approval(calls) is False

    def test_write_needs_approval(self):
        calls = [
            ToolCallRequest(id="c1", name="apply_patch", arguments={"path": "Main.java", "unified_diff": "@@ ..."}),
        ]
        assert needs_approval(calls) is True

    def test_undo_needs_approval(self):
        calls = [
            ToolCallRequest(id="c1", name="undo_patch", arguments={"path": "Main.java"}),
        ]
        assert needs_approval(calls) is True

    def test_run_tests_needs_approval(self):
        calls = [
            ToolCallRequest(id="c1", name="run_tests", arguments={"tool": "maven", "goals": ["test"]}),
        ]
        assert needs_approval(calls) is True

    def test_mixed_needs_approval(self):
        calls = [
            ToolCallRequest(id="c1", name="search_code", arguments={"query": "test"}),
            ToolCallRequest(id="c2", name="apply_patch", arguments={"path": "Main.java", "unified_diff": "@@ ..."}),
        ]
        assert needs_approval(calls) is True


class TestBuildApprovalRequest:
    """Tests for build_approval_request function."""

    def test_build_for_write(self):
        calls = [
            ToolCallRequest(id="c1", name="apply_patch", arguments={"path": "src/Main.java"}),
        ]
        req = build_approval_request(calls)
        assert "文件修改" in req.summary
        assert "src/Main.java" in req.summary

    def test_build_for_execution(self):
        calls = [
            ToolCallRequest(id="c1", name="run_tests", arguments={"tool": "maven", "goals": ["test"]}),
        ]
        req = build_approval_request(calls)
        assert "执行构建" in req.summary

    def test_build_for_mixed(self):
        calls = [
            ToolCallRequest(id="c1", name="apply_patch", arguments={"path": "Main.java"}),
            ToolCallRequest(id="c2", name="run_tests", arguments={"tool": "maven", "goals": ["test"]}),
        ]
        req = build_approval_request(calls)
        assert "文件修改" in req.summary
        assert "执行构建" in req.summary


class TestCreateDeniedResults:
    """Tests for create_denied_results function."""

    def test_create_denied_results(self):
        calls = [
            ToolCallRequest(id="c1", name="apply_patch", arguments={"path": "Main.java"}),
            ToolCallRequest(id="c2", name="run_tests", arguments={"tool": "maven", "goals": ["test"]}),
        ]
        results = create_denied_results(calls, "用户拒绝")
        assert len(results) == 2
        assert results[0]["tool_call_id"] == "c1"
        assert results[0]["status"] == ToolStatus.DENIED
        assert "用户拒绝" in results[0]["output"]
        assert results[1]["tool_call_id"] == "c2"

    def test_default_reason(self):
        calls = [
            ToolCallRequest(id="c1", name="apply_patch", arguments={}),
        ]
        results = create_denied_results(calls)
        assert "用户拒绝操作" in results[0]["output"]


class TestApprovalFlowStatePreservation:
    """Verify that approval flow preserves state correctly."""

    def test_denied_results_are_valid_tool_messages(self):
        """Denied results should be convertible to ToolMessages."""
        from langchain_core.messages import ToolMessage

        calls = [
            ToolCallRequest(id="call_123", name="apply_patch", arguments={"path": "Main.java"}),
        ]
        results = create_denied_results(calls, "不安全")

        # Convert to ToolMessage (as done in request_approval node)
        messages = []
        for dr in results:
            msg = ToolMessage(
                content=dr["output"],
                tool_call_id=dr["tool_call_id"],
                name=dr["name"],
            )
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].tool_call_id == "call_123"
        assert messages[0].name == "apply_patch"
        assert "不安全" in messages[0].content
