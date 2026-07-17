"""Approval flow for dangerous operations.

Supports LangGraph interrupt/resume pattern.
When an operation needs approval, the graph pauses via interrupt().
After user approval, the graph resumes from the interrupt point.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent.models import ToolCallRequest, ToolStatus


class ApprovalRequest(BaseModel):
    """Request for user approval of a pending operation."""

    tool_calls: list[ToolCallRequest] = Field(description="Tool calls requiring approval")
    summary: str = Field(description="Human-readable summary of what will happen")
    diffs: dict[str, str] = Field(
        default_factory=dict,
        description="File path → unified diff for file modifications",
    )
    commands: list[str] = Field(
        default_factory=list,
        description="Commands that will be executed",
    )


class ApprovalDecision(BaseModel):
    """User's approval decision."""

    approved: bool
    reason: str = ""


# Tools that are read-only and never need approval
READ_ONLY_TOOLS: set[str] = {
    "list_files",
    "read_file",
    "search_code",
    "git_status",
    "git_diff",
    "git_log",
}

# Tools that modify files and always need approval
WRITE_TOOLS: set[str] = {
    "apply_patch",
    "undo_patch",
}

# Tools that execute external processes and need approval
EXEC_TOOLS: set[str] = {
    "run_tests",
}


def needs_approval(tool_calls: list[ToolCallRequest]) -> bool:
    """Check if any tool call requires user approval.

    Args:
        tool_calls: List of pending tool calls

    Returns:
        True if any tool call requires approval
    """
    for tc in tool_calls:
        if tc.name in WRITE_TOOLS or tc.name in EXEC_TOOLS:
            return True
    return False


def build_approval_request(tool_calls: list[ToolCallRequest]) -> ApprovalRequest:
    """Build an approval request from pending tool calls.

    Args:
        tool_calls: List of tool calls that need approval

    Returns:
        ApprovalRequest with human-readable summary
    """
    write_calls = [tc for tc in tool_calls if tc.name in WRITE_TOOLS]
    exec_calls = [tc for tc in tool_calls if tc.name in EXEC_TOOLS]

    summary_parts = []

    if write_calls:
        files = [tc.arguments.get("path", "unknown") for tc in write_calls]
        summary_parts.append(f"文件修改: {', '.join(files)}")

    if exec_calls:
        for tc in exec_calls:
            tool_name = tc.arguments.get("tool", "unknown")
            goals = tc.arguments.get("goals", [])
            summary_parts.append(f"执行构建: {tool_name} {' '.join(goals)}")

    return ApprovalRequest(
        tool_calls=tool_calls,
        summary="; ".join(summary_parts) if summary_parts else "未知操作",
    )


def create_denied_results(
    tool_calls: list[ToolCallRequest],
    reason: str = "用户拒绝操作",
) -> list[dict[str, Any]]:
    """Create denied ToolResult messages for rejected tool calls.

    Args:
        tool_calls: The tool calls that were denied
        reason: Reason for denial

    Returns:
        List of ToolMessage-compatible dicts
    """
    results = []
    for tc in tool_calls:
        results.append({
            "tool_call_id": tc.id,
            "name": tc.name,
            "status": ToolStatus.DENIED,
            "output": f"操作被拒绝: {reason}",
        })
    return results
