"""Agent permission layer with role-based access control.

Each agent role has specific permissions:
- RESEARCHER: read-only (search, read, git)
- CODER: read + write (apply_patch, undo_patch)
- TESTER: read + restricted execution (run_tests only)
- VERIFIER: read-only (cannot modify code)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from agent.models import ToolCallRequest


class AgentRole(str, Enum):
    """Agent roles with different permission levels."""
    RESEARCHER = "researcher"
    CODER = "coder"
    TESTER = "tester"
    VERIFIER = "verifier"


# Tool name sets for permission checking
READ_TOOLS = {
    "list_files", "read_file", "search_code",
    "git_status", "git_diff", "git_log",
}

WRITE_TOOLS = {
    "apply_patch", "undo_patch",
}

EXEC_TOOLS = {
    "run_tests",
}

# Role → allowed tool sets
ROLE_PERMISSIONS: dict[AgentRole, set[str]] = {
    AgentRole.RESEARCHER: READ_TOOLS,
    AgentRole.CODER: READ_TOOLS | WRITE_TOOLS,
    AgentRole.TESTER: READ_TOOLS | EXEC_TOOLS,
    AgentRole.VERIFIER: READ_TOOLS,
}


class PermissionViolationError(Exception):
    """Raised when an agent attempts an unauthorized operation."""

    def __init__(self, role: AgentRole, tool_name: str):
        self.role = role
        self.tool_name = tool_name
        super().__init__(
            f"Agent role '{role.value}' 无权使用工具 '{tool_name}'"
        )


class PermissionManager:
    """Manages agent permissions and tool access control.

    Enforces role-based access control for all tool calls.
    """

    def can_use_tool(self, role: AgentRole, tool_name: str) -> bool:
        """Check if a role can use a specific tool."""
        allowed = ROLE_PERMISSIONS.get(role, set())
        return tool_name in allowed

    def check_tool(self, role: AgentRole, tool_name: str) -> tuple[bool, str]:
        """Check tool permission and return (allowed, reason)."""
        if self.can_use_tool(role, tool_name):
            return True, "允许"
        return False, f"角色 '{role.value}' 无权使用工具 '{tool_name}'"

    def filter_tool_calls(
        self,
        role: AgentRole,
        tool_calls: list[ToolCallRequest],
    ) -> tuple[list[ToolCallRequest], list[ToolCallRequest]]:
        """Filter tool calls into allowed and denied lists.

        Returns:
            (allowed_calls, denied_calls)
        """
        allowed = []
        denied = []
        for tc in tool_calls:
            if self.can_use_tool(role, tc.name):
                allowed.append(tc)
            else:
                denied.append(tc)
        return allowed, denied

    def get_allowed_tools(self, role: AgentRole) -> set[str]:
        """Get the set of tools allowed for a role."""
        return ROLE_PERMISSIONS.get(role, set()).copy()

    def assert_tool_allowed(self, role: AgentRole, tool_name: str) -> None:
        """Assert that a tool is allowed, raise PermissionViolationError if not."""
        if not self.can_use_tool(role, tool_name):
            raise PermissionViolationError(role, tool_name)
