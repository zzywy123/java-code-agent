"""MCP Server using the official mcp Python SDK.

Exposes agent tools (search, read, git, test) as MCP tools
via stdio transport. Reuses the existing ToolRegistry and PermissionManager.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from agent.agents.permission import AgentRole, PermissionManager
from agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class CodingAgentMCPServer:
    """MCP Server that exposes coding agent tools.

    Uses the official mcp Python SDK with stdio transport.
    Maps agent tools to MCP tools with proper error handling.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        permission_manager: PermissionManager,
        server_name: str = "java-coding-agent",
        server_version: str = "0.2.0",
        role: AgentRole = AgentRole.RESEARCHER,
    ) -> None:
        self._tools = tool_registry
        self._permissions = permission_manager
        self._server = Server(server_name)
        self._version = server_version
        self._role = role
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register MCP handlers."""

        @self._server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List all available tools."""
            return self._list_mcp_tools()

        @self._server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any]
        ) -> CallToolResult:
            """Handle a tool call."""
            return await self._handle_tool_call(name, arguments)

    def _list_mcp_tools(self) -> list[Tool]:
        """Convert agent tools to MCP tool format."""
        mcp_tools = []
        for tool in self._tools.get_all_tools():
            if not self._permissions.can_use_tool(self._role, tool.name):
                continue
            mcp_tools.append(Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.parameters_schema,
            ))
        return mcp_tools

    async def _handle_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        """Execute a tool call and return MCP-formatted result."""
        # Execute through the agent's tool registry
        # Use a generic agent role for MCP calls (reader by default)
        # Check permission
        allowed, reason = self._permissions.check_tool(self._role, name)
        if not allowed:
            return self._result("denied", f"权限拒绝: {reason}", is_error=True)

        # Execute the tool
        try:
            result = self._tools.execute(
                name=name,
                tool_call_id=f"mcp_{name}",
                **arguments,
            )

            # Map ToolStatus to MCP response
            return self._result(
                result.status.value,
                result.output,
                metadata=result.metadata,
                is_error=result.status.value != "success",
            )

        except Exception as e:
            logger.error("MCP tool call failed: %s %s: %s", name, arguments, e)
            return self._result("execution_error", f"执行异常: {e}", is_error=True)

    @staticmethod
    def _result(
        status: str,
        output: str,
        *,
        metadata: dict[str, Any] | None = None,
        is_error: bool = False,
    ) -> CallToolResult:
        payload = json.dumps(
            {"status": status, "output": output, "metadata": metadata or {}},
            ensure_ascii=False,
        )
        return CallToolResult(
            content=[TextContent(type="text", text=payload)],
            isError=is_error,
        )

    async def run(self) -> None:
        """Run the MCP server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )


def create_mcp_server(
    tool_registry: ToolRegistry,
    permission_manager: PermissionManager | None = None,
    role: AgentRole = AgentRole.RESEARCHER,
) -> CodingAgentMCPServer:
    """Factory function to create an MCP server."""
    if permission_manager is None:
        permission_manager = PermissionManager()
    return CodingAgentMCPServer(
        tool_registry=tool_registry,
        permission_manager=permission_manager,
        role=role,
    )


def main() -> None:
    """Run a role-scoped stdio MCP server for the configured repository."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=os.environ.get("AGENT_REPO_ROOT", "."))
    parser.add_argument("--role", choices=[role.value for role in AgentRole], default="researcher")
    args = parser.parse_args()

    from agent.tools.factory import create_tool_registry

    registry = create_tool_registry(Path(args.repo_root).resolve())
    server = create_mcp_server(registry, role=AgentRole(args.role))
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
