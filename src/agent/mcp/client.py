"""MCP Client using the official mcp Python SDK.

Connects to an MCP server and provides a high-level interface
for listing tools and calling them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class CodingAgentMCPClient:
    """MCP Client that connects to an MCP server.

    Uses the official mcp Python SDK with stdio transport.
    """

    def __init__(
        self,
        server_command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        """Initialize the MCP client.

        Args:
            server_command: Command to start the MCP server
                           e.g., ["python", "-m", "agent.mcp.server"]
        """
        self._server_params = StdioServerParameters(
            command=server_command[0],
            args=server_command[1:],
            env=env,
            cwd=cwd,
        )
        self._session: ClientSession | None = None
        self._tools_cache: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Connect to the MCP server."""
        # The connection is managed per-call in the context manager
        await self.list_tools()

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server.

        Returns:
            List of tool definitions with name, description, and inputSchema
        """
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()

                tools = []
                for tool in result.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    })

                self._tools_cache = tools
                return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool on the MCP server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Result dict with 'content' list and 'isError' flag
        """
        if arguments is None:
            arguments = {}

        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)

                # Parse result
                content_parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        content_parts.append(item.text)
                    else:
                        content_parts.append(str(item))

                return {
                    "content": content_parts,
                    "isError": result.isError if hasattr(result, "isError") else False,
                }

    def get_cached_tools(self) -> list[dict[str, Any]]:
        """Get the cached tool list from the last list_tools call."""
        return self._tools_cache


class MCPToolAdapter:
    """Map MCP calls back into the project's structured ToolResult model."""

    def __init__(self, client: CodingAgentMCPClient, permission_manager, role) -> None:
        self._client = client
        self._permissions = permission_manager
        self._role = role

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        from agent.models import ToolResult, ToolStatus

        self._permissions.assert_tool_allowed(self._role, name)
        raw = await self._client.call_tool(name, arguments)
        text = "\n".join(raw.get("content", []))
        try:
            payload = json.loads(text)
            status = ToolStatus(payload.get("status", "execution_error"))
            return ToolResult(
                tool_call_id=f"mcp_{name}",
                name=name,
                status=status,
                output=payload.get("output", ""),
                metadata=payload.get("metadata", {}),
            )
        except (json.JSONDecodeError, ValueError):
            return ToolResult(
                tool_call_id=f"mcp_{name}",
                name=name,
                status=ToolStatus.EXECUTION_ERROR if raw.get("isError") else ToolStatus.SUCCESS,
                output=text,
            )

    def call_tool_sync(self, name: str, arguments: dict[str, Any]):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call_tool(name, arguments))
        raise RuntimeError("同步MCP适配器不能在运行中的事件循环内调用")


def create_mcp_client(
    server_command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> CodingAgentMCPClient:
    """Factory function to create an MCP client."""
    return CodingAgentMCPClient(server_command=server_command, env=env, cwd=cwd)
