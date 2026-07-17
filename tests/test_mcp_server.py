"""Tests for MCP Server and Client.

Validates:
- MCP tool listing
- MCP tool call execution
- Error mapping (success, permission denied, invalid argument, not found)
- MCP client connection and tool call
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.agents.permission import AgentRole, PermissionManager
from agent.mcp.client import MCPToolAdapter, create_mcp_client
from agent.mcp.server import CodingAgentMCPServer, create_mcp_server
from agent.models import ToolResult, ToolStatus
from agent.tools.base import ToolRegistry


def payload(result) -> dict:
    return json.loads(result.content[0].text)


@pytest.fixture
def mock_registry() -> ToolRegistry:
    """Create a mock tool registry with sample tools."""
    registry = ToolRegistry()

    # Create mock tools
    mock_tool = MagicMock()
    mock_tool.name = "search_code"
    mock_tool.description = "Search code"
    mock_tool.parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    mock_tool.to_openai_tool.return_value = {
        "type": "function",
        "function": {"name": "search_code", "description": "Search code", "parameters": {}},
    }
    registry.register(mock_tool)

    mock_tool2 = MagicMock()
    mock_tool2.name = "read_file"
    mock_tool2.description = "Read file"
    mock_tool2.parameters_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    mock_tool2.to_openai_tool.return_value = {
        "type": "function",
        "function": {"name": "read_file", "description": "Read file", "parameters": {}},
    }
    registry.register(mock_tool2)

    return registry


@pytest.fixture
def permission_manager() -> PermissionManager:
    return PermissionManager()


class TestMCPServer:
    """MCP Server tests."""

    def test_list_mcp_tools(self, mock_registry, permission_manager):
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        tools = server._list_mcp_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "search_code" in names
        assert "read_file" in names

    def test_tool_schema_passed_through(self, mock_registry, permission_manager):
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        tools = server._list_mcp_tools()
        search_tool = next(t for t in tools if t.name == "search_code")
        assert "query" in search_tool.inputSchema["properties"]

    @pytest.mark.asyncio
    async def test_handle_tool_call_success(self, mock_registry, permission_manager):
        mock_registry.execute = MagicMock(return_value=ToolResult(
            tool_call_id="test",
            name="search_code",
            status=ToolStatus.SUCCESS,
            output="Found 5 matches",
        ))
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        result = await server._handle_tool_call("search_code", {"query": "test"})
        assert result.isError is False
        assert payload(result)["output"] == "Found 5 matches"

    @pytest.mark.asyncio
    async def test_handle_tool_call_permission_denied(self, mock_registry, permission_manager):
        # Try to call a write tool with researcher role
        mock_registry.execute = MagicMock(return_value=ToolResult(
            tool_call_id="test",
            name="apply_patch",
            status=ToolStatus.DENIED,
            output="权限拒绝",
        ))
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        # apply_patch is not registered, so it will be permission denied
        result = await server._handle_tool_call("apply_patch", {"path": "test.java"})
        assert result.isError is True
        assert payload(result)["status"] == "denied"

    @pytest.mark.asyncio
    async def test_handle_tool_call_invalid_argument(self, mock_registry, permission_manager):
        mock_registry.execute = MagicMock(return_value=ToolResult(
            tool_call_id="test",
            name="search_code",
            status=ToolStatus.INVALID_ARGUMENT,
            output="参数错误: 缺少 query",
        ))
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        result = await server._handle_tool_call("search_code", {})
        assert result.isError is True
        assert payload(result)["status"] == "invalid_argument"

    @pytest.mark.asyncio
    async def test_handle_tool_call_not_found(self, mock_registry, permission_manager):
        """Tool not in registry → permission denied (tool not in allowed set)."""
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        result = await server._handle_tool_call("nonexistent_tool", {})
        # Permission check happens first — unknown tool is denied
        assert result.isError is True
        assert payload(result)["status"] == "denied"

    @pytest.mark.asyncio
    async def test_handle_tool_call_exception(self, mock_registry, permission_manager):
        mock_registry.execute = MagicMock(side_effect=Exception("unexpected error"))
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        result = await server._handle_tool_call("search_code", {"query": "test"})
        assert result.isError is True
        assert payload(result)["status"] == "execution_error"

    def test_create_mcp_server_factory(self, mock_registry):
        server = create_mcp_server(mock_registry)
        assert isinstance(server, CodingAgentMCPServer)


class TestMCPErrorMapping:
    """MCP error code mapping tests."""

    @pytest.mark.asyncio
    async def test_success_maps_to_content(self, mock_registry, permission_manager):
        mock_registry.execute = MagicMock(return_value=ToolResult(
            tool_call_id="t", name="search_code",
            status=ToolStatus.SUCCESS, output="result text",
        ))
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        result = await server._handle_tool_call("search_code", {"query": "x"})
        assert payload(result)["output"] == "result text"

    @pytest.mark.asyncio
    async def test_denied_maps_to_permission_error(self, mock_registry, permission_manager):
        mock_registry.execute = MagicMock(return_value=ToolResult(
            tool_call_id="t", name="search_code",
            status=ToolStatus.DENIED, output="安全拒绝",
        ))
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        result = await server._handle_tool_call("search_code", {"query": "x"})
        assert payload(result)["status"] == "denied"

    @pytest.mark.asyncio
    async def test_execution_error_maps_to_error(self, mock_registry, permission_manager):
        mock_registry.execute = MagicMock(return_value=ToolResult(
            tool_call_id="t", name="search_code",
            status=ToolStatus.EXECUTION_ERROR, output="执行失败",
        ))
        server = CodingAgentMCPServer(mock_registry, permission_manager)
        result = await server._handle_tool_call("search_code", {"query": "x"})
        assert payload(result)["status"] == "execution_error"


def test_adapter_caches_discovered_capabilities_and_rejects_unknown_tool():
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=[{"name": "search_code"}])
    client.call_tool = AsyncMock()
    adapter = MCPToolAdapter(client, PermissionManager(), AgentRole.RESEARCHER)

    assert adapter.initialize_sync() == {"search_code"}
    result = asyncio.run(adapter.call_tool("git_status", {}))

    assert result.status == ToolStatus.NOT_FOUND
    client.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_stdio_client_calls_real_server(tmp_repo):
    """Exercise initialize, tools/list and tools/call over a real stdio process."""
    env = os.environ.copy()
    env["AGENT_REPO_ROOT"] = str(tmp_repo)
    project_root = Path(__file__).parent.parent
    client = create_mcp_client(
        [
            sys.executable,
            "-m",
            "agent.mcp.server",
            "--repo-root",
            str(tmp_repo),
            "--role",
            "researcher",
        ],
        env=env,
        cwd=project_root,
    )
    tools = await client.list_tools()
    assert "search_code" in {tool["name"] for tool in tools}

    adapter = MCPToolAdapter(client, PermissionManager(), AgentRole.RESEARCHER)
    result = await adapter.call_tool(
        "search_code",
        {"query": "greet", "path": ".", "file_pattern": "*.java"},
    )
    assert result.status == ToolStatus.SUCCESS
    assert "Hello.java" in result.output
