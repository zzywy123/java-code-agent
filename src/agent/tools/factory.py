"""Factories for the shared tool layer."""

from pathlib import Path

from agent.tools.base import ToolRegistry
from agent.tools.build_tools import RunTestsTool
from agent.tools.file_tools import ListFilesTool, ReadFileTool
from agent.tools.git_tools import GitDiffTool, GitLogTool, GitStatusTool
from agent.tools.patch_tools import ApplyPatchTool, UndoPatchTool
from agent.tools.search_tools import SearchCodeTool


def create_tool_registry(repo_root: Path) -> ToolRegistry:
    """Create the complete registry used by CLI, workflows and MCP."""
    registry = ToolRegistry()
    registry.register(ListFilesTool(repo_root=repo_root))
    registry.register(ReadFileTool(repo_root=repo_root))
    registry.register(SearchCodeTool(repo_root=repo_root))
    registry.register(ApplyPatchTool(repo_root=repo_root))
    registry.register(UndoPatchTool(repo_root=repo_root))
    registry.register(RunTestsTool(repo_root=repo_root))
    registry.register(GitStatusTool(repo_root=repo_root))
    registry.register(GitDiffTool(repo_root=repo_root))
    registry.register(GitLogTool(repo_root=repo_root))
    return registry
