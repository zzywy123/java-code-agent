"""Tests for file tools: list_files and read_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.models import ToolStatus
from agent.tools.file_tools import ListFilesTool, ReadFileTool


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a sample repository."""
    src = tmp_path / "src" / "main" / "java" / "com" / "example"
    src.mkdir(parents=True)
    (src / "Main.java").write_text(
        "package com.example;\n\npublic class Main {\n"
        "    public static void main(String[] args) {\n"
        '        System.out.println("Hello");\n'
        "    }\n}\n",
        encoding="utf-8",
    )
    (src / "Utils.java").write_text(
        "package com.example;\n\npublic class Utils {\n"
        '    public static String format(String s) { return s.trim(); }\n'
        "}\n",
        encoding="utf-8",
    )
    # Create a subdirectory
    (tmp_path / "src" / "test").mkdir(parents=True)
    return tmp_path


class TestListFilesTool:
    """Tests for list_files tool."""

    def test_list_root(self, repo: Path):
        tool = ListFilesTool(repo_root=repo)
        result = tool.execute(tool_call_id="t1", path=".")
        assert result.status == ToolStatus.SUCCESS
        assert "src" in result.output

    def test_list_subdirectory(self, repo: Path):
        tool = ListFilesTool(repo_root=repo)
        result = tool.execute(tool_call_id="t2", path="src/main/java/com/example")
        assert result.status == ToolStatus.SUCCESS
        assert "Main.java" in result.output
        assert "Utils.java" in result.output

    def test_list_with_pattern(self, repo: Path):
        tool = ListFilesTool(repo_root=repo)
        result = tool.execute(tool_call_id="t3", path="src/main/java/com/example", pattern="*.java")
        assert result.status == ToolStatus.SUCCESS
        assert "Main.java" in result.output

    def test_list_nonexistent(self, repo: Path):
        tool = ListFilesTool(repo_root=repo)
        result = tool.execute(tool_call_id="t4", path="nonexistent")
        assert result.status == ToolStatus.NOT_FOUND

    def test_list_path_traversal(self, repo: Path):
        tool = ListFilesTool(repo_root=repo)
        result = tool.execute(tool_call_id="t5", path="../../etc")
        assert result.status == ToolStatus.DENIED


class TestReadFileTool:
    """Tests for read_file tool."""

    def test_read_full_file(self, repo: Path):
        tool = ReadFileTool(repo_root=repo)
        result = tool.execute(tool_call_id="t1", path="src/main/java/com/example/Main.java")
        assert result.status == ToolStatus.SUCCESS
        assert "public class Main" in result.output
        assert "1 |" in result.output  # Line numbers

    def test_read_line_range(self, repo: Path):
        tool = ReadFileTool(repo_root=repo)
        result = tool.execute(tool_call_id="t2", path="src/main/java/com/example/Main.java", start_line=3, end_line=5)
        assert result.status == ToolStatus.SUCCESS
        assert "public class Main" in result.output
        # Should not include line 1
        assert "package com.example" not in result.output

    def test_read_nonexistent(self, repo: Path):
        tool = ReadFileTool(repo_root=repo)
        result = tool.execute(tool_call_id="t3", path="Missing.java")
        assert result.status == ToolStatus.NOT_FOUND

    def test_read_no_path(self, repo: Path):
        tool = ReadFileTool(repo_root=repo)
        result = tool.execute(tool_call_id="t4", path="")
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_read_path_traversal(self, repo: Path):
        tool = ReadFileTool(repo_root=repo)
        result = tool.execute(tool_call_id="t5", path="../../etc/passwd")
        assert result.status == ToolStatus.DENIED

    def test_read_line_out_of_range(self, repo: Path):
        tool = ReadFileTool(repo_root=repo)
        result = tool.execute(tool_call_id="t6", path="src/main/java/com/example/Main.java", start_line=1000)
        assert result.status == ToolStatus.INVALID_ARGUMENT
