"""Tests for build tools: run_tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.models import ToolStatus
from agent.tools.build_tools import RunTestsTool


class TestRunTestsTool:
    """Tests for run_tests tool."""

    def test_invalid_tool(self, tmp_path: Path):
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t1", tool="invalid", goals=["test"])
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_empty_goals(self, tmp_path: Path):
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t2", tool="maven", goals=[])
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_no_tool(self, tmp_path: Path):
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t3", tool="", goals=["test"])
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_invalid_extra_arg(self, tmp_path: Path):
        """Invalid extra args should be rejected by validate_argv."""
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(
            tool_call_id="t5",
            tool="maven",
            goals=["test"],
            extra_args=["--malicious"],
        )
        # Should be rejected (DENIED or INVALID_ARGUMENT)
        assert result.status in {ToolStatus.DENIED, ToolStatus.INVALID_ARGUMENT}

    def test_maven_execution(self, tmp_path: Path):
        """When mvn is available, it should execute (may fail due to no pom.xml)."""
        import shutil
        if not shutil.which("mvn"):
            pytest.skip("mvn not in PATH")
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t6", tool="maven", goals=["test"])
        # May fail for various reasons (no pom.xml, validation issues, etc.)
        assert result.status in {
            ToolStatus.EXECUTION_ERROR, ToolStatus.NOT_FOUND,
            ToolStatus.SUCCESS, ToolStatus.DENIED,
        }

    def test_gradle_not_found(self, tmp_path: Path):
        """When gradle is not in PATH, it should return NOT_FOUND."""
        import shutil
        if shutil.which("gradle"):
            pytest.skip("gradle is in PATH")
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t7", tool="gradle", goals=["test"])
        assert result.status == ToolStatus.NOT_FOUND
