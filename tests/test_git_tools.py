"""Tests for git tools: git_status, git_diff, git_log."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.models import ToolStatus
from agent.tools.git_tools import GitDiffTool, GitLogTool, GitStatusTool


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a git repository with initial commit."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True, check=True)

    src = tmp_path / "src"
    src.mkdir()
    (src / "Main.java").write_text("public class Main {}", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init commit"], cwd=str(tmp_path), capture_output=True, check=True)

    return tmp_path


class TestGitStatusTool:
    """Tests for git_status tool."""

    def test_clean_repo(self, git_repo: Path):
        tool = GitStatusTool(repo_root=git_repo)
        result = tool.execute(tool_call_id="t1")
        assert result.status == ToolStatus.SUCCESS
        # Clean repo shows nothing or only branch info
        assert "M " not in result.output and "??" not in result.output

    def test_dirty_repo(self, git_repo: Path):
        # Modify a file
        (git_repo / "src" / "Main.java").write_text("public class Main { modified }", encoding="utf-8")
        tool = GitStatusTool(repo_root=git_repo)
        result = tool.execute(tool_call_id="t2")
        assert result.status == ToolStatus.SUCCESS
        assert "Main.java" in result.output


class TestGitDiffTool:
    """Tests for git_diff tool."""

    def test_no_changes(self, git_repo: Path):
        tool = GitDiffTool(repo_root=git_repo)
        result = tool.execute(tool_call_id="t1")
        assert result.status == ToolStatus.SUCCESS
        assert "没有变更" in result.output

    def test_with_changes(self, git_repo: Path):
        (git_repo / "src" / "Main.java").write_text("public class Main { changed }", encoding="utf-8")
        tool = GitDiffTool(repo_root=git_repo)
        result = tool.execute(tool_call_id="t2")
        assert result.status == ToolStatus.SUCCESS
        assert "Main.java" in result.output

    def test_diff_specific_file(self, git_repo: Path):
        (git_repo / "src" / "Main.java").write_text("public class Main { changed }", encoding="utf-8")
        tool = GitDiffTool(repo_root=git_repo)
        result = tool.execute(tool_call_id="t3", path="src/Main.java")
        assert result.status == ToolStatus.SUCCESS

    def test_staged_diff(self, git_repo: Path):
        (git_repo / "src" / "Main.java").write_text(
            "public class Main { staged }",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "src/Main.java"], cwd=git_repo, check=True)
        tool = GitDiffTool(repo_root=git_repo)

        unstaged = tool.execute(tool_call_id="unstaged")
        staged = tool.execute(tool_call_id="staged", staged=True)

        assert "没有变更" in unstaged.output
        assert "Main.java" in staged.output

    def test_non_git_directory_returns_real_git_error(self, tmp_path: Path):
        tool = GitDiffTool(repo_root=tmp_path)

        result = tool.execute(tool_call_id="not-a-repo")

        assert result.status == ToolStatus.EXECUTION_ERROR
        assert "Git 命令失败" in result.output
        assert "当前快照" not in result.output


class TestGitLogTool:
    """Tests for git_log tool."""

    def test_log(self, git_repo: Path):
        tool = GitLogTool(repo_root=git_repo)
        result = tool.execute(tool_call_id="t1")
        assert result.status == ToolStatus.SUCCESS
        assert "init commit" in result.output

    def test_log_with_count(self, git_repo: Path):
        tool = GitLogTool(repo_root=git_repo)
        result = tool.execute(tool_call_id="t2", count=1)
        assert result.status == ToolStatus.SUCCESS
