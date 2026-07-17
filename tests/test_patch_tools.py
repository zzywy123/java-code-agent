"""Tests for patch tools: apply_patch and undo_patch."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.models import ToolStatus
from agent.tools.patch_tools import (
    ApplyPatchTool,
    UndoPatchTool,
    apply_diff_to_content,
    compute_hash,
    reverse_diff,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a sample repository with a Java file."""
    src = tmp_path / "src" / "main" / "java"
    src.mkdir(parents=True)
    (src / "Calculator.java").write_text(
        "package com.example;\n\n"
        "public class Calculator {\n"
        "    public int add(int a, int b) {\n"
        "        return a + b;\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    return tmp_path


class TestComputeHash:
    """Tests for compute_hash function."""

    def test_deterministic(self):
        h1 = compute_hash("hello")
        h2 = compute_hash("hello")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_hash("hello")
        h2 = compute_hash("world")
        assert h1 != h2

    def test_empty_string(self):
        h = compute_hash("")
        assert len(h) == 64  # SHA-256 hex length


class TestDiffOperations:
    """Tests for diff parsing and application."""

    def test_apply_simple_diff(self):
        original = "line1\nold line2\nline3\n"
        diff = "@@ -1,3 +1,3 @@\n line1\n-old line2\n+new line2\n line3"
        result = apply_diff_to_content(original, diff)
        assert "new line2" in result
        assert "old line2" not in result

    def test_reverse_diff(self):
        diff = "@@ -1,3 +1,3 @@\n line1\n-old\n+new\n line3"
        reversed_diff = reverse_diff(diff)
        assert "+old" in reversed_diff
        assert "-new" in reversed_diff


class TestApplyPatchTool:
    """Tests for apply_patch tool."""

    def test_apply_patch(self, repo: Path):
        tool = ApplyPatchTool(repo_root=repo)
        diff = (
            "@@ -3,3 +3,3 @@\n"
            "     public int add(int a, int b) {\n"
            "-        return a + b;\n"
            "+        return a + b + 1; // bug fix\n"
            "     }"
        )
        result = tool.execute(
            tool_call_id="t1",
            path="src/main/java/Calculator.java",
            unified_diff=diff,
        )
        assert result.status == ToolStatus.SUCCESS
        assert "修改文件" in result.output

        # Verify the file was actually modified
        content = (repo / "src" / "main" / "java" / "Calculator.java").read_text()
        assert "return a + b + 1" in content

    def test_apply_patch_nonexistent(self, repo: Path):
        tool = ApplyPatchTool(repo_root=repo)
        result = tool.execute(
            tool_call_id="t2",
            path="Missing.java",
            unified_diff="@@ -1 +1 @@\n-old\n+new",
        )
        assert result.status == ToolStatus.NOT_FOUND

    def test_apply_patch_no_diff(self, repo: Path):
        tool = ApplyPatchTool(repo_root=repo)
        result = tool.execute(
            tool_call_id="t3",
            path="src/main/java/Calculator.java",
            unified_diff="",
        )
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_apply_patch_path_traversal(self, repo: Path):
        tool = ApplyPatchTool(repo_root=repo)
        result = tool.execute(
            tool_call_id="t4",
            path="../../etc/passwd",
            unified_diff="@@ -1 +1 @@\n-old\n+new",
        )
        assert result.status == ToolStatus.DENIED

    def test_create_new_file(self, repo: Path):
        tool = ApplyPatchTool(repo_root=repo)
        diff = "+++ New.java\n+package com.example;\n+\n+public class New {}\n"
        result = tool.execute(
            tool_call_id="t5",
            path="src/main/java/New.java",
            unified_diff=diff,
            create_new=True,
        )
        assert result.status == ToolStatus.SUCCESS
        assert "创建新文件" in result.output
        assert (repo / "src" / "main" / "java" / "New.java").exists()

    def test_create_new_file_already_exists(self, repo: Path):
        tool = ApplyPatchTool(repo_root=repo)
        result = tool.execute(
            tool_call_id="t6",
            path="src/main/java/Calculator.java",
            unified_diff="+content",
            create_new=True,
        )
        assert result.status == ToolStatus.INVALID_ARGUMENT


class TestUndoPatchTool:
    """Tests for undo_patch tool."""

    def test_undo_patch(self, repo: Path):
        tool = ApplyPatchTool(repo_root=repo)
        diff = (
            "@@ -3,3 +3,3 @@\n"
            "     public int add(int a, int b) {\n"
            "-        return a + b;\n"
            "+        return a + b + 1;\n"
            "     }"
        )
        result = tool.execute(
            tool_call_id="t1",
            path="src/main/java/Calculator.java",
            unified_diff=diff,
        )
        assert result.status == ToolStatus.SUCCESS

        # Get the hash from the patch record
        patch_data = result.metadata["patch_record"]
        hash_before = patch_data["content_hash_before"]

        # Verify file was modified
        content = (repo / "src" / "main" / "java" / "Calculator.java").read_text()
        assert "return a + b + 1" in content

        # Undo the patch
        undo_tool = UndoPatchTool(repo_root=repo)
        result = undo_tool.execute(
            tool_call_id="t2",
            path="src/main/java/Calculator.java",
            unified_diff=diff,
            hash_before=hash_before,
        )
        assert result.status == ToolStatus.SUCCESS

        # Verify file was restored
        content = (repo / "src" / "main" / "java" / "Calculator.java").read_text()
        assert "return a + b;" in content
        assert "return a + b + 1" not in content

    def test_undo_new_file(self, repo: Path):
        # Create a new file
        tool = ApplyPatchTool(repo_root=repo)
        diff = "+++ New.java\n+package com.example;\n+public class New {}\n"
        result = tool.execute(
            tool_call_id="t3",
            path="src/main/java/New.java",
            unified_diff=diff,
            create_new=True,
        )
        assert result.status == ToolStatus.SUCCESS
        assert (repo / "src" / "main" / "java" / "New.java").exists()

        # Undo (delete) the new file
        undo_tool = UndoPatchTool(repo_root=repo)
        result = undo_tool.execute(
            tool_call_id="t4",
            path="src/main/java/New.java",
            unified_diff=diff,
            is_new_file=True,
        )
        assert result.status == ToolStatus.SUCCESS
        assert "已删除" in result.output
        assert not (repo / "src" / "main" / "java" / "New.java").exists()

    def test_undo_nonexistent(self, repo: Path):
        tool = UndoPatchTool(repo_root=repo)
        result = tool.execute(
            tool_call_id="t5",
            path="Missing.java",
            unified_diff="@@ -1 +1 @@\n-old\n+new",
        )
        assert result.status == ToolStatus.NOT_FOUND
