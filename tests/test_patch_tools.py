"""Tests for patch tools: apply_patch and undo_patch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.models import ToolStatus
from agent.tools.patch_tools import (
    ApplyPatchTool,
    UndoPatchTool,
    compute_hash,
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

    def test_hunk_location_is_respected_when_content_is_duplicated(self, repo: Path):
        target = repo / "repeated.txt"
        target.write_text("same\nmiddle\nsame\n", encoding="utf-8")
        tool = ApplyPatchTool(repo_root=repo)

        result = tool.execute(
            tool_call_id="location",
            path="repeated.txt",
            unified_diff="@@ -3 +3 @@\n-same\n+changed",
        )

        assert result.status == ToolStatus.SUCCESS
        assert target.read_text(encoding="utf-8") == "same\nmiddle\nchanged\n"

    def test_wrong_hunk_location_uses_unique_content_fallback(self, repo: Path, monkeypatch):
        monkeypatch.setattr(
            "agent.tools.patch_tools._run_git_apply",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=1,
                stderr="forced strict failure",
                stdout="",
            ),
        )
        tool = ApplyPatchTool(repo_root=repo)

        result = tool.execute(
            tool_call_id="wrong-location",
            path="src/main/java/Calculator.java",
            unified_diff=(
                "@@ -999,8 +999,3 @@\n"
                "-        return a + b;\n"
                "+        return a + b + 1;\n"
            ),
        )

        assert result.status == ToolStatus.SUCCESS
        assert "唯一内容匹配回退" in result.output
        assert "return a + b + 1;" in (
            repo / "src" / "main" / "java" / "Calculator.java"
        ).read_text(encoding="utf-8")

    def test_content_fallback_rejects_ambiguous_replacement(self, repo: Path):
        target = repo / "repeated.txt"
        target.write_text("same\nmiddle\nsame\n", encoding="utf-8")
        tool = ApplyPatchTool(repo_root=repo)

        result = tool.execute(
            tool_call_id="ambiguous",
            path="repeated.txt",
            unified_diff="@@ -999 +999 @@\n-same\n+changed",
        )

        assert result.status == ToolStatus.EXECUTION_ERROR
        assert "歧义" in result.output
        assert target.read_text(encoding="utf-8") == "same\nmiddle\nsame\n"


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

    def test_undo_patch_created_by_content_fallback(self, repo: Path):
        diff = "@@ -999,9 +999,2 @@\n-        return a + b;\n+        return a - b;"
        apply_result = ApplyPatchTool(repo_root=repo).execute(
            tool_call_id="fallback-apply",
            path="src/main/java/Calculator.java",
            unified_diff=diff,
        )
        assert apply_result.status == ToolStatus.SUCCESS

        undo_result = UndoPatchTool(repo_root=repo).execute(
            tool_call_id="fallback-undo",
            path="src/main/java/Calculator.java",
            unified_diff=diff,
            hash_before=apply_result.metadata["patch_record"]["content_hash_before"],
        )

        assert undo_result.status == ToolStatus.SUCCESS
        assert "return a + b;" in (
            repo / "src" / "main" / "java" / "Calculator.java"
        ).read_text(encoding="utf-8")
