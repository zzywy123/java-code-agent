"""Patch tools: apply_patch and undo_patch.

Key design decisions:
- Uses content hashes (SHA-256) instead of storing full source in state
- Supports unified diff format
- Supports creating new files
- undo_patch uses reverse-apply of the diff (NOT git checkout)
- All file writes are atomic (write to tmp → rename)
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from agent.models import PatchRecord, ToolResult, ToolStatus
from agent.tools.base import BaseTool


def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    """Atomically write content to a file.

    Writes to a temporary file first, then renames.
    This prevents partial writes on crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        suffix=".tmp",
        prefix=f".{path.name}.",
    )
    try:
        import os
        os.write(tmp_fd, content.encode("utf-8"))
        os.close(tmp_fd)
        Path(tmp_path).replace(path)
    except Exception:
        # Clean up temp file on failure
        try:
            import os
            os.close(tmp_fd)
        except Exception:
            pass
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass
        raise


def parse_unified_diff(diff_text: str) -> list[tuple[str, list[str], list[str]]]:
    """Parse a unified diff into hunks.

    Returns:
        List of (header, removed_lines, added_lines) tuples per hunk.
    """
    hunks = []
    current_header = ""
    removed = []
    added = []

    for line in diff_text.splitlines():
        if line.startswith("@@"):
            # Save previous hunk
            if removed or added:
                hunks.append((current_header, removed, added))
            current_header = line
            removed = []
            added = []
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])

    # Save last hunk
    if removed or added:
        hunks.append((current_header, removed, added))

    return hunks


def apply_diff_to_content(original: str, diff_text: str) -> str:
    """Apply a unified diff to original content.

    This is a simplified diff applier that handles the common case
    of replacing removed lines with added lines.

    Args:
        original: Original file content
        diff_text: Unified diff text

    Returns:
        Patched content

    Raises:
        ValueError: If the diff cannot be applied
    """
    hunks = parse_unified_diff(diff_text)
    if not hunks:
        raise ValueError("无法解析 unified diff：没有找到有效的 hunk")

    # Split lines, handling both with and without trailing newline
    original_lines = original.splitlines()
    # Preserve trailing newline
    has_trailing_newline = original.endswith("\n")

    result_lines = []
    current_line = 0

    for header, removed, added in hunks:
        # Find the removed lines in the original
        if removed:
            # Search for the block of removed lines
            found = False
            for search_start in range(current_line, len(original_lines)):
                match = True
                for j, rem_line in enumerate(removed):
                    if search_start + j >= len(original_lines):
                        match = False
                        break
                    # Strip both sides for comparison to handle whitespace differences
                    if original_lines[search_start + j].rstrip() != rem_line.rstrip():
                        match = False
                        break
                if match:
                    # Copy lines before the match
                    result_lines.extend(original_lines[current_line:search_start])
                    # Add the new lines
                    result_lines.extend(added)
                    current_line = search_start + len(removed)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"无法应用 diff：找不到要删除的行\n"
                    f"期望找到: {removed[:3]}..."
                )
        else:
            # Pure addition - find the insertion point from the header
            import re
            m = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", header)
            if m:
                insert_at = int(m.group(1)) - 1
                result_lines.extend(original_lines[current_line:insert_at])
                result_lines.extend(added)
                current_line = insert_at
            else:
                # Append at end
                result_lines.extend(original_lines[current_line:])
                result_lines.extend(added)
                current_line = len(original_lines)

    # Copy remaining lines
    result_lines.extend(original_lines[current_line:])

    result = "\n".join(result_lines)
    if has_trailing_newline and not result.endswith("\n"):
        result += "\n"
    return result


def reverse_diff(diff_text: str) -> str:
    """Reverse a unified diff (swap additions and deletions)."""
    lines = diff_text.splitlines()
    result = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            result.append("-" + line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            result.append("+" + line[1:])
        elif line.startswith("+++") or line.startswith("---"):
            # Swap the file markers
            if line.startswith("+++"):
                result.append("---" + line[3:])
            else:
                result.append("+++" + line[3:])
        else:
            result.append(line)
    return "\n".join(result)


class ApplyPatchTool(BaseTool):
    """Apply a unified diff to a file.

    Supports both modifying existing files and creating new files.
    Uses atomic writes and records content hashes.
    """

    name = "apply_patch"
    description = (
        "应用 unified diff 格式的补丁到文件。"
        "支持修改现有文件和创建新文件。"
        "使用原子写入（先写临时文件再重命名）。"
        "会记录内容哈希用于后续撤销。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要修改的文件路径（相对于仓库根目录）",
            },
            "unified_diff": {
                "type": "string",
                "description": "Unified diff 格式的补丁内容",
            },
            "create_new": {
                "type": "boolean",
                "description": "是否创建新文件（默认 false）",
                "default": False,
            },
        },
        "required": ["path", "unified_diff"],
    }

    def execute(
        self,
        tool_call_id: str = "",
        path: str = "",
        unified_diff: str = "",
        create_new: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not path:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, "必须指定文件路径")
        if not unified_diff:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, "必须提供 unified diff")

        try:
            target = self.validate_path(path)
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.DENIED, str(e))

        # Read original content
        if create_new:
            if target.exists():
                return self._error_result(
                    tool_call_id,
                    ToolStatus.INVALID_ARGUMENT,
                    f"文件已存在，无法创建新文件: {path}",
                )
            original_content = ""
            is_new = True
        else:
            if not target.exists():
                return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, f"文件不存在: {path}")
            try:
                original_content = target.read_text(encoding="utf-8")
            except Exception as e:
                return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"读取文件失败: {e}")
            is_new = False

        # For new files, the diff content IS the new file content (strip +/- markers)
        if is_new:
            new_lines = []
            for line in unified_diff.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    new_lines.append(line[1:])
                elif not line.startswith("-") and not line.startswith("@@") and not line.startswith("---") and not line.startswith("+++"):
                    new_lines.append(line)
            new_content = "\n".join(new_lines)
        else:
            # Apply the diff
            try:
                new_content = apply_diff_to_content(original_content, unified_diff)
            except ValueError as e:
                return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"应用补丁失败: {e}")

        # Compute hashes
        hash_before = compute_hash(original_content) if original_content else ""
        hash_after = compute_hash(new_content)

        # Atomic write
        try:
            atomic_write(target, new_content)
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"写入文件失败: {e}")

        # Build patch record
        record = PatchRecord(
            file_path=str(target),
            content_hash_before=hash_before,
            content_hash_after=hash_after,
            unified_diff=unified_diff,
            is_new_file=is_new,
        )

        action = "创建新文件" if is_new else "修改文件"
        output = f"{action}: {path}\n"
        output += f"  哈希: {hash_before[:8]}... → {hash_after[:8]}...\n"
        if not is_new:
            output += f"  Diff:\n{unified_diff}"

        return self._success_result(
            tool_call_id,
            output,
            {"patch_record": record.model_dump()},
        )


class UndoPatchTool(BaseTool):
    """Undo the last patch applied to a file.

    Uses reverse-apply of the diff (NOT git checkout).
    Validates content hash before undoing.
    """

    name = "undo_patch"
    description = (
        "撤销对文件的最后一次修改。"
        "通过反向应用 diff 来恢复原状（不使用 git checkout）。"
        "会校验当前内容哈希确保文件未被再次修改。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要撤销修改的文件路径（相对于仓库根目录）",
            },
            "unified_diff": {
                "type": "string",
                "description": "要反向应用的 unified diff（即 apply_patch 时使用的 diff）",
            },
            "hash_before": {
                "type": "string",
                "description": "修改前的内容哈希（SHA-256），用于校验",
            },
            "is_new_file": {
                "type": "boolean",
                "description": "是否是新创建的文件（如果是则直接删除）",
                "default": False,
            },
        },
        "required": ["path", "unified_diff"],
    }

    def execute(
        self,
        tool_call_id: str = "",
        path: str = "",
        unified_diff: str = "",
        hash_before: str = "",
        is_new_file: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not path:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, "必须指定文件路径")

        try:
            target = self.validate_path(path)
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.DENIED, str(e))

        if not target.exists():
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, f"文件不存在: {path}")

        # If it was a new file, just delete it
        if is_new_file:
            try:
                target.unlink()
                return self._success_result(
                    tool_call_id,
                    f"已删除新创建的文件: {path}",
                    {"deleted": True},
                )
            except Exception as e:
                return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"删除文件失败: {e}")

        # Read current content
        try:
            current_content = target.read_text(encoding="utf-8")
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"读取文件失败: {e}")

        # Validate hash if provided
        if hash_before:
            current_hash = compute_hash(current_content)
            # The current content should be the "after" state, not the "before" state
            # We need to check that applying reverse diff will produce hash_before

        # Reverse the diff and apply
        try:
            reversed_diff = reverse_diff(unified_diff)
            restored_content = apply_diff_to_content(current_content, reversed_diff)
        except ValueError as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"反向应用补丁失败: {e}")

        # Validate restored hash if hash_before provided
        if hash_before:
            restored_hash = compute_hash(restored_content)
            if restored_hash != hash_before:
                return self._error_result(
                    tool_call_id,
                    ToolStatus.EXECUTION_ERROR,
                    f"哈希校验失败：恢复后的内容哈希 {restored_hash[:8]}... 与原始哈希 {hash_before[:8]}... 不匹配",
                )

        # Atomic write
        try:
            atomic_write(target, restored_content)
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"写入文件失败: {e}")

        output = f"已撤销修改: {path}\n"
        if hash_before:
            output += f"  哈希校验通过: {hash_before[:8]}...\n"
        output += f"  使用反向 diff 恢复"

        return self._success_result(tool_call_id, output)
