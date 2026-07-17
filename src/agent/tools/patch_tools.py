"""Patch tools backed by Git's unified-diff implementation."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent.models import PatchRecord, ToolResult, ToolStatus
from agent.tools.base import BaseTool


def compute_hash(content: str) -> str:
    """Compute a SHA-256 hash for patch integrity checks."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize_diff(diff_text: str, relative_path: str, *, create_new: bool) -> str:
    """Bind a model-produced single-file diff to the validated target path."""
    lines = diff_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    hunk_start = next((index for index, line in enumerate(lines) if line.startswith("@@")), None)

    if hunk_start is None:
        if not create_new:
            raise ValueError("无法解析 unified diff：没有找到有效的 hunk")
        added = [
            line[1:]
            for line in lines
            if line.startswith("+") and not line.startswith("+++")
        ]
        if not added:
            raise ValueError("新文件补丁没有新增内容")
        body = [f"@@ -0,0 +1,{len(added)} @@", *(f"+{line}" for line in added)]
    else:
        body = lines[hunk_start:]
        if any(
            line.startswith(("diff --git ", "--- ", "+++ "))
            for line in body[1:]
        ):
            raise ValueError("一次补丁调用只能修改 path 指定的一个文件")

    old_path = "/dev/null" if create_new else f"a/{relative_path}"
    header = [
        f"diff --git a/{relative_path} b/{relative_path}",
        *(["new file mode 100644"] if create_new else []),
        f"--- {old_path}",
        f"+++ b/{relative_path}",
    ]
    return "\n".join([*header, *body]) + "\n"


def _run_git_apply(repo_root: Path, patch: str, *, check: bool, reverse: bool = False) -> subprocess.CompletedProcess[str]:
    # LLMs often emit a minimal hunk with no context lines. Git requires
    # --unidiff-zero for that valid but less forgiving unified-diff form.
    argv = [
        "git",
        "apply",
        "--recount",
        "--unidiff-zero",
        "--whitespace=nowarn",
    ]
    if check:
        argv.append("--check")
    if reverse:
        argv.append("--reverse")
    argv.append("-")
    return subprocess.run(
        argv,
        cwd=str(repo_root),
        input=patch,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _atomic_write(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _parse_replacement_hunks(diff_text: str) -> list[tuple[str, list[str], list[str]]]:
    """Parse replacement hunks without trusting model-produced line counts."""
    lines = diff_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    hunk_starts = [index for index, line in enumerate(lines) if line.startswith("@@")]
    if not hunk_starts:
        raise ValueError("没有找到有效的 hunk")

    hunks: list[tuple[str, list[str], list[str]]] = []
    for position, start in enumerate(hunk_starts):
        end = hunk_starts[position + 1] if position + 1 < len(hunk_starts) else len(lines)
        old_lines: list[str] = []
        new_lines: list[str] = []
        removed_count = 0
        for line in lines[start + 1:end]:
            if line == r"\ No newline at end of file":
                continue
            if not line or line[0] not in {" ", "+", "-"}:
                raise ValueError("hunk 包含无法识别的行")
            marker, content = line[0], line[1:]
            if marker in {" ", "-"}:
                old_lines.append(content)
            if marker in {" ", "+"}:
                new_lines.append(content)
            if marker == "-":
                removed_count += 1
        if removed_count == 0:
            raise ValueError("安全回退只支持包含删除行的替换补丁")
        hunks.append((lines[start], old_lines, new_lines))
    return hunks


def _matching_blocks(lines: list[str], expected: list[str]) -> list[int]:
    """Return exact, or otherwise trailing-whitespace-normalized, matches."""
    if not expected or len(expected) > len(lines):
        return []

    exact = [
        index
        for index in range(len(lines) - len(expected) + 1)
        if lines[index:index + len(expected)] == expected
    ]
    if exact:
        return exact

    return [
        index
        for index in range(len(lines) - len(expected) + 1)
        if all(
            lines[index + offset].rstrip() == expected_line.rstrip()
            for offset, expected_line in enumerate(expected)
        )
    ]


def _find_unique_block(lines: list[str], expected: list[str]) -> int:
    """Find exactly one block, allowing trailing-whitespace differences."""
    matches = _matching_blocks(lines, expected)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("要替换的代码块出现多次，拒绝歧义修改")
    raise ValueError("找不到要替换的代码块")


def _validate_patch_ambiguity(original: str, diff_text: str) -> None:
    """Reject a duplicate target unless the hunk location selects one exactly."""
    try:
        hunks = _parse_replacement_hunks(diff_text)
    except ValueError:
        return

    lines = original.splitlines()
    for header, old_lines, _ in hunks:
        matches = _matching_blocks(lines, old_lines)
        if len(matches) <= 1:
            continue
        location = re.match(r"@@\s+-(\d+)", header)
        hinted_index = int(location.group(1)) - 1 if location else -1
        if hinted_index not in matches:
            raise ValueError("要替换的代码块出现多次，且 hunk 行号无法消除歧义")


def _apply_content_fallback(original: str, diff_text: str, *, reverse: bool = False) -> str:
    """Apply unambiguous replacements when Git rejects malformed hunk metadata."""
    hunks = _parse_replacement_hunks(diff_text)
    current = original.splitlines()
    for _, old_lines, new_lines in hunks:
        expected, replacement = (
            (new_lines, old_lines) if reverse else (old_lines, new_lines)
        )
        start = _find_unique_block(current, expected)
        current[start:start + len(expected)] = replacement

    result = "\n".join(current)
    if original.endswith(("\n", "\r")):
        result += "\n"
    return result


class ApplyPatchTool(BaseTool):
    """Validate and apply one unified diff using ``git apply``."""

    name = "apply_patch"
    description = (
        "应用 unified diff 格式的单文件补丁。"
        "补丁先通过 git apply --check 校验，再由 Git 原生算法应用。"
        "支持修改现有文件和创建新文件，并记录内容哈希用于撤销。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目标文件路径（相对于仓库根目录）"},
            "unified_diff": {"type": "string", "description": "Unified diff 格式的补丁内容"},
            "create_new": {"type": "boolean", "description": "是否创建新文件", "default": False},
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
            relative_path = target.relative_to(self.repo_root.resolve()).as_posix()
        except Exception as exc:
            return self._error_result(tool_call_id, ToolStatus.DENIED, str(exc))

        if create_new and target.exists():
            return self._error_result(
                tool_call_id, ToolStatus.INVALID_ARGUMENT, f"文件已存在，无法创建新文件: {path}"
            )
        if not create_new and not target.exists():
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, f"文件不存在: {path}")

        try:
            original_content = target.read_text(encoding="utf-8") if target.exists() else ""
            if not create_new:
                _validate_patch_ambiguity(original_content, unified_diff)
            normalized = _normalize_diff(unified_diff, relative_path, create_new=create_new)
            checked = _run_git_apply(self.repo_root, normalized, check=True)
        except FileNotFoundError:
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, "找不到 git 命令")
        except subprocess.TimeoutExpired:
            return self._error_result(tool_call_id, ToolStatus.TIMEOUT, "补丁校验超时")
        except (OSError, UnicodeError, ValueError) as exc:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"应用补丁失败: {exc}")

        if checked.returncode != 0:
            git_reason = (checked.stderr or checked.stdout).strip()
            if create_new:
                return self._error_result(
                    tool_call_id,
                    ToolStatus.EXECUTION_ERROR,
                    f"git apply --check 失败: {git_reason}",
                )
            try:
                new_content = _apply_content_fallback(original_content, unified_diff)
                _atomic_write(target, new_content)
                apply_mode = "唯一内容匹配回退"
            except (OSError, UnicodeError, ValueError) as exc:
                return self._error_result(
                    tool_call_id,
                    ToolStatus.EXECUTION_ERROR,
                    f"git apply --check 失败: {git_reason}; 安全回退失败: {exc}",
                )
        else:
            try:
                applied = _run_git_apply(self.repo_root, normalized, check=False)
            except FileNotFoundError:
                return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, "找不到 git 命令")
            except subprocess.TimeoutExpired:
                return self._error_result(tool_call_id, ToolStatus.TIMEOUT, "应用补丁超时")
            if applied.returncode != 0:
                reason = (applied.stderr or applied.stdout).strip()
                return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"git apply 失败: {reason}")
            new_content = target.read_text(encoding="utf-8")
            apply_mode = "git apply"

        hash_before = compute_hash(original_content) if not create_new else ""
        hash_after = compute_hash(new_content)
        record = PatchRecord(
            file_path=str(target),
            content_hash_before=hash_before,
            content_hash_after=hash_after,
            unified_diff=unified_diff,
            is_new_file=create_new,
        )
        action = "创建新文件" if create_new else "修改文件"
        return self._success_result(
            tool_call_id,
            f"{action}: {path}\n  方式: {apply_mode}\n  哈希: {hash_before[:8]}... -> {hash_after[:8]}...",
            {"patch_record": record.model_dump()},
        )


class UndoPatchTool(BaseTool):
    """Reverse one previously applied patch using ``git apply --reverse``."""

    name = "undo_patch"
    description = (
        "撤销最后一次单文件补丁。"
        "先通过 git apply --reverse --check 校验，再反向应用并核对原始内容哈希。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要撤销的文件路径"},
            "unified_diff": {"type": "string", "description": "apply_patch 时使用的 unified diff"},
            "hash_before": {"type": "string", "description": "修改前的 SHA-256 内容哈希"},
            "is_new_file": {"type": "boolean", "description": "补丁是否创建了新文件", "default": False},
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
        if not unified_diff:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, "必须提供 unified diff")
        try:
            target = self.validate_path(path)
            relative_path = target.relative_to(self.repo_root.resolve()).as_posix()
        except Exception as exc:
            return self._error_result(tool_call_id, ToolStatus.DENIED, str(exc))
        if not target.exists():
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, f"文件不存在: {path}")

        current_content = target.read_text(encoding="utf-8")
        try:
            normalized = _normalize_diff(unified_diff, relative_path, create_new=is_new_file)
            checked = _run_git_apply(self.repo_root, normalized, check=True, reverse=True)
        except FileNotFoundError:
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, "找不到 git 命令")
        except subprocess.TimeoutExpired:
            return self._error_result(tool_call_id, ToolStatus.TIMEOUT, "撤销补丁校验超时")
        except ValueError as exc:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"撤销补丁失败: {exc}")
        if checked.returncode != 0:
            git_reason = (checked.stderr or checked.stdout).strip()
            if is_new_file:
                return self._error_result(
                    tool_call_id,
                    ToolStatus.EXECUTION_ERROR,
                    f"git apply --reverse --check 失败: {git_reason}",
                )
            try:
                restored_content = _apply_content_fallback(
                    current_content,
                    unified_diff,
                    reverse=True,
                )
                _atomic_write(target, restored_content)
                undo_mode = "唯一内容匹配回退"
            except (OSError, UnicodeError, ValueError) as exc:
                return self._error_result(
                    tool_call_id,
                    ToolStatus.EXECUTION_ERROR,
                    f"git apply --reverse --check 失败: {git_reason}; 安全回退失败: {exc}",
                )
        else:
            try:
                reversed_result = _run_git_apply(
                    self.repo_root,
                    normalized,
                    check=False,
                    reverse=True,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                status = ToolStatus.NOT_FOUND if isinstance(exc, FileNotFoundError) else ToolStatus.TIMEOUT
                return self._error_result(tool_call_id, status, f"撤销补丁失败: {exc}")
            if reversed_result.returncode != 0:
                reason = (reversed_result.stderr or reversed_result.stdout).strip()
                return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"git apply --reverse 失败: {reason}")
            restored_content = target.read_text(encoding="utf-8") if target.exists() else ""
            undo_mode = "git apply --reverse"

        if is_new_file:
            return self._success_result(tool_call_id, f"已删除新创建的文件: {path}", {"deleted": True})

        if hash_before and compute_hash(restored_content) != hash_before:
            try:
                _atomic_write(target, current_content)
                rollback_note = "，已恢复撤销前状态"
            except OSError:
                rollback_note = "，且自动恢复失败"
            return self._error_result(
                tool_call_id,
                ToolStatus.EXECUTION_ERROR,
                f"哈希校验失败：恢复内容与原始内容不一致{rollback_note}",
            )
        return self._success_result(
            tool_call_id,
            f"已撤销修改: {path}\n  方式: {undo_mode}\n  哈希校验通过",
        )
