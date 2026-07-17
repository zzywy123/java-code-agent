"""Git tools: git_status, git_diff, git_log.

These are read-only git operations that provide repository context.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent.models import ToolResult, ToolStatus
from agent.tools.base import BaseTool


class GitStatusTool(BaseTool):
    """Show git working tree status."""

    name = "git_status"
    description = (
        "显示 Git 工作区状态。"
        "列出已修改、已暂存、未跟踪的文件。"
        "用于了解当前有哪些变更。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, tool_call_id: str = "", **kwargs: Any) -> ToolResult:
        return self._run_git(tool_call_id, ["git", "status", "--short", "--branch"])

    def _run_git(self, tool_call_id: str, argv: list[str]) -> ToolResult:
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.repo_root),
                capture_output=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return self._error_result(tool_call_id, ToolStatus.TIMEOUT, "Git 命令超时")
        except FileNotFoundError:
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, "Git 未安装")
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"Git 执行失败: {e}")

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            return self._error_result(
                tool_call_id,
                ToolStatus.EXECUTION_ERROR,
                f"Git 命令失败 (exit {proc.returncode}): {stderr}",
            )

        if not stdout:
            return self._success_result(tool_call_id, "工作区干净，没有未提交的变更")

        return self._success_result(tool_call_id, f"Git Status:\n{stdout}")


class GitDiffTool(BaseTool):
    """Show git diff for working tree or staged changes."""

    name = "git_diff"
    description = (
        "显示 Git diff 内容。"
        "可查看工作区或暂存区的变更。"
        "可指定文件路径查看特定文件的变更。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要查看 diff 的文件路径（相对于仓库根目录），默认查看所有变更",
                "default": "",
            },
            "staged": {
                "type": "boolean",
                "description": "是否查看暂存区的 diff（默认 false，查看工作区）",
                "default": False,
            },
        },
        "required": [],
    }

    MAX_DIFF_CHARS = 20000

    def execute(
        self,
        tool_call_id: str = "",
        path: str = "",
        staged: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        argv = ["git", "diff"]
        if staged:
            argv.append("--staged")

        if path:
            try:
                target = self.validate_path(path)
                argv.append(str(target))
            except Exception as e:
                return self._error_result(tool_call_id, ToolStatus.DENIED, str(e))

        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.repo_root),
                capture_output=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return self._error_result(tool_call_id, ToolStatus.TIMEOUT, "Git diff 超时")
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"Git diff 失败: {e}")

        output = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return self._error_result(
                tool_call_id,
                ToolStatus.EXECUTION_ERROR,
                f"Git 命令失败 (exit {proc.returncode}): {stderr}",
            )
        if not output:
            return self._success_result(tool_call_id, "没有变更")

        truncated = False
        if len(output) > self.MAX_DIFF_CHARS:
            output = output[:self.MAX_DIFF_CHARS] + "\n\n[Diff 截断]"
            truncated = True

        return self._success_result(
            tool_call_id,
            f"Git Diff:\n{output}",
            {"truncated": truncated},
        )


class GitLogTool(BaseTool):
    """Show recent git commit log."""

    name = "git_log"
    description = (
        "显示最近的 Git 提交记录。"
        "包含提交哈希、作者、日期和提交信息。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "显示的提交数量（默认 10）",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": [],
    }

    def execute(self, tool_call_id: str = "", count: int = 10, **kwargs: Any) -> ToolResult:
        argv = [
            "git", "log",
            f"--max-count={count}",
            "--format=%h %ad %an: %s",
            "--date=short",
        ]

        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.repo_root),
                capture_output=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return self._error_result(tool_call_id, ToolStatus.TIMEOUT, "Git log 超时")
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"Git log 失败: {e}")

        output = (proc.stdout or "").strip()
        if not output:
            return self._success_result(tool_call_id, "没有提交记录")

        return self._success_result(tool_call_id, f"Git Log (最近 {count} 条):\n{output}")
