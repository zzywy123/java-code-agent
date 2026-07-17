"""File operation tools: list_files and read_file.

Both tools are read-only and operate within the repository boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.models import ToolResult, ToolStatus
from agent.tools.base import BaseTool


class ListFilesTool(BaseTool):
    """List files and directories in a given path."""

    name = "list_files"
    description = (
        "列出指定目录下的文件和子目录。"
        "返回相对路径列表，标注是文件还是目录。"
        "用于了解项目结构和文件布局。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要列出的目录路径（相对于仓库根目录），默认为 '.'",
                "default": ".",
            },
            "pattern": {
                "type": "string",
                "description": "文件名 glob 模式，如 '*.java'，默认列出所有",
                "default": "*",
            },
        },
        "required": [],
    }

    def execute(self, tool_call_id: str = "", path: str = ".", pattern: str = "*", **kwargs: Any) -> ToolResult:
        try:
            target = self.validate_path(path)
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.DENIED, str(e))

        if not target.exists():
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, f"目录不存在: {path}")
        if not target.is_dir():
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, f"不是目录: {path}")

        try:
            entries = sorted(target.glob(pattern))
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"列出文件失败: {e}")

        lines = []
        for entry in entries:
            # Skip excluded directories
            if entry.name.startswith(".") and entry.is_dir():
                continue
            rel = entry.relative_to(self.repo_root)
            prefix = "📁" if entry.is_dir() else "📄"
            lines.append(f"{prefix} {rel.as_posix()}")

        if not lines:
            return self._success_result(tool_call_id, f"目录为空: {path}")

        output = f"目录 {path} 内容 ({len(lines)} 项):\n" + "\n".join(lines)
        return self._success_result(tool_call_id, output, {"count": len(lines)})


class ReadFileTool(BaseTool):
    """Read file contents with optional line range."""

    name = "read_file"
    description = (
        "读取文件内容，支持指定行范围。"
        "返回文件内容及行号。"
        "用于查看代码实现、理解业务逻辑。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（相对于仓库根目录）",
            },
            "start_line": {
                "type": "integer",
                "description": "起始行号（从1开始），默认为1",
                "default": 1,
                "minimum": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "结束行号（包含），默认为-1表示到文件末尾",
                "default": -1,
            },
        },
        "required": ["path"],
    }

    MAX_LINES = 500  # Maximum lines to read at once

    def execute(
        self,
        tool_call_id: str = "",
        path: str = "",
        start_line: int = 1,
        end_line: int = -1,
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
        if not target.is_file():
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, f"不是文件: {path}")

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"无法读取（非文本文件）: {path}")
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"读取失败: {e}")

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply line range
        start_idx = max(0, start_line - 1)  # Convert to 0-based
        end_idx = total_lines if end_line == -1 else min(end_line, total_lines)

        if start_idx >= total_lines:
            return self._error_result(
                tool_call_id,
                ToolStatus.INVALID_ARGUMENT,
                f"起始行 {start_line} 超出文件总行数 {total_lines}",
            )

        selected = lines[start_idx:end_idx]

        # Truncate if too many lines
        truncated = False
        if len(selected) > self.MAX_LINES:
            selected = selected[:self.MAX_LINES]
            truncated = True

        # Format with line numbers
        numbered_lines = []
        for i, line in enumerate(selected, start=start_idx + 1):
            numbered_lines.append(f"{i:4d} | {line}")

        output = f"文件: {path} (第 {start_idx + 1}-{min(end_idx, start_idx + len(selected))} 行，共 {total_lines} 行)\n"
        output += "```\n"
        output += "\n".join(numbered_lines)
        output += "\n```"

        if truncated:
            output += f"\n\n[内容截断: 超过 {self.MAX_LINES} 行]"

        return self._success_result(
            tool_call_id,
            output,
            {"total_lines": total_lines, "lines_shown": len(numbered_lines)},
        )
