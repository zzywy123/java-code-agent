"""Code search tool using ripgrep.

Falls back to Python-based search if ripgrep is not available.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent.models import ToolResult, ToolStatus
from agent.tools.base import BaseTool


class SearchCodeTool(BaseTool):
    """Search code using ripgrep (or Python fallback)."""

    name = "search_code"
    description = (
        "在代码仓库中搜索文本或正则表达式。"
        "返回匹配的文件路径、行号和内容。"
        "用于查找方法定义、类引用、错误信息等。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或正则表达式",
            },
            "path": {
                "type": "string",
                "description": "搜索范围（相对于仓库根目录的路径），默认为 '.'",
                "default": ".",
            },
            "file_pattern": {
                "type": "string",
                "description": "文件名 glob 模式，如 '*.java'，默认搜索所有文本文件",
                "default": "*",
            },
        },
        "required": ["query"],
    }

    MAX_RESULTS = 50

    def execute(
        self,
        tool_call_id: str = "",
        query: str = "",
        path: str = ".",
        file_pattern: str = "*",
        **kwargs: Any,
    ) -> ToolResult:
        if not query:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, "必须指定搜索关键词")

        try:
            target = self.validate_path(path)
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.DENIED, str(e))

        if not target.exists():
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, f"路径不存在: {path}")

        # Try ripgrep first
        rg_path = shutil.which("rg")
        if rg_path:
            return self._search_with_rg(tool_call_id, query, target, file_pattern, rg_path)
        return self._search_with_python(tool_call_id, query, target, file_pattern)

    def _search_with_rg(
        self,
        tool_call_id: str,
        query: str,
        target: Path,
        file_pattern: str,
        rg_path: str,
    ) -> ToolResult:
        """Search using ripgrep."""
        argv = [
            rg_path,
            "--no-heading",
            "--line-number",
            "--color=never",
            "--max-count=10",  # Max matches per file
            "--glob", file_pattern,
            "--type-add", "java:*.java",
            "--type-add", "xml:*.xml",
            query,
            str(target),
        ]

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.repo_root),
            )
        except subprocess.TimeoutExpired:
            return self._error_result(tool_call_id, ToolStatus.TIMEOUT, "搜索超时（30秒）")
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"ripgrep 执行失败: {e}")

        output = proc.stdout.strip()
        if not output:
            return self._success_result(tool_call_id, f"未找到匹配: '{query}'", {"match_count": 0})

        # Truncate results
        lines = output.splitlines()
        truncated = len(lines) > self.MAX_RESULTS
        if truncated:
            lines = lines[:self.MAX_RESULTS]

        # Format results with relative paths
        formatted = []
        for line in lines:
            # rg output format: file:line:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                try:
                    rel_path = Path(parts[0]).relative_to(self.repo_root).as_posix()
                    formatted.append(f"  {rel_path}:{parts[1]}: {parts[2]}")
                except ValueError:
                    formatted.append(f"  {line}")
            else:
                formatted.append(f"  {line}")

        result = f"搜索 '{query}' 找到 {len(formatted)} 个匹配:\n" + "\n".join(formatted)
        if truncated:
            result += f"\n\n[结果截断: 共 {len(lines)} 条，显示前 {self.MAX_RESULTS} 条]"

        return self._success_result(
            tool_call_id,
            result,
            {"match_count": len(formatted), "truncated": truncated},
        )

    def _search_with_python(
        self,
        tool_call_id: str,
        query: str,
        target: Path,
        file_pattern: str,
    ) -> ToolResult:
        """Fallback: search using Python regex."""
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as e:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, f"无效的正则表达式: {e}")

        matches = []
        try:
            files = list(target.rglob(file_pattern))
        except Exception as e:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"文件遍历失败: {e}")

        for filepath in files:
            if not filepath.is_file():
                continue
            # Skip binary files
            if filepath.suffix in {".class", ".jar", ".war", ".ear", ".png", ".jpg", ".gif"}:
                continue
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for i, line in enumerate(content.splitlines(), 1):
                if pattern.search(line):
                    try:
                        rel_path = filepath.relative_to(self.repo_root).as_posix()
                    except ValueError:
                        rel_path = str(filepath)
                    matches.append(f"  {rel_path}:{i}: {line.strip()}")
                    if len(matches) >= self.MAX_RESULTS:
                        break
            if len(matches) >= self.MAX_RESULTS:
                break

        if not matches:
            return self._success_result(tool_call_id, f"未找到匹配: '{query}'", {"match_count": 0})

        result = f"搜索 '{query}' 找到 {len(matches)} 个匹配:\n" + "\n".join(matches)
        return self._success_result(
            tool_call_id,
            result,
            {"match_count": len(matches)},
        )
