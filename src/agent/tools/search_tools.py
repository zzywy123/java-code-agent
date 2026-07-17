"""Code search with per-file path protection."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agent.models import ToolResult, ToolStatus
from agent.security.path_guard import EXCLUDED_DIRS, SENSITIVE_PATTERNS
from agent.tools.base import BaseTool


class SearchCodeTool(BaseTool):
    """Search code using ripgrep, with a protected Python fallback."""

    name = "search_code"
    description = (
        "在代码仓库中搜索文本或正则表达式。"
        "返回通过安全路径校验的文件路径、行号和内容。"
        "用于查找方法定义、类引用、错误信息等。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词或正则表达式"},
            "path": {
                "type": "string",
                "description": "搜索范围（相对于仓库根目录），默认为 '.'",
                "default": ".",
            },
            "file_pattern": {
                "type": "string",
                "description": "文件名 glob 模式，如 '*.java'",
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
        except Exception as exc:
            return self._error_result(tool_call_id, ToolStatus.DENIED, str(exc))
        if not target.exists():
            return self._error_result(tool_call_id, ToolStatus.NOT_FOUND, f"路径不存在: {path}")

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
        relative_target = target.relative_to(self.repo_root.resolve()).as_posix() or "."
        argv = [
            rg_path,
            "--json",
            "--glob-case-insensitive",
            "--max-count=10",
            "--glob",
            file_pattern,
        ]
        for directory in sorted(EXCLUDED_DIRS):
            argv.extend(["--glob", f"!**/{directory}/**"])
        for pattern in SENSITIVE_PATTERNS:
            argv.extend(["--glob", f"!*{pattern}*"])
        argv.extend([query, relative_target])

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                cwd=str(self.repo_root),
            )
        except subprocess.TimeoutExpired:
            return self._error_result(tool_call_id, ToolStatus.TIMEOUT, "搜索超时（30秒）")
        except Exception as exc:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"ripgrep 执行失败: {exc}")

        if proc.returncode not in {0, 1}:
            reason = (proc.stderr or "ripgrep 执行失败").strip()
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, reason)

        formatted: list[str] = []
        truncated = False
        for raw_line in proc.stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data", {})
            path_text = str(data.get("path", {}).get("text", ""))
            try:
                matched_path = self.validate_path(path_text)
            except Exception:
                continue
            rel_path = matched_path.relative_to(self.repo_root.resolve()).as_posix()
            line_number = int(data.get("line_number") or 0)
            content = str(data.get("lines", {}).get("text", "")).rstrip("\r\n")
            formatted.append(f"  {rel_path}:{line_number}: {content}")
            if len(formatted) >= self.MAX_RESULTS:
                truncated = True
                break

        return self._format_result(tool_call_id, query, formatted, truncated)

    def _search_with_python(
        self,
        tool_call_id: str,
        query: str,
        target: Path,
        file_pattern: str,
    ) -> ToolResult:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, f"无效的正则表达式: {exc}")

        try:
            files = target.rglob(file_pattern) if target.is_dir() else [target]
            matches: list[str] = []
            for candidate in files:
                if not candidate.is_file():
                    continue
                try:
                    filepath = self.validate_path(str(candidate))
                except Exception:
                    continue
                try:
                    content = filepath.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    continue
                for line_number, line in enumerate(content.splitlines(), 1):
                    if pattern.search(line):
                        rel_path = filepath.relative_to(self.repo_root.resolve()).as_posix()
                        matches.append(f"  {rel_path}:{line_number}: {line.strip()}")
                        if len(matches) >= self.MAX_RESULTS:
                            return self._format_result(tool_call_id, query, matches, True)
        except Exception as exc:
            return self._error_result(tool_call_id, ToolStatus.EXECUTION_ERROR, f"文件遍历失败: {exc}")
        return self._format_result(tool_call_id, query, matches, False)

    def _format_result(
        self,
        tool_call_id: str,
        query: str,
        matches: list[str],
        truncated: bool,
    ) -> ToolResult:
        if not matches:
            return self._success_result(tool_call_id, f"未找到匹配: '{query}'", {"match_count": 0})
        output = f"搜索 '{query}' 找到 {len(matches)} 个匹配:\n" + "\n".join(matches)
        if truncated:
            output += f"\n\n[结果截断: 显示前 {self.MAX_RESULTS} 条]"
        return self._success_result(
            tool_call_id,
            output,
            {"match_count": len(matches), "truncated": truncated},
        )
