"""Build and test tools: run_tests.

Uses structured parameters (tool, goals, module) instead of arbitrary commands.
Validates all arguments through command_guard before execution.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from agent.models import ToolResult, ToolStatus
from agent.security.command_guard import (
    CommandViolationError,
    build_gradle_argv,
    build_maven_argv,
    validate_argv,
)
from agent.security.sandbox import run_sandboxed
from agent.tools.base import BaseTool


class RunTestsTool(BaseTool):
    """Run Maven or Gradle build/test commands.

    Only accepts structured parameters, never arbitrary command strings.
    All arguments are validated through command_guard.
    """

    name = "run_tests"
    description = (
        "执行 Maven 或 Gradle 构建/测试命令。"
        "只接受结构化参数（tool、goals、module），不接受任意命令。"
        "支持 maven 和 gradle 两种构建工具。"
        "返回构建输出、测试结果和退出码。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": ["maven", "gradle"],
                "description": "构建工具类型",
            },
            "goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "构建目标，如 ['test'] 或 ['clean', 'test']",
            },
            "module": {
                "type": "string",
                "description": "Maven 模块名（-pl 参数），默认为空",
                "default": "",
            },
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "额外参数，如 ['-DskipTests=true']",
                "default": [],
            },
        },
        "required": ["tool", "goals"],
    }

    def execute(
        self,
        tool_call_id: str = "",
        tool: str = "",
        goals: list[str] | None = None,
        module: str = "",
        extra_args: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not tool:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, "必须指定构建工具 (maven/gradle)")
        if not goals:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, "必须指定构建目标 (goals)")

        # Build argv through command_guard
        try:
            if tool == "maven":
                wrapper = self.repo_root / ("mvnw.cmd" if os.name == "nt" else "mvnw")
                import shutil
                mvn_path = str(wrapper) if wrapper.exists() else shutil.which("mvn")
                if not mvn_path:
                    # Try common Windows location
                    for p in [r"E:\maven\apache-maven-3.5.01\bin\mvn.cmd",
                              r"C:\Program Files\Maven\bin\mvn.cmd"]:
                        if Path(p).exists():
                            mvn_path = p
                            break
                if not mvn_path:
                    return self._error_result(
                        tool_call_id,
                        ToolStatus.NOT_FOUND,
                        "找不到 mvn 命令，请确保 Maven 已安装并在 PATH 中",
                    )
                argv = [mvn_path, "-B"] + goals
                if module:
                    argv.extend(["-pl", module])
                if extra_args:
                    argv.extend(extra_args)
            elif tool == "gradle":
                import shutil
                wrapper = self.repo_root / ("gradlew.bat" if os.name == "nt" else "gradlew")
                gradle_path = str(wrapper) if wrapper.exists() else shutil.which("gradle")
                if not gradle_path:
                    return self._error_result(
                        tool_call_id,
                        ToolStatus.NOT_FOUND,
                        "找不到 gradle 命令，请确保 Gradle 已安装并在 PATH 中",
                    )
                argv = [gradle_path, "--no-daemon", "--console=plain"] + goals
                if module:
                    argv.extend(["-p", module])
                if extra_args:
                    argv.extend(extra_args)
            else:
                return self._error_result(
                    tool_call_id,
                    ToolStatus.INVALID_ARGUMENT,
                    f"不支持的构建工具: {tool}. 只支持 maven 和 gradle",
                )
        except CommandViolationError as e:
            return self._error_result(tool_call_id, ToolStatus.INVALID_ARGUMENT, str(e))

        # Final validation
        allowed, reason = validate_argv(argv)
        if not allowed:
            return self._error_result(tool_call_id, ToolStatus.DENIED, f"命令验证失败: {reason}")

        # Execute in sandbox
        child_env = os.environ.copy()
        configured_java_home = child_env.pop("AGENT_JAVA_HOME", "")
        if configured_java_home:
            child_env["JAVA_HOME"] = configured_java_home
        elif os.name == "nt":
            # The host may expose a stale JAVA_HOME while java.exe on PATH is current.
            child_env.pop("JAVA_HOME", None)
        if tool == "maven":
            encoding_opts = (
                "-Dfile.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8 "
                "-Dsun.stderr.encoding=UTF-8"
            )
            child_env["MAVEN_OPTS"] = (
                f"{child_env.get('MAVEN_OPTS', '')} {encoding_opts}"
            ).strip()

        result = run_sandboxed(
            argv=argv,
            cwd=str(self.repo_root),
            timeout=120,  # Default timeout
            max_output_chars=50000,
            env=child_env,
        )

        # Build output
        output_parts = []
        output_parts.append(f"命令: {' '.join(argv)}")
        output_parts.append(f"退出码: {result.return_code}")
        output_parts.append(f"耗时: {result.duration_seconds}秒")

        if result.timed_out:
            output_parts.append("[超时]")

        if result.stdout:
            output_parts.append(f"\n--- STDOUT ---\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"\n--- STDERR ---\n{result.stderr}")

        output = "\n".join(output_parts)

        # Determine status
        if result.timed_out:
            status = ToolStatus.TIMEOUT
        elif result.return_code == 0:
            status = ToolStatus.SUCCESS
        else:
            status = ToolStatus.EXECUTION_ERROR

        return ToolResult(
            tool_call_id=tool_call_id,
            name=self.name,
            status=status,
            output=output,
            metadata={
                "exit_code": result.return_code,
                "timed_out": result.timed_out,
                "duration": result.duration_seconds,
                "truncated": result.truncated,
            },
        )
