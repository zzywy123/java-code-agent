"""Build and test tools: run_tests.

Uses structured parameters (tool, goals, module) instead of arbitrary commands.
Validates all arguments through command_guard before execution.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
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


def detect_build_tool(repo_root: Path) -> str | None:
    """Detect the repository build tool from its standard build files."""
    if (repo_root / "pom.xml").is_file():
        return "maven"
    if any((repo_root / name).is_file() for name in ("build.gradle", "build.gradle.kts")):
        return "gradle"
    return None


def _resolve_build_executable(repo_root: Path, tool: str) -> str | None:
    if tool == "maven":
        wrapper = repo_root / ("mvnw.cmd" if os.name == "nt" else "mvnw")
        if wrapper.is_file():
            return str(wrapper.resolve())
        executable = shutil.which("mvn")
        if executable:
            return executable
        if os.name == "nt":
            for candidate in (
                Path(r"E:\maven\apache-maven-3.5.01\bin\mvn.cmd"),
                Path(r"C:\Program Files\Maven\bin\mvn.cmd"),
            ):
                if candidate.is_file():
                    return str(candidate)
        return None

    wrapper = repo_root / ("gradlew.bat" if os.name == "nt" else "gradlew")
    if wrapper.is_file():
        return str(wrapper.resolve())
    return shutil.which("gradle")


def _is_jdk_home(path: str | Path) -> bool:
    home = Path(path)
    javac_name = "javac.exe" if os.name == "nt" else "javac"
    return (home / "bin" / javac_name).is_file()


def _discover_java_home(environment: dict[str, str]) -> str | None:
    """Find a real JDK, preferring explicit configuration and valid JAVA_HOME."""
    for variable in ("AGENT_JAVA_HOME", "JAVA_HOME"):
        candidate = environment.get(variable, "").strip()
        if candidate and _is_jdk_home(candidate):
            return str(Path(candidate).resolve())

    javac = shutil.which("javac")
    if not javac:
        return None

    try:
        result = subprocess.run(
            [javac, "-J-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        settings = f"{result.stdout}\n{result.stderr}"
        matches = re.findall(
            r"^\s*(?:application\.home|java\.home)\s*=\s*(.+?)\s*$",
            settings,
            flags=re.MULTILINE,
        )
        for candidate in matches:
            if _is_jdk_home(candidate):
                return str(Path(candidate).resolve())
    except (OSError, subprocess.TimeoutExpired):
        pass

    inferred = Path(javac).resolve().parent.parent
    return str(inferred) if _is_jdk_home(inferred) else None


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

        # Build through the goal/task allowlist before resolving executables.
        try:
            if tool == "maven":
                executable = _resolve_build_executable(self.repo_root, tool)
                use_wrapper = bool(executable and Path(executable).resolve().parent == self.repo_root.resolve())
                argv = build_maven_argv(
                    goals,
                    module=module,
                    extra_args=extra_args,
                    use_wrapper=use_wrapper,
                )
                if not executable:
                    return self._error_result(
                        tool_call_id,
                        ToolStatus.NOT_FOUND,
                        "找不到 mvn 命令，请确保 Maven 已安装并在 PATH 中",
                    )
                argv[0] = executable
            elif tool == "gradle":
                executable = _resolve_build_executable(self.repo_root, tool)
                use_wrapper = bool(executable and Path(executable).resolve().parent == self.repo_root.resolve())
                argv = build_gradle_argv(
                    goals,
                    project_path=module,
                    extra_args=extra_args,
                    use_wrapper=use_wrapper,
                )
                if not executable:
                    return self._error_result(
                        tool_call_id,
                        ToolStatus.NOT_FOUND,
                        "找不到 gradle 命令，请确保 Gradle 已安装并在 PATH 中",
                    )
                argv[0] = executable
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
        java_home = _discover_java_home(child_env)
        child_env.pop("AGENT_JAVA_HOME", None)
        if java_home:
            child_env["JAVA_HOME"] = java_home
            child_env["PATH"] = str(Path(java_home) / "bin") + os.pathsep + child_env.get("PATH", "")
        elif os.name == "nt":
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
        if java_home:
            output_parts.append(f"JDK: {java_home}")
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
                "java_home": java_home,
            },
        )
