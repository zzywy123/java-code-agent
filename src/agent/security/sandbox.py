"""Sandbox for subprocess execution.

Provides timeout enforcement and output truncation.
Never uses shell=True - all commands are executed as argv lists.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of a sandboxed subprocess execution."""

    argv: list[str]
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    truncated: bool  # True if output was truncated


def run_sandboxed(
    argv: list[str],
    cwd: str | None = None,
    timeout: int = 120,
    max_output_chars: int = 50000,
    env: dict[str, str] | None = None,
) -> ExecutionResult:
    """Execute a command in a sandboxed subprocess.

    NEVER uses shell=True. Commands must be argv lists.
    Enforces timeout and output size limits.

    Args:
        argv: Command as [executable, arg1, arg2, ...]
        cwd: Working directory for the command
        timeout: Maximum execution time in seconds
        max_output_chars: Maximum output characters before truncation
        env: Environment variables (None = inherit from parent)

    Returns:
        ExecutionResult with stdout, stderr, return code, and metadata
    """
    start_time = time.monotonic()
    timed_out = False
    truncated = False

    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            # NEVER set shell=True
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        return_code = proc.returncode

    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
        stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
        # Add timeout notice to stderr
        stderr += f"\n[TIMEOUT] 命令在 {timeout} 秒后超时终止"
        return_code = -1

    except FileNotFoundError:
        stdout = ""
        stderr = f"[ERROR] 可执行文件未找到: {argv[0]}"
        return_code = -2

    except PermissionError:
        stdout = ""
        stderr = f"[ERROR] 权限不足: {argv[0]}"
        return_code = -3

    except Exception as e:
        stdout = ""
        stderr = f"[ERROR] 执行异常: {e}"
        return_code = -4

    duration = time.monotonic() - start_time

    # Truncate output if needed
    if len(stdout) > max_output_chars:
        stdout = stdout[:max_output_chars] + f"\n\n[输出截断: 超过 {max_output_chars} 字符]"
        truncated = True
    if len(stderr) > max_output_chars:
        stderr = stderr[:max_output_chars] + f"\n\n[输出截断: 超过 {max_output_chars} 字符]"
        truncated = True

    return ExecutionResult(
        argv=argv,
        return_code=return_code,
        stdout=stdout.strip(),
        stderr=stderr.strip(),
        timed_out=timed_out,
        duration_seconds=round(duration, 2),
        truncated=truncated,
    )
