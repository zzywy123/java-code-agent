"""Command execution security using argv-level validation.

No shell=True is ever used. Commands are validated as argv lists.
Only specific executables and argument patterns are allowed.
"""

from __future__ import annotations

import re

# Allowed executables (matched against basename of argv[0])
ALLOWED_EXECUTABLES: set[str] = {
    "mvn",
    "mvnw",
    "mvnw.cmd",
    "gradle",
    "gradlew",
    "gradlew.bat",
    "java",
    "javac",
    "git",
}

# Blocked patterns checked against the full command string
BLOCKED_PATTERNS: list[str] = [
    r";",
    r"&&",
    r"\|\|",
    r"\|",
    r"`",
    r"\$\(",
    r">>",
    r">(?![&])",  # redirect (but not 2>&1)
    r"<",
    r"rm\s",
    r"chmod",
    r"chown",
    r"curl",
    r"wget",
    r"sudo",
    r"su\s",
    r"eval",
    r"exec",
    r"nc\s",
    r"bash\s+-[ci]",
    r"sh\s+-[ci]",
    r"powershell",
    r"cmd\s*/[cC]",
]

# Allowed Maven goals
ALLOWED_MAVEN_GOALS: set[str] = {
    "clean",
    "compile",
    "test",
    "install",
    "package",
    "verify",
    "validate",
    "dependency:tree",
    "dependency:resolve",
}

# Allowed Gradle tasks
ALLOWED_GRADLE_TASKS: set[str] = {
    "clean",
    "build",
    "test",
    "compileJava",
    "compileTestJava",
    "dependencies",
    "check",
}

# Allowed extra argument prefixes
ALLOWED_ARG_PREFIXES: list[str] = [
    "-D",  # Maven/Gradle properties
    "-P",  # Maven profiles
    "-pl",  # Maven module selection
    "-am",  # Maven also-make
    "--fail-at-end",
    "-T",  # Maven threads
    "-p",  # Gradle project
    "--tests",  # Gradle test filter
    "-x",  # Gradle exclude task
    "--no-daemon",
    "--console=plain",
]


class CommandViolationError(Exception):
    """Raised when a command violates security constraints."""

    def __init__(self, message: str, command: list[str] | None = None):
        super().__init__(message)
        self.command = command or []


def validate_argv(argv: list[str]) -> tuple[bool, str]:
    """Validate that an argv list is safe to execute.

    Args:
        argv: Command as a list of arguments [executable, arg1, arg2, ...]

    Returns:
        (allowed, reason) tuple. If allowed is False, reason explains why.
    """
    if not argv:
        return False, "空命令"

    exe = argv[0]

    # Extract basename for matching
    exe_basename = exe.replace("\\", "/").split("/")[-1]
    if exe_basename.lower().endswith(".exe"):
        exe_basename = exe_basename[:-4]

    # Check executable is allowed (case-insensitive on Windows)
    import os
    if os.name == "nt":
        allowed = {e.lower() for e in ALLOWED_EXECUTABLES}
        if exe_basename.lower() not in allowed:
            return False, f"可执行文件 '{exe_basename}' 不在允许列表中"
    else:
        if exe_basename not in ALLOWED_EXECUTABLES:
            return False, f"可执行文件 '{exe_basename}' 不在允许列表中"

    # Check for blocked patterns in the full command string
    full_cmd = " ".join(argv)
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, full_cmd, re.IGNORECASE):
            return False, f"命令包含被阻止的模式: {pattern}"

    # Validate args don't contain shell metacharacters
    for i, arg in enumerate(argv[1:], 1):
        if any(c in arg for c in (";", "|", "&", "`", "$", "(", ")")):
            return False, f"参数 {i} 包含 shell 元字符: {arg}"

    return True, "ok"


def build_maven_argv(
    goals: list[str],
    module: str = "",
    extra_args: list[str] | None = None,
    use_wrapper: bool = True,
) -> list[str]:
    """Build a validated Maven command argv.

    Args:
        goals: Maven goals (e.g., ["clean", "test"])
        module: Optional module selection (-pl flag)
        extra_args: Optional extra arguments (validated against allowlist)
        use_wrapper: Whether to use mvnw instead of mvn

    Returns:
        Validated argv list

    Raises:
        CommandViolationError: If any goal or argument is not allowed
    """
    # Validate goals
    for goal in goals:
        if goal not in ALLOWED_MAVEN_GOALS:
            raise CommandViolationError(
                f"Maven goal '{goal}' 不在允许列表中. 允许: {ALLOWED_MAVEN_GOALS}",
                goals,
            )

    import os
    if use_wrapper:
        # Use mvnw.cmd on Windows, ./mvnw on Unix
        exe = "mvnw.cmd" if os.name == "nt" else "./mvnw"
    else:
        exe = "mvn"
    argv = [exe, "-B"] + goals  # -B for batch mode (non-interactive)

    if module:
        argv.extend(["-pl", module])

    if extra_args:
        for arg in extra_args:
            if not any(arg.startswith(p) for p in ALLOWED_ARG_PREFIXES):
                raise CommandViolationError(
                    f"额外参数 '{arg}' 不在允许前缀列表中",
                    extra_args,
                )
            argv.append(arg)

    return argv


def build_gradle_argv(
    tasks: list[str],
    project_path: str = "",
    extra_args: list[str] | None = None,
    use_wrapper: bool = True,
) -> list[str]:
    """Build a validated Gradle command argv.

    Args:
        tasks: Gradle tasks (e.g., ["clean", "test"])
        project_path: Optional project path (-p flag)
        extra_args: Optional extra arguments
        use_wrapper: Whether to use gradlew instead of gradle

    Returns:
        Validated argv list

    Raises:
        CommandViolationError: If any task or argument is not allowed
    """
    for task in tasks:
        if task not in ALLOWED_GRADLE_TASKS:
            raise CommandViolationError(
                f"Gradle task '{task}' 不在允许列表中. 允许: {ALLOWED_GRADLE_TASKS}",
                tasks,
            )

    exe = "./gradlew" if use_wrapper else "gradle"
    argv = [exe, "--no-daemon", "--console=plain"] + tasks

    if project_path:
        argv.extend(["-p", project_path])

    if extra_args:
        for arg in extra_args:
            if not any(arg.startswith(p) for p in ALLOWED_ARG_PREFIXES):
                raise CommandViolationError(
                    f"额外参数 '{arg}' 不在允许前缀列表中",
                    extra_args,
                )
            argv.append(arg)

    return argv
