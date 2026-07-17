"""Path traversal protection for the Java Coding Agent.

Uses relative_to for boundary checking instead of string prefix matching.
Handles symlinks by resolving and re-validating.
Detects and blocks .env files and other sensitive patterns.
"""

from __future__ import annotations

from pathlib import Path

# Directories excluded from agent access
EXCLUDED_DIRS: set[str] = {
    ".git",
    "node_modules",
    "target",
    "build",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".gradle",
}

# File extensions blocked from agent access
EXCLUDED_EXTENSIONS: set[str] = {
    ".class",
    ".jar",
    ".war",
    ".ear",
    ".key",
    ".pem",
    ".p12",
    ".jks",
    ".keystore",
}

# Sensitive file patterns (matched against filename)
SENSITIVE_PATTERNS: list[str] = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.staging",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "credentials",
    "secret",
    "password",
]


class PathViolationError(Exception):
    """Raised when a path operation violates security constraints."""

    def __init__(self, message: str, path: str = ""):
        super().__init__(message)
        self.path = path


def normalize_and_validate(path: str, repo_root: Path) -> Path:
    """Normalize and validate that a path is within the repository root.

    Uses relative_to for boundary checking instead of string prefix matching.
    Handles symlinks by resolving and re-validating.
    Detects .env files and other sensitive patterns.

    Args:
        path: The path to validate (relative to repo_root, or absolute within repo_root)
        repo_root: The repository root directory (must be resolved/absolute)

    Returns:
        Resolved absolute path within repo_root

    Raises:
        PathViolationError: If the path violates any security constraint
    """
    repo_resolved = repo_root.resolve()

    # Construct the target path
    if Path(path).is_absolute():
        target = Path(path)
    else:
        target = repo_resolved / path

    # Resolve to absolute path (but don't follow symlinks yet)
    try:
        target = target.resolve()
    except (OSError, ValueError) as e:
        raise PathViolationError(f"路径解析失败: {path} ({e})", path)

    # --- Boundary check using relative_to ---
    try:
        rel = target.relative_to(repo_resolved)
    except ValueError:
        raise PathViolationError(
            f"路径穿越: {path} 解析后为 {target}，不在仓库 {repo_resolved} 内",
            path,
        )

    # --- Symlink check ---
    # If the original (unresolved) path is a symlink, resolve its target
    # and verify the target is also within repo_root
    original = repo_resolved / path if not Path(path).is_absolute() else Path(path)
    if original.is_symlink():
        try:
            symlink_target = original.resolve()
            symlink_target.relative_to(repo_resolved)
        except ValueError:
            raise PathViolationError(
                f"符号链接逃逸: {path} 指向 {symlink_target}，超出仓库边界",
                path,
            )

    # --- Excluded directory check ---
    for part in rel.parts:
        if part in EXCLUDED_DIRS:
            raise PathViolationError(f"排除目录: {path} (包含 '{part}')", path)

    # --- Excluded extension check ---
    if target.suffix.lower() in EXCLUDED_EXTENSIONS:
        raise PathViolationError(f"排除文件类型: {path} ({target.suffix})", path)

    # --- Sensitive file pattern check ---
    filename = target.name.lower()
    for pattern in SENSITIVE_PATTERNS:
        if pattern in filename:
            raise PathViolationError(f"敏感文件: {path} (匹配 '{pattern}')", path)

    return target


def is_within_repo(path: Path, repo_root: Path) -> bool:
    """Check if a path is within the repository root without raising.

    Returns True if path is within repo_root, False otherwise.
    """
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False
