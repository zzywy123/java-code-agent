"""Secure repository imports for the Streamlit application."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

DEFAULT_ALLOWED_GIT_HOSTS = ("github.com", "gitee.com", "gitlab.com")
DEFAULT_MAX_FILES = 3000
DEFAULT_MAX_BYTES = 100 * 1024 * 1024
RAW_UPLOAD_FILE_MULTIPLIER = 20
IGNORED_UPLOAD_DIRS = {
    ".git",
    ".agent-index",
    ".checkpoints",
    ".gradle",
    ".idea",
    ".memory",
    ".mypy_cache",
    ".observability",
    ".pytest_cache",
    ".vscode",
    "__pycache__",
    "build",
    "node_modules",
    "out",
    "target",
}
SENSITIVE_UPLOAD_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
SENSITIVE_UPLOAD_SUFFIXES = {
    ".jks",
    ".key",
    ".keystore",
    ".p12",
    ".pem",
    ".pfx",
}


class RepositoryImportError(ValueError):
    """Raised when a repository source is invalid or cannot be imported."""


@dataclass(frozen=True)
class RepositoryImportResult:
    repo_root: Path
    source: str
    file_count: int
    total_bytes: int
    ignored_files: int = 0


class RepositoryImportService:
    """Import repositories into isolated server-side workspaces."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        allowed_server_roots: Iterable[Path | str] = (),
        allowed_git_hosts: Iterable[str] = DEFAULT_ALLOWED_GIT_HOSTS,
        max_files: int = DEFAULT_MAX_FILES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.allowed_server_roots = tuple(
            Path(root).expanduser().resolve() for root in allowed_server_roots
        )
        self.allowed_git_hosts = {
            host.strip().lower() for host in allowed_git_hosts if host.strip()
        }
        self.max_files = max_files
        self.max_bytes = max_bytes

    @classmethod
    def from_environment(cls) -> "RepositoryImportService":
        workspace_root = os.environ.get("AGENT_WORKSPACE_ROOT", ".uploaded-workspaces")
        raw_roots = os.environ.get(
            "AGENT_SERVER_PATH_ROOTS",
            os.environ.get("AGENT_REPO_ROOT", ""),
        )
        roots = [item for item in raw_roots.split(os.pathsep) if item.strip()]
        raw_hosts = os.environ.get(
            "AGENT_GIT_ALLOWED_HOSTS",
            ",".join(DEFAULT_ALLOWED_GIT_HOSTS),
        )
        hosts = [item for item in raw_hosts.split(",") if item.strip()]
        return cls(
            workspace_root,
            allowed_server_roots=roots,
            allowed_git_hosts=hosts,
            max_files=int(os.environ.get("AGENT_UPLOAD_MAX_FILES", DEFAULT_MAX_FILES)),
            max_bytes=int(os.environ.get("AGENT_UPLOAD_MAX_MB", 100)) * 1024 * 1024,
        )

    def import_uploaded_directory(
        self,
        files: Iterable[Any],
        *,
        owner_id: str,
        repository_name: str = "",
    ) -> RepositoryImportResult:
        """Persist one browser directory upload and create a clean Git baseline."""
        uploaded = list(files)
        if not uploaded:
            raise RepositoryImportError("请选择一个本地仓库目录")
        raw_file_limit = max(self.max_files, self.max_files * RAW_UPLOAD_FILE_MULTIPLIER)
        if len(uploaded) > raw_file_limit:
            raise RepositoryImportError(
                f"上传内容的原始文件数量超过安全限制：{len(uploaded)} > {raw_file_limit}"
            )

        normalized = [self._normalize_upload_path(str(item.name)) for item in uploaded]
        common_root = self._common_upload_root(normalized)
        import_items = [
            (item, self._strip_common_root(upload_path, common_root))
            for item, upload_path in zip(uploaded, normalized, strict=True)
        ]
        accepted_items = [
            (item, relative)
            for item, relative in import_items
            if not self._should_ignore_upload(relative)
        ]
        if len(accepted_items) > self.max_files:
            raise RepositoryImportError(
                f"有效文件数量超过限制：{len(accepted_items)} > {self.max_files}"
            )
        inferred_name = common_root or repository_name or "uploaded-repository"
        stage, destination = self._prepare_destination(owner_id, repository_name or inferred_name)
        ignored = len(import_items) - len(accepted_items)
        total_bytes = 0

        try:
            for item, relative in accepted_items:
                content = self._uploaded_bytes(item)
                total_bytes += len(content)
                if total_bytes > self.max_bytes:
                    raise RepositoryImportError(
                        f"仓库大小超过限制：最大 {self.max_bytes // (1024 * 1024)} MB"
                    )
                target = (stage / Path(*relative.parts)).resolve()
                try:
                    target.relative_to(stage)
                except ValueError as exc:
                    raise RepositoryImportError("上传路径超出工作区") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

            self._validate_java_repository(stage)
            self._initialize_git_baseline(stage)
            stage.replace(destination)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

        return RepositoryImportResult(
            repo_root=destination,
            source="upload",
            file_count=len(accepted_items),
            total_bytes=total_bytes,
            ignored_files=ignored,
        )

    def clone_repository(
        self,
        url: str,
        *,
        owner_id: str,
        branch: str = "",
    ) -> RepositoryImportResult:
        """Shallow-clone one public HTTPS repository from an allowed host."""
        normalized_url, repository_name = self._validate_git_url(url)
        branch = branch.strip()
        if branch and (
            not re.fullmatch(r"[A-Za-z0-9._/-]{1,200}", branch)
            or branch.startswith(("-", "/"))
            or ".." in branch
        ):
            raise RepositoryImportError("分支名称不合法")

        stage, destination = self._prepare_destination(owner_id, repository_name)
        shutil.rmtree(stage, ignore_errors=True)
        argv = [
            "git",
            "-c",
            "protocol.file.allow=never",
            "clone",
            "--depth",
            "1",
            "--single-branch",
        ]
        if branch:
            argv.extend(["--branch", branch])
        argv.extend([normalized_url, str(stage)])
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"

        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                env=environment,
            )
            if result.returncode != 0:
                reason = (result.stderr or result.stdout).strip()
                raise RepositoryImportError(f"Git 克隆失败：{reason[:1000]}")
            self._reject_worktree_symlinks(stage)
            self._validate_java_repository(stage)
            file_count, total_bytes = self._repository_size(stage)
            if file_count > self.max_files or total_bytes > self.max_bytes:
                raise RepositoryImportError("克隆后的仓库超过文件数量或容量限制")
            stage.replace(destination)
        except FileNotFoundError as exc:
            shutil.rmtree(stage, ignore_errors=True)
            raise RepositoryImportError("服务器未安装 git 命令") from exc
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(stage, ignore_errors=True)
            raise RepositoryImportError("Git 克隆超时") from exc
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

        return RepositoryImportResult(
            repo_root=destination,
            source="git",
            file_count=file_count,
            total_bytes=total_bytes,
        )

    def validate_server_repository(self, path: str) -> RepositoryImportResult:
        """Validate a repository path that already exists on the server."""
        if not path.strip():
            raise RepositoryImportError("请输入服务器仓库路径")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise RepositoryImportError("仓库目录不存在")
        if self.allowed_server_roots and not any(
            self._is_within(resolved, root) for root in self.allowed_server_roots
        ):
            raise RepositoryImportError("该路径不在服务器允许的工作区内")
        self._reject_worktree_symlinks(resolved)
        self._validate_java_repository(resolved)
        file_count, total_bytes = self._repository_size(resolved)
        return RepositoryImportResult(
            repo_root=resolved,
            source="server",
            file_count=file_count,
            total_bytes=total_bytes,
        )

    def _prepare_destination(self, owner_id: str, name: str) -> tuple[Path, Path]:
        owner = self._safe_name(owner_id, fallback="anonymous")
        repository = self._safe_name(name, fallback="repository")
        owner_root = (self.workspace_root / owner).resolve()
        owner_root.mkdir(parents=True, exist_ok=True)
        owner_root.relative_to(self.workspace_root)
        suffix = uuid.uuid4().hex[:10]
        stage = owner_root / f".import-{suffix}"
        destination = owner_root / f"{repository}-{suffix}"
        stage.mkdir()
        return stage, destination

    def _validate_git_url(self, url: str) -> tuple[str, str]:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not host:
            raise RepositoryImportError("Git 地址必须使用 HTTPS")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RepositoryImportError("Git 地址不能包含凭据、查询参数或片段")
        try:
            port = parsed.port
        except ValueError as exc:
            raise RepositoryImportError("Git 地址端口不合法") from exc
        if port not in {None, 443}:
            raise RepositoryImportError("Git 地址只能使用 HTTPS 默认端口")
        if not any(
            host == allowed or host.endswith(f".{allowed}")
            for allowed in self.allowed_git_hosts
        ):
            raise RepositoryImportError("Git 地址不在允许的代码托管平台列表中")
        repository = PurePosixPath(parsed.path).name.removesuffix(".git")
        if not repository or repository in {".", "/"}:
            raise RepositoryImportError("Git 地址缺少仓库名称")
        return parsed.geturl(), repository

    def _initialize_git_baseline(self, repository: Path) -> None:
        commands = (
            ["git", "init", "-q"],
            ["git", "add", "--all"],
            [
                "git",
                "-c",
                "user.name=Java Coding Agent",
                "-c",
                "user.email=agent@localhost.invalid",
                "commit",
                "-qm",
                "导入仓库基线",
            ],
        )
        for argv in commands:
            try:
                result = subprocess.run(
                    argv,
                    cwd=repository,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
            except FileNotFoundError as exc:
                raise RepositoryImportError("服务器未安装 git 命令") from exc
            except subprocess.TimeoutExpired as exc:
                raise RepositoryImportError("初始化 Git 仓库超时") from exc
            if result.returncode != 0:
                reason = (result.stderr or result.stdout).strip()
                raise RepositoryImportError(f"初始化 Git 仓库失败：{reason[:1000]}")

    def _repository_size(self, repository: Path) -> tuple[int, int]:
        file_count = 0
        total_bytes = 0
        for candidate in repository.rglob("*"):
            if not candidate.is_file() or ".git" in candidate.relative_to(repository).parts:
                continue
            file_count += 1
            try:
                total_bytes += candidate.stat().st_size
            except OSError:
                continue
        return file_count, total_bytes

    @staticmethod
    def _reject_worktree_symlinks(repository: Path) -> None:
        for candidate in repository.rglob("*"):
            relative = candidate.relative_to(repository)
            if candidate.is_symlink():
                raise RepositoryImportError(f"仓库包含不支持的符号链接：{relative}")

    @staticmethod
    def _validate_java_repository(repository: Path) -> None:
        if not any(repository.rglob("*.java")):
            raise RepositoryImportError("目录中没有 Java 源文件")

    @staticmethod
    def _normalize_upload_path(name: str) -> PurePosixPath:
        normalized = PurePosixPath(name.replace("\\", "/"))
        if normalized.is_absolute() or not normalized.parts:
            raise RepositoryImportError("上传文件路径不合法")
        if any(part in {"", ".", ".."} for part in normalized.parts):
            raise RepositoryImportError("上传文件路径包含目录穿越")
        return normalized

    @staticmethod
    def _common_upload_root(paths: list[PurePosixPath]) -> str:
        first_parts = {path.parts[0] for path in paths if len(path.parts) > 1}
        return next(iter(first_parts)) if len(first_parts) == 1 and all(
            len(path.parts) > 1 for path in paths
        ) else ""

    @staticmethod
    def _strip_common_root(path: PurePosixPath, common_root: str) -> PurePosixPath:
        if common_root and path.parts[0] == common_root:
            return PurePosixPath(*path.parts[1:])
        return path

    @staticmethod
    def _should_ignore_upload(path: PurePosixPath) -> bool:
        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts & IGNORED_UPLOAD_DIRS:
            return True
        filename = path.name.lower()
        if filename in SENSITIVE_UPLOAD_NAMES or filename.startswith(".env."):
            return True
        return PurePosixPath(filename).suffix in SENSITIVE_UPLOAD_SUFFIXES

    @staticmethod
    def _uploaded_bytes(uploaded_file: Any) -> bytes:
        if hasattr(uploaded_file, "getvalue"):
            value = uploaded_file.getvalue()
        else:
            value = uploaded_file.read()
        if not isinstance(value, bytes):
            raise RepositoryImportError("上传文件内容不是二进制数据")
        return value

    @staticmethod
    def _safe_name(value: str, *, fallback: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
        return (normalized or fallback)[:80]

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
