"""Repository import boundary and security tests."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent.repository_import import RepositoryImportError, RepositoryImportService


@dataclass
class UploadedFile:
    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


def java_uploads() -> list[UploadedFile]:
    return [
        UploadedFile("sample/src/main/java/App.java", b"class App {}"),
        UploadedFile("sample/README.md", b"sample"),
    ]


def test_upload_rebuilds_structure_and_creates_clean_git_baseline(tmp_path: Path):
    service = RepositoryImportService(tmp_path / "workspaces")

    result = service.import_uploaded_directory(java_uploads(), owner_id="browser-1")

    assert result.file_count == 2
    assert (result.repo_root / "src/main/java/App.java").read_bytes() == b"class App {}"
    assert (result.repo_root / ".git").is_dir()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=result.repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ""


def test_upload_filters_metadata_build_outputs_and_exact_sensitive_files(tmp_path: Path):
    service = RepositoryImportService(tmp_path / "workspaces")
    uploads = java_uploads() + [
        UploadedFile("sample/.git/config", b"ignored"),
        UploadedFile("sample/.env.production", b"TOKEN=ignored"),
        UploadedFile("sample/target/App.class", b"ignored"),
        UploadedFile("sample/src/PasswordService.java", b"class PasswordService {}"),
        UploadedFile("sample/src/SecretService.java", b"class SecretService {}"),
    ]

    result = service.import_uploaded_directory(uploads, owner_id="browser-1")

    assert result.ignored_files == 3
    assert not (result.repo_root / ".env.production").exists()
    assert not (result.repo_root / "target").exists()
    assert (result.repo_root / "src/PasswordService.java").is_file()
    assert (result.repo_root / "src/SecretService.java").is_file()


def test_upload_rejects_path_traversal(tmp_path: Path):
    service = RepositoryImportService(tmp_path / "workspaces")
    with pytest.raises(RepositoryImportError, match="目录穿越"):
        service.import_uploaded_directory(
            [UploadedFile("sample/../App.java", b"class App {}")],
            owner_id="browser-1",
        )


def test_upload_limits_effective_file_count_and_total_bytes(tmp_path: Path):
    file_limited = RepositoryImportService(tmp_path / "files", max_files=1)
    with pytest.raises(RepositoryImportError, match="有效文件数量"):
        file_limited.import_uploaded_directory(java_uploads(), owner_id="browser-1")

    byte_limited = RepositoryImportService(tmp_path / "bytes", max_bytes=5)
    with pytest.raises(RepositoryImportError, match="仓库大小"):
        byte_limited.import_uploaded_directory(
            [UploadedFile("sample/App.java", b"class App {}")],
            owner_id="browser-1",
        )


@pytest.mark.parametrize(
    "url,message",
    [
        ("http://github.com/org/repo.git", "HTTPS"),
        ("https://localhost/org/repo.git", "允许"),
        ("https://github.com:invalid/org/repo.git", "端口"),
        ("https://token@github.com/org/repo.git", "凭据"),
    ],
)
def test_git_url_validation_rejects_unsafe_sources(
    tmp_path: Path, url: str, message: str
):
    service = RepositoryImportService(tmp_path / "workspaces")
    with pytest.raises(RepositoryImportError, match=message):
        service._validate_git_url(url)


def test_server_path_must_be_under_allowed_root(tmp_path: Path):
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    (allowed / "App.java").write_text("class App {}", encoding="utf-8")
    (denied / "App.java").write_text("class App {}", encoding="utf-8")
    service = RepositoryImportService(
        tmp_path / "workspaces", allowed_server_roots=[allowed]
    )

    assert service.validate_server_repository(str(allowed)).source == "server"
    with pytest.raises(RepositoryImportError, match="允许"):
        service.validate_server_repository(str(denied))


def test_repository_symlinks_are_rejected(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    target = tmp_path / "outside.java"
    target.write_text("class Outside {}", encoding="utf-8")
    link = repository / "App.java"
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"当前环境不能创建符号链接: {exc}")

    service = RepositoryImportService(tmp_path / "workspaces")
    with pytest.raises(RepositoryImportError, match="符号链接"):
        service._reject_worktree_symlinks(repository)
