"""Tests for path_guard module - path traversal protection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.security.path_guard import (
    PathViolationError,
    is_within_repo,
    normalize_and_validate,
)


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Create a sample repo structure."""
    src = tmp_path / "src" / "main" / "java" / "com" / "example"
    src.mkdir(parents=True)
    (src / "Main.java").write_text("public class Main {}", encoding="utf-8")

    # Create a .git directory (should be blocked)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git config", encoding="utf-8")

    # Create a .env file (should be blocked)
    (tmp_path / ".env").write_text("SECRET=abc123", encoding="utf-8")

    # Create a target directory (should be blocked)
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "Main.class").write_bytes(b"\xca\xfe\xba\xbe")

    return tmp_path


class TestNormalizeAndValidate:
    """Tests for normalize_and_validate function."""

    def test_valid_relative_path(self, repo_root: Path):
        result = normalize_and_validate("src/main/java/com/example/Main.java", repo_root)
        assert result.exists()
        assert result.name == "Main.java"

    def test_valid_absolute_path_within_repo(self, repo_root: Path):
        abs_path = str(repo_root / "src" / "main" / "java" / "com" / "example" / "Main.java")
        result = normalize_and_validate(abs_path, repo_root)
        assert result.exists()

    def test_path_traversal_rejected(self, repo_root: Path):
        with pytest.raises(PathViolationError, match="路径穿越"):
            normalize_and_validate("../../etc/passwd", repo_root)

    def test_deep_traversal_rejected(self, repo_root: Path):
        with pytest.raises(PathViolationError, match="路径穿越"):
            normalize_and_validate("src/../../../../../../etc/passwd", repo_root)

    def test_dot_git_blocked(self, repo_root: Path):
        with pytest.raises(PathViolationError, match="排除目录"):
            normalize_and_validate(".git/config", repo_root)

    def test_target_dir_blocked(self, repo_root: Path):
        with pytest.raises(PathViolationError, match="排除目录"):
            normalize_and_validate("target/Main.class", repo_root)

    def test_env_file_blocked(self, repo_root: Path):
        with pytest.raises(PathViolationError, match="敏感文件"):
            normalize_and_validate(".env", repo_root)

    def test_env_local_blocked(self, repo_root: Path):
        env_local = repo_root / ".env.local"
        env_local.write_text("LOCAL=true", encoding="utf-8")
        with pytest.raises(PathViolationError, match="敏感文件"):
            normalize_and_validate(".env.local", repo_root)

    def test_class_extension_blocked(self, repo_root: Path):
        # target/Main.class hits excluded directory first, which is fine
        # Test with a class file outside excluded dirs
        (repo_root / "lib").mkdir()
        (repo_root / "lib" / "App.class").write_bytes(b"\xca\xfe\xba\xbe")
        with pytest.raises(PathViolationError, match="排除文件类型"):
            normalize_and_validate("lib/App.class", repo_root)

    def test_jar_extension_blocked(self, repo_root: Path):
        # Create a jar file in a non-excluded directory
        (repo_root / "lib").mkdir()
        (repo_root / "lib" / "app.jar").write_bytes(b"PK")
        with pytest.raises(PathViolationError, match="排除文件类型"):
            normalize_and_validate("lib/app.jar", repo_root)

    def test_pem_extension_blocked(self, repo_root: Path):
        # .pem files are blocked by SENSITIVE_PATTERNS (containing "pem")
        (repo_root / "ssl").mkdir()
        (repo_root / "ssl" / "server.pem").write_text("-----BEGIN CERTIFICATE-----", encoding="utf-8")
        with pytest.raises(PathViolationError, match="敏感文件|排除文件类型"):
            normalize_and_validate("ssl/server.pem", repo_root)

    @pytest.mark.skipif(os.name == "nt", reason="Symlinks require privileges on Windows")
    def test_symlink_escape_blocked(self, repo_root: Path):
        """Test that symlinks pointing outside repo are blocked."""
        outside = repo_root.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        symlink = repo_root / "src" / "escape.txt"
        symlink.symlink_to(outside)
        with pytest.raises(PathViolationError, match="符号链接逃逸"):
            normalize_and_validate("src/escape.txt", repo_root)

    @pytest.mark.skipif(os.name == "nt", reason="Symlinks require privileges on Windows")
    def test_symlink_within_repo_allowed(self, repo_root: Path):
        """Test that symlinks pointing within repo are allowed."""
        target = repo_root / "src" / "main" / "java" / "com" / "example" / "Main.java"
        link = repo_root / "src" / "link.java"
        link.symlink_to(target)
        # This should pass (symlink target is within repo)
        result = normalize_and_validate("src/link.java", repo_root)
        assert result.exists()


class TestIsWithinRepo:
    """Tests for is_within_repo helper."""

    def test_path_within_repo(self, repo_root: Path):
        inner = repo_root / "src" / "Main.java"
        assert is_within_repo(inner, repo_root) is True

    def test_path_outside_repo(self, repo_root: Path):
        outside = repo_root.parent / "outside.txt"
        assert is_within_repo(outside, repo_root) is False

    def test_parent_directory(self, repo_root: Path):
        assert is_within_repo(repo_root.parent, repo_root) is False
