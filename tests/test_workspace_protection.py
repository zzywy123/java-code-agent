"""Tests for workspace protection - agent cannot modify its own source code."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.security.path_guard import PathViolationError, normalize_and_validate


@pytest.fixture
def project_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Create a workspace with both agent source and target repo.

    Returns (agent_root, repo_root).
    """
    # Agent's own source code
    agent_root = tmp_path / "java-coding-agent"
    agent_src = agent_root / "src" / "agent"
    agent_tools = agent_src / "tools"
    agent_tools.mkdir(parents=True)
    (agent_src / "__init__.py").write_text("", encoding="utf-8")
    (agent_src / "config.py").write_text("# agent config", encoding="utf-8")
    (agent_tools / "__init__.py").write_text("", encoding="utf-8")

    # Target repository
    repo_root = tmp_path / "target-repo"
    repo_src = repo_root / "src" / "main" / "java"
    repo_src.mkdir(parents=True)
    (repo_src / "App.java").write_text("public class App {}", encoding="utf-8")

    return agent_root, repo_root


class TestWorkspaceProtection:
    """Verify agent cannot access or modify its own source code."""

    def test_agent_source_outside_repo_boundary(self, project_workspace):
        """Agent's own source is outside the target repo boundary."""
        agent_root, repo_root = project_workspace
        agent_file = str(agent_root / "src" / "agent" / "config.py")

        # Trying to access agent source via absolute path should fail
        with pytest.raises(PathViolationError, match="路径穿越"):
            normalize_and_validate(agent_file, repo_root)

    def test_agent_source_via_traversal(self, project_workspace):
        """Agent cannot reach its own source via path traversal."""
        agent_root, repo_root = project_workspace
        # Calculate relative path from repo to agent
        rel_path = os.path.relpath(agent_root, repo_root)
        traversal = rel_path.replace("\\", "/") + "/src/agent/config.py"

        with pytest.raises(PathViolationError):
            normalize_and_validate(traversal, repo_root)

    def test_target_repo_files_accessible(self, project_workspace):
        """Files within the target repo are accessible."""
        _, repo_root = project_workspace
        result = normalize_and_validate("src/main/java/App.java", repo_root)
        assert result.exists()
        assert result.name == "App.java"

    def test_parent_directory_not_accessible(self, project_workspace):
        """Parent directory of repo is not accessible."""
        _, repo_root = project_workspace
        with pytest.raises(PathViolationError, match="路径穿越"):
            normalize_and_validate("../", repo_root)

    def test_sibling_directory_not_accessible(self, project_workspace):
        """Sibling directories of repo are not accessible."""
        _, repo_root = project_workspace
        with pytest.raises(PathViolationError, match="路径穿越"):
            normalize_and_validate("../java-coding-agent/src/agent/config.py", repo_root)
