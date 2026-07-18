"""Tests for build tools: run_tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.models import ToolStatus
from agent.tools.build_tools import (
    RunTestsTool,
    _discover_java_home,
    _remove_secrets,
    detect_build_tool,
)


class TestRunTestsTool:
    """Tests for run_tests tool."""

    def test_invalid_tool(self, tmp_path: Path):
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t1", tool="invalid", goals=["test"])
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_empty_goals(self, tmp_path: Path):
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t2", tool="maven", goals=[])
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_no_tool(self, tmp_path: Path):
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t3", tool="", goals=["test"])
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_invalid_extra_arg(self, tmp_path: Path):
        """Invalid extra args should be rejected by validate_argv."""
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(
            tool_call_id="t5",
            tool="maven",
            goals=["test"],
            extra_args=["--malicious"],
        )
        # Should be rejected (DENIED or INVALID_ARGUMENT)
        assert result.status in {ToolStatus.DENIED, ToolStatus.INVALID_ARGUMENT}

    def test_disallowed_maven_goal_is_rejected_before_lookup(self, tmp_path: Path):
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(
            tool_call_id="goal",
            tool="maven",
            goals=["org.codehaus.mojo:exec-maven-plugin:exec"],
        )
        assert result.status == ToolStatus.INVALID_ARGUMENT

    def test_maven_execution(self, tmp_path: Path):
        """When mvn is available, it should execute (may fail due to no pom.xml)."""
        import shutil
        if not shutil.which("mvn"):
            pytest.skip("mvn not in PATH")
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t6", tool="maven", goals=["test"])
        # May fail for various reasons (no pom.xml, validation issues, etc.)
        assert result.status in {
            ToolStatus.EXECUTION_ERROR, ToolStatus.NOT_FOUND,
            ToolStatus.SUCCESS, ToolStatus.DENIED,
        }

    def test_gradle_not_found(self, tmp_path: Path):
        """When gradle is not in PATH, it should return NOT_FOUND."""
        import shutil
        if shutil.which("gradle"):
            pytest.skip("gradle is in PATH")
        tool = RunTestsTool(repo_root=tmp_path)
        result = tool.execute(tool_call_id="t7", tool="gradle", goals=["test"])
        assert result.status == ToolStatus.NOT_FOUND


def test_detect_build_tool_prefers_maven(tmp_path: Path):
    (tmp_path / "build.gradle.kts").write_text("plugins {}", encoding="utf-8")
    assert detect_build_tool(tmp_path) == "gradle"
    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    assert detect_build_tool(tmp_path) == "maven"


def test_discover_java_home_ignores_jre_and_uses_javac_runtime(tmp_path, monkeypatch):
    jre = tmp_path / "jre"
    jre.mkdir()
    jdk = tmp_path / "jdk"
    (jdk / "bin").mkdir(parents=True)
    javac_name = "javac.exe" if __import__("os").name == "nt" else "javac"
    (jdk / "bin" / javac_name).write_text("", encoding="utf-8")
    monkeypatch.setattr("agent.tools.build_tools.shutil.which", lambda name: "javac-proxy")
    monkeypatch.setattr(
        "agent.tools.build_tools.subprocess.run",
        lambda *args, **kwargs: type("Result", (), {
            "stdout": "",
            "stderr": f"    java.home = {jdk}\n",
        })(),
    )

    result = _discover_java_home({"JAVA_HOME": str(jre)})

    assert result == str(jdk.resolve())


def test_remove_secrets_keeps_build_configuration():
    environment = {
        "DEEPSEEK_API_KEY": "secret",
        "GITHUB_TOKEN": "secret",
        "DATABASE_PASSWORD": "secret",
        "PATH": "bin",
        "JAVA_HOME": "jdk",
        "MAVEN_OPTS": "-Xmx1g",
    }

    _remove_secrets(environment)

    assert environment == {
        "PATH": "bin",
        "JAVA_HOME": "jdk",
        "MAVEN_OPTS": "-Xmx1g",
    }
