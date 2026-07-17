"""Tests for command_guard module - argv-level command validation."""

from __future__ import annotations

import pytest

from agent.security.command_guard import (
    ALLOWED_EXECUTABLES,
    ALLOWED_MAVEN_GOALS,
    ALLOWED_GRADLE_TASKS,
    CommandViolationError,
    build_gradle_argv,
    build_maven_argv,
    validate_argv,
)


class TestValidateArgv:
    """Tests for validate_argv function."""

    def test_empty_command_rejected(self):
        allowed, reason = validate_argv([])
        assert allowed is False
        assert "空命令" in reason

    def test_mvn_allowed(self):
        allowed, _ = validate_argv(["./mvnw", "clean", "test"])
        assert allowed is True

    def test_gradle_allowed(self):
        allowed, _ = validate_argv(["./gradlew", "test"])
        assert allowed is True

    def test_java_allowed(self):
        allowed, _ = validate_argv(["java", "-version"])
        assert allowed is True

    def test_git_allowed(self):
        allowed, _ = validate_argv(["git", "status"])
        assert allowed is True

    def test_rm_blocked(self):
        allowed, reason = validate_argv(["rm", "-rf", "/"])
        assert allowed is False
        assert "不在允许列表中" in reason

    def test_curl_blocked(self):
        allowed, reason = validate_argv(["curl", "http://evil.com"])
        assert allowed is False

    def test_sudo_blocked(self):
        allowed, reason = validate_argv(["sudo", "rm", "-rf", "/"])
        assert allowed is False

    def test_pipe_blocked_in_args(self):
        allowed, reason = validate_argv(["./mvnw", "test", "|", "cat"])
        assert allowed is False
        assert "shell 元字符" in reason or "被阻止的模式" in reason

    def test_semicolon_blocked(self):
        allowed, reason = validate_argv(["./mvnw", "test;rm -rf /"])
        assert allowed is False

    def test_backtick_blocked(self):
        allowed, reason = validate_argv(["./mvnw", "`whoami`"])
        assert allowed is False

    def test_dollar_paren_blocked(self):
        allowed, reason = validate_argv(["./mvnw", "$(whoami)"])
        assert allowed is False

    def test_redirect_blocked(self):
        allowed, reason = validate_argv(["./mvnw", "test", ">", "/tmp/out"])
        assert allowed is False

    def test_powershell_blocked(self):
        allowed, reason = validate_argv(["powershell", "-c", "Get-Process"])
        assert allowed is False

    def test_cmd_blocked(self):
        allowed, reason = validate_argv(["cmd", "/c", "dir"])
        assert allowed is False

    def test_exe_extension_handled(self):
        """Windows executables with .exe extension should work."""
        allowed, _ = validate_argv(["./mvnw.cmd", "clean", "test"])
        assert allowed is True


class TestBuildMavenArgv:
    """Tests for build_maven_argv function."""

    def test_basic_test(self):
        import os
        argv = build_maven_argv(["test"])
        expected_exe = "mvnw.cmd" if os.name == "nt" else "./mvnw"
        assert argv == [expected_exe, "-B", "test"]

    def test_clean_test(self):
        import os
        argv = build_maven_argv(["clean", "test"])
        expected_exe = "mvnw.cmd" if os.name == "nt" else "./mvnw"
        assert argv == [expected_exe, "-B", "clean", "test"]

    def test_with_module(self):
        argv = build_maven_argv(["test"], module="order-service")
        assert "-pl" in argv
        assert "order-service" in argv

    def test_with_extra_args(self):
        argv = build_maven_argv(["test"], extra_args=["-DskipITs=true"])
        assert "-DskipITs=true" in argv

    def test_without_wrapper(self):
        argv = build_maven_argv(["test"], use_wrapper=False)
        assert argv[0] == "mvn"

    def test_invalid_goal_rejected(self):
        with pytest.raises(CommandViolationError, match="goal"):
            build_maven_argv(["deploy"])

    def test_invalid_extra_arg_rejected(self):
        with pytest.raises(CommandViolationError, match="额外参数"):
            build_maven_argv(["test"], extra_args=["--malicious"])

    def test_all_allowed_goals(self):
        for goal in ALLOWED_MAVEN_GOALS:
            argv = build_maven_argv([goal])
            assert len(argv) >= 3


class TestBuildGradleArgv:
    """Tests for build_gradle_argv function."""

    def test_basic_test(self):
        argv = build_gradle_argv(["test"])
        assert argv[0] == "./gradlew"
        assert "test" in argv

    def test_with_project_path(self):
        argv = build_gradle_argv(["test"], project_path="subproject")
        assert "-p" in argv
        assert "subproject" in argv

    def test_without_wrapper(self):
        argv = build_gradle_argv(["test"], use_wrapper=False)
        assert argv[0] == "gradle"

    def test_invalid_task_rejected(self):
        with pytest.raises(CommandViolationError, match="task"):
            build_gradle_argv(["publishToMavenLocal"])

    def test_all_allowed_tasks(self):
        for task in ALLOWED_GRADLE_TASKS:
            argv = build_gradle_argv([task])
            assert len(argv) >= 3
