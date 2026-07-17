"""Tester Agent: runs tests and builds with restricted commands.

The Tester can:
- Read code (read_file, search_code, list_files)
- Run tests (run_tests) — only Maven/Gradle commands
- View git status (git_status, git_diff)

Cannot modify files or execute arbitrary commands.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.agents.artifacts import ArtifactFactory
from agent.agents.permission import AgentRole, PermissionManager
from agent.models import AgentArtifact, TestResultArtifact
from agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class TesterAgent:
    """Agent for running tests and builds.

    Role: TESTER (read + restricted execution)
    Capabilities: search, read, run_tests
    """

    __test__ = False

    def __init__(
        self,
        tool_registry: ToolRegistry,
        permission_manager: PermissionManager,
    ) -> None:
        self._tools = tool_registry
        self._permissions = permission_manager
        self._role = AgentRole.TESTER

    def run(self, task: str, context: dict[str, Any] | None = None) -> TestResultArtifact:
        """Execute a test task.

        Args:
            task: Description of what to test
            context: Optional context from Coder

        Returns:
            TestResultArtifact with test results
        """
        logger.info("Tester: %s", task[:100])

        # Verify execution permission
        self._permissions.assert_tool_allowed(self._role, "run_tests")

        # Run Maven tests by default
        return self.run_maven_tests(["test"])

    def run_maven_tests(self, goals: list[str], module: str = "") -> TestResultArtifact:
        """Run Maven tests."""
        self._permissions.assert_tool_allowed(self._role, "run_tests")

        result = self._tools.execute(
            name="run_tests",
            tool_call_id="tester_maven",
            tool="maven",
            goals=goals,
            module=module,
        )

        # Parse test results from output
        output = result.output
        tests_passed = 0
        tests_failed = 0

        # Simple parsing of Maven test output
        import re
        pass_match = re.search(r"Tests run: (\d+), Failures: (\d+)", output)
        if pass_match:
            total = int(pass_match.group(1))
            failures = int(pass_match.group(2))
            tests_failed = failures
            tests_passed = total - failures

        return ArtifactFactory.create_test_result_artifact(
            command=f"mvn {' '.join(goals)}",
            exit_code=result.metadata.get("exit_code", -1),
            stdout=output,
            stderr="",
            tests_passed=tests_passed,
            tests_failed=tests_failed,
        )

    def run_gradle_tests(self, tasks: list[str]) -> TestResultArtifact:
        """Run Gradle tests."""
        self._permissions.assert_tool_allowed(self._role, "run_tests")

        result = self._tools.execute(
            name="run_tests",
            tool_call_id="tester_gradle",
            tool="gradle",
            goals=tasks,
        )

        return ArtifactFactory.create_test_result_artifact(
            command=f"gradle {' '.join(tasks)}",
            exit_code=result.metadata.get("exit_code", -1),
            stdout=result.output,
        )
