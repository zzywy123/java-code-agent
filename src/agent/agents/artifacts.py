"""Artifact factory for structured agent handoff.

Provides serialization/deserialization of AgentArtifact union types.
Each artifact type is a Pydantic model with a discriminator field.
"""

from __future__ import annotations

from typing import Any, Literal

from agent.models import (
    ARTIFACT_TYPES,
    AgentArtifact,
    CodeChangeArtifact,
    ReviewArtifact,
    SearchArtifact,
    TestResultArtifact,
    parse_artifact,
)


class ArtifactFactory:
    """Factory for creating and parsing agent artifacts.

    All artifacts are Pydantic models with an 'artifact_type' discriminator.
    """

    @staticmethod
    def create_search_artifact(
        query: str,
        results: list[Any] | None = None,
        analysis: str = "",
        relevant_files: list[str] | None = None,
        direct_answer: str | None = None,
        render_hint: Literal["diff", "text"] | None = None,
    ) -> SearchArtifact:
        """Create a search results artifact."""
        return SearchArtifact(
            query=query,
            results=results or [],
            analysis=analysis,
            relevant_files=relevant_files or [],
            direct_answer=direct_answer,
            render_hint=render_hint,
        )

    @staticmethod
    def create_code_change_artifact(
        description: str,
        patches: list[Any] | None = None,
        affected_files: list[str] | None = None,
        rationale: str = "",
    ) -> CodeChangeArtifact:
        """Create a code change artifact."""
        return CodeChangeArtifact(
            description=description,
            patches=patches or [],
            affected_files=affected_files or [],
            rationale=rationale,
        )

    @staticmethod
    def create_test_result_artifact(
        command: str,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
        tests_passed: int = 0,
        tests_failed: int = 0,
    ) -> TestResultArtifact:
        """Create a test result artifact."""
        return TestResultArtifact(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            success=exit_code == 0 and tests_failed == 0,
        )

    @staticmethod
    def create_review_artifact(
        approved: bool,
        issues: list[str] | None = None,
        suggestions: list[str] | None = None,
        summary: str = "",
    ) -> ReviewArtifact:
        """Create a review verdict artifact."""
        return ReviewArtifact(
            approved=approved,
            issues=issues or [],
            suggestions=suggestions or [],
            summary=summary,
        )

    @staticmethod
    def parse(data: dict[str, Any]) -> AgentArtifact:
        """Parse a dict into the correct artifact type."""
        return parse_artifact(data)

    @staticmethod
    def to_dict(artifact: AgentArtifact) -> dict[str, Any]:
        """Serialize an artifact to a dict."""
        return artifact.model_dump(mode="json")
