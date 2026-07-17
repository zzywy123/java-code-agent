"""Coder Agent: code modification with write permission.

The Coder can:
- Read code (read_file, search_code, list_files)
- Modify code (apply_patch, undo_patch)
- View git status (git_status, git_diff)

All write operations require approval through the permission layer.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.agents.artifacts import ArtifactFactory
from agent.agents.permission import AgentRole, PermissionManager
from agent.models import AgentArtifact, CodeChangeArtifact, PatchRecord
from agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class CoderAgent:
    """Agent for code modification with write access.

    Role: CODER (read + write)
    Capabilities: search, read, patch, undo
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        permission_manager: PermissionManager,
    ) -> None:
        self._tools = tool_registry
        self._permissions = permission_manager
        self._role = AgentRole.CODER

    def run(self, task: str, context: dict[str, Any] | None = None) -> CodeChangeArtifact:
        """Execute a coding task.

        Args:
            task: Description of the code change needed
            context: Optional context from Researcher

        Returns:
            CodeChangeArtifact with proposed changes
        """
        logger.info("Coder: %s", task[:100])

        # Verify write permission
        self._permissions.assert_tool_allowed(self._role, "apply_patch")

        # Extract file paths and changes from context
        affected_files = []
        if context:
            # Get relevant files from researcher's analysis
            search_artifact = context.get("search_artifact")
            if search_artifact and hasattr(search_artifact, "relevant_files"):
                affected_files = search_artifact.relevant_files

        return ArtifactFactory.create_code_change_artifact(
            description=task,
            affected_files=affected_files,
            rationale=f"Coder 准备修改 {len(affected_files)} 个文件",
        )

    def apply_patch(self, path: str, unified_diff: str) -> PatchRecord | None:
        """Apply a patch to a file.

        Requires write permission. Returns PatchRecord on success.
        """
        self._permissions.assert_tool_allowed(self._role, "apply_patch")

        result = self._tools.execute(
            name="apply_patch",
            tool_call_id="coder_patch",
            path=path,
            unified_diff=unified_diff,
        )

        if result.status.value == "success":
            patch_data = result.metadata.get("patch_record")
            if patch_data:
                return PatchRecord.model_validate(patch_data)
        return None

    def undo_patch(self, path: str, unified_diff: str, hash_before: str) -> bool:
        """Undo a patch."""
        self._permissions.assert_tool_allowed(self._role, "undo_patch")

        result = self._tools.execute(
            name="undo_patch",
            tool_call_id="coder_undo",
            path=path,
            unified_diff=unified_diff,
            hash_before=hash_before,
        )
        return result.status.value == "success"
