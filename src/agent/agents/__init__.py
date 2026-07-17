"""Multi-Agent subsystem.

Provides Supervisor, Researcher, Coder, Tester, and Verifier agents
with role-based permission isolation and structured artifact handoff.
"""

from agent.agents.artifacts import ArtifactFactory
from agent.agents.coder import CoderAgent
from agent.agents.permission import AgentRole, PermissionManager
from agent.agents.researcher import ResearcherAgent
from agent.agents.supervisor import SupervisorAgent
from agent.agents.tester import TesterAgent
from agent.agents.verifier import VerifierAgent

__all__ = [
    "AgentRole",
    "ArtifactFactory",
    "CoderAgent",
    "PermissionManager",
    "ResearcherAgent",
    "SupervisorAgent",
    "TesterAgent",
    "VerifierAgent",
]
