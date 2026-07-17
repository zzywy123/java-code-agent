"""Core data models for the Java Coding Agent.

All models use Pydantic v2 for validation.
AgentState uses LangGraph's TypedDict with add_messages reducer.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union
import time
import uuid

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    """Tool execution result status."""

    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"  # Security layer rejected the operation
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"  # File/resource not found
    INVALID_ARGUMENT = "invalid_argument"  # Bad parameters
    EXECUTION_ERROR = "execution_error"  # Runtime failure (e.g., test failure)
    PENDING_APPROVAL = "pending_approval"


class ToolCallRequest(BaseModel):
    """Structured representation of an LLM tool call."""

    id: str = Field(description="Tool call ID from LLM response")
    name: str = Field(description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolResult(BaseModel):
    """Structured result from tool execution."""

    tool_call_id: str = Field(description="Matching tool call ID")
    name: str = Field(description="Tool name")
    status: ToolStatus
    output: str = Field(default="", description="Text output (truncated if needed)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatchRecord(BaseModel):
    """Record of a file modification.

    Stores content hashes and unified diff instead of full source code.
    """

    file_path: str = Field(description="Normalized absolute path")
    content_hash_before: str = Field(description="SHA-256 of original content")
    content_hash_after: str = Field(description="SHA-256 of patched content")
    unified_diff: str = Field(description="Unified diff text")
    is_new_file: bool = Field(default=False, description="True if this creates a new file")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# Phase 2: RAG / Indexing Models
# ============================================================


class CodeSlice(BaseModel):
    """A method-level or class-level slice of Java source code.

    Produced by the JavaSlicer using tree-sitter AST parsing.
    """

    module: str = Field(description="Module name (from pom.xml artifactId)")
    package: str = Field(description="Java package (e.g. com.example.order)")
    class_name: str = Field(description="Fully qualified class name")
    method_name: str = Field(description="Method name, or '<class>' for class-level slice")
    file_path: str = Field(description="Relative path from repo root")
    start_line: int = Field(description="1-based start line (inclusive)")
    end_line: int = Field(description="1-based end line (inclusive)")
    content: str = Field(description="Source code content of the slice")
    imports: list[str] = Field(default_factory=list, description="Import statements from the file")
    docstring: str = Field(default="", description="JavaDoc or block comment above the method/class")
    symbol_signature: str = Field(
        default="",
        description="Canonical symbol signature: package.ClassName.methodName(paramTypes)",
    )


class CodeChunk(BaseModel):
    """A chunk of code stored in the index.

    chunk_id = SHA-256(file_path + symbol_signature + content_hash).
    """

    chunk_id: str = Field(description="SHA-256(file_path + symbol_signature + content_hash)")
    slice: CodeSlice
    embedding: list[float] | None = Field(default=None, description="Embedding vector")
    file_hash: str = Field(description="SHA-256 of the entire file content at indexing time")
    indexed_at: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    """A single result from hybrid search."""

    chunk: CodeChunk
    score: float = Field(description="Combined relevance score")
    source: str = Field(description="Origin: 'vector', 'bm25', or 'hybrid'")
    rank: int = Field(default=0, description="Rank position in result list")


class IndexStats(BaseModel):
    """Statistics from an indexing operation."""

    files_scanned: int = 0
    files_updated: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


class RAGResult(BaseModel):
    """Result from an Agentic RAG retrieval cycle."""

    answer: str = Field(default="", description="Synthesized answer from RAG")
    sources: list[SearchResult] = Field(default_factory=list, description="Retrieved sources")
    rounds_used: int = Field(default=0, description="Number of retrieval rounds executed")
    evidence_sufficient: bool = Field(default=False, description="Whether evidence was deemed sufficient")
    degraded: bool = Field(default=False, description="Whether the system degraded to fallback")
    queries_used: list[str] = Field(default_factory=list, description="All queries used (original + rewrites)")


# ============================================================
# Phase 2: Agent Artifact Models (Structured Pydantic Union)
# ============================================================


class SearchArtifact(BaseModel):
    """Artifact from Researcher: search results and analysis."""

    artifact_type: str = Field(default="search_results", description="Artifact type discriminator")
    query: str = Field(description="Original search query")
    results: list[SearchResult] = Field(default_factory=list)
    analysis: str = Field(default="", description="Researcher's analysis of the results")
    relevant_files: list[str] = Field(default_factory=list, description="File paths deemed relevant")
    tool_evidence: list[str] = Field(
        default_factory=list,
        description="Validated read-only tool output forwarded to downstream agents",
    )
    direct_answer: str | None = Field(
        default=None,
        description="Verbatim read-only tool result that bypasses RAG answer generation",
    )
    render_hint: Literal["diff", "text"] | None = Field(
        default=None,
        description="Safe UI rendering mode for a direct tool result",
    )


class CodeChangeArtifact(BaseModel):
    """Artifact from Coder: proposed code changes."""

    artifact_type: str = Field(default="code_change", description="Artifact type discriminator")
    description: str = Field(description="Description of the change")
    patches: list[PatchRecord] = Field(default_factory=list, description="Proposed patches")
    affected_files: list[str] = Field(default_factory=list)
    rationale: str = Field(default="", description="Why this change is needed")


class TestResultArtifact(BaseModel):
    """Artifact from Tester: test execution results."""

    artifact_type: str = Field(default="test_result", description="Artifact type discriminator")
    command: str = Field(description="Command that was executed")
    exit_code: int = Field(description="Process exit code")
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    tests_passed: int = 0
    tests_failed: int = 0
    success: bool = Field(description="Whether all tests passed")

    __test__ = False


class ReviewArtifact(BaseModel):
    """Artifact from Verifier: code review verdict."""

    artifact_type: str = Field(default="review", description="Artifact type discriminator")
    approved: bool = Field(description="Whether the change is approved")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    summary: str = Field(default="", description="Overall review summary")


# ============================================================
# Phase 3B: Application service and event models
# ============================================================

StreamEventType = Literal[
    "agent_thinking",
    "tool_call",
    "tool_result",
    "agent_switch",
    "rag_retrieval",
    "approval_request",
    "patch_applied",
    "test_result",
    "review_result",
    "memory_saved",
    "rework",
    "token_usage",
    "error",
    "done",
]


class StreamEvent(BaseModel):
    """One persisted, UI-independent workflow event."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    event_type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = Field(
        default=None,
        description="Stable tool call or interrupt ID used for replay deduplication",
    )


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""


class SubmitResult(BaseModel):
    session_id: str
    status: Literal["completed", "interrupted", "error"]
    events: list[StreamEvent] = Field(default_factory=list)
    final_answer: str | None = None
    patches: list[PatchRecord] = Field(default_factory=list)
    needs_approval: bool = False
    approval_data: dict[str, Any] | None = None
    error: str | None = None


class SessionState(BaseModel):
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    patches: list[PatchRecord] = Field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None
    needs_approval: bool = False
    approval_data: dict[str, Any] | None = None
    event_count: int = 0


class SessionSummary(BaseModel):
    session_id: str
    name: str = ""
    event_count: int = 0


# Discriminated union of all artifact types
AgentArtifact = Union[SearchArtifact, CodeChangeArtifact, TestResultArtifact, ReviewArtifact]

# Artifact type name → class mapping for deserialization
ARTIFACT_TYPES: dict[str, type[BaseModel]] = {
    "search_results": SearchArtifact,
    "code_change": CodeChangeArtifact,
    "test_result": TestResultArtifact,
    "review": ReviewArtifact,
}


def parse_artifact(data: dict[str, Any]) -> AgentArtifact:
    """Deserialize an artifact dict into the correct Pydantic model."""
    artifact_type = data.get("artifact_type", "")
    cls = ARTIFACT_TYPES.get(artifact_type)
    if cls is None:
        raise ValueError(f"Unknown artifact_type: {artifact_type}")
    return cls.model_validate(data)


# --- LangGraph State ---
# AgentState is defined in agent_state.py to avoid circular imports
