"""Agent state definition for LangGraph.

Separated from models.py to avoid circular imports.
Uses add_messages reducer for automatic message accumulation.
"""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent.models import PatchRecord, ToolCallRequest


class AgentState(TypedDict):
    """LangGraph agent state.

    Uses add_messages reducer for automatic message accumulation.
    Router functions must NOT modify this state directly.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    consecutive_failures: int
    pending_tool_calls: list[ToolCallRequest]
    patches: list[PatchRecord]
    final_answer: str | None
    error: str | None
    approval_rejected: NotRequired[bool]
    session_id: NotRequired[str]
