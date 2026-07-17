"""Structured observability models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TraceStatus = Literal["running", "interrupted", "ok", "error", "timeout"]


class TraceEvent(BaseModel):
    timestamp: float
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str
    provider: str
    estimated: bool = False
    cost: float | None = None
    duration_ms: float = 0.0


class ToolMetric(BaseModel):
    name: str
    status: str
    duration_ms: float
    trace_id: str
    session_id: str | None = None


class TraceSpan(BaseModel):
    span_id: str
    parent_span_id: str | None = None
    trace_id: str
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: TraceStatus = "running"
    events: list[TraceEvent] = Field(default_factory=list)
    children: list["TraceSpan"] = Field(default_factory=list)


class TraceTree(BaseModel):
    trace_id: str
    session_id: str
    project_id: str | None = None
    root: TraceSpan
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float | None = None
    token_usages: list[TokenUsage] = Field(default_factory=list)
    tool_metrics: list[ToolMetric] = Field(default_factory=list)


class ToolAggregate(BaseModel):
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_duration_ms: float = 0.0


class GlobalMetrics(BaseModel):
    trace_count: int = 0
    session_count: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float | None = None
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    average_trace_duration_ms: float = 0.0
    tools: dict[str, ToolAggregate] = Field(default_factory=dict)
