"""Context-local trace and session bindings."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator


current_trace: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "current_trace",
    default=None,
)
current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session",
    default=None,
)
current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span_id",
    default=None,
)


@contextmanager
def activate_trace(collector: Any, session_id: str) -> Iterator[None]:
    trace_token = current_trace.set(collector)
    session_token = current_session.set(session_id)
    span_token = current_span_id.set(collector.root_span_id)
    try:
        yield
    finally:
        current_span_id.reset(span_token)
        current_session.reset(session_token)
        current_trace.reset(trace_token)
