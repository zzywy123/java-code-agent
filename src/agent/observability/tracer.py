"""Hierarchical trace collector backed by contextvars."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from agent.observability.context import current_session, current_span_id, current_trace
from agent.observability.models import TokenUsage, ToolMetric, TraceEvent, TraceSpan, TraceTree


class TraceCollector:
    """Collect mutable flat spans and render them as a TraceTree."""

    def __init__(
        self,
        session_id: str,
        trace_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        now = time.time()
        self.session_id = session_id
        self.project_id = project_id
        self.trace_id = trace_id or str(uuid.uuid4())
        self.root_span_id = self.trace_id
        self._spans: dict[str, TraceSpan] = {
            self.root_span_id: TraceSpan(
                span_id=self.root_span_id,
                trace_id=self.trace_id,
                name="workflow.session",
                start_time=now,
            )
        }
        self._token_usages: list[TokenUsage] = []
        self._tool_metrics: list[ToolMetric] = []

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        parent_id = current_span_id.get() or self.root_span_id
        span = TraceSpan(
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_id,
            trace_id=self.trace_id,
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self._spans[span.span_id] = span
        token = current_span_id.set(span.span_id)
        try:
            yield span
        except TimeoutError as exc:
            self._finish_span(span, "timeout")
            span.events.append(TraceEvent(
                timestamp=time.time(),
                name="exception",
                attributes={"type": type(exc).__name__},
            ))
            raise
        except Exception as exc:
            if type(exc).__name__ in {"GraphInterrupt", "NodeInterrupt"}:
                self._finish_span(span, "interrupted")
            else:
                self._finish_span(span, "error")
                span.events.append(TraceEvent(
                    timestamp=time.time(),
                    name="exception",
                    attributes={"type": type(exc).__name__},
                ))
            raise
        else:
            final_status = span.status if span.status != "running" else "ok"
            self._finish_span(span, final_status)
        finally:
            current_span_id.reset(token)

    def record_token_usage(self, usage: TokenUsage) -> None:
        self._token_usages.append(usage)
        span = self._spans.get(current_span_id.get() or self.root_span_id)
        if span is not None:
            span.events.append(TraceEvent(
                timestamp=time.time(),
                name="token_usage",
                attributes=usage.model_dump(mode="json"),
            ))

    def token_usage_count(self) -> int:
        return len(self._token_usages)

    def token_usages_since(self, index: int) -> list[TokenUsage]:
        return list(self._token_usages[index:])

    def record_tool_metric(
        self,
        name: str,
        status: str,
        duration_ms: float,
    ) -> None:
        self._tool_metrics.append(ToolMetric(
            name=name,
            status=status,
            duration_ms=duration_ms,
            trace_id=self.trace_id,
            session_id=current_session.get(),
        ))

    def interrupt(self) -> None:
        self._finish_root("interrupted")

    def resume(self) -> None:
        root = self._spans[self.root_span_id]
        root.status = "running"
        root.end_time = None
        root.attributes["resume_count"] = int(root.attributes.get("resume_count", 0)) + 1
        root.attributes["active_started_at"] = time.time()

    def finish(self, status: str = "ok") -> None:
        self._finish_root(status)

    def to_tree(self) -> TraceTree:
        span_copies = {
            span_id: span.model_copy(deep=True, update={"children": []})
            for span_id, span in self._spans.items()
        }
        for span_id, span in span_copies.items():
            if span_id == self.root_span_id:
                continue
            parent = span_copies.get(span.parent_span_id or self.root_span_id)
            if parent is not None:
                parent.children.append(span)
        for span in span_copies.values():
            span.children.sort(key=lambda child: child.start_time)

        total_cost = None
        known_costs = [usage.cost for usage in self._token_usages if usage.cost is not None]
        if known_costs and len(known_costs) == len(self._token_usages):
            total_cost = sum(known_costs)
        root = span_copies[self.root_span_id]
        return TraceTree(
            trace_id=self.trace_id,
            session_id=self.session_id,
            project_id=self.project_id,
            root=root,
            total_duration_ms=float(root.duration_ms or 0.0),
            total_tokens=sum(usage.total_tokens for usage in self._token_usages),
            total_cost=total_cost,
            token_usages=list(self._token_usages),
            tool_metrics=list(self._tool_metrics),
        )

    @classmethod
    def from_tree(cls, tree: TraceTree) -> "TraceCollector":
        collector = cls(tree.session_id, tree.trace_id, tree.project_id)
        collector._spans = {}

        def visit(span: TraceSpan) -> None:
            collector._spans[span.span_id] = span.model_copy(deep=True, update={"children": []})
            for child in span.children:
                visit(child)

        visit(tree.root)
        collector.root_span_id = tree.root.span_id
        collector._token_usages = list(tree.token_usages)
        collector._tool_metrics = list(tree.tool_metrics)
        return collector

    @staticmethod
    def _finish_span(span: TraceSpan, status: str) -> None:
        end = time.time()
        span.end_time = end
        span.duration_ms = max(0.0, (end - span.start_time) * 1000)
        span.status = status  # type: ignore[assignment]

    def _finish_root(self, status: str) -> None:
        root = self._spans[self.root_span_id]
        end = time.time()
        active_started = float(root.attributes.pop("active_started_at", root.start_time))
        accumulated = float(root.attributes.get("active_duration_ms", 0.0))
        accumulated += max(0.0, (end - active_started) * 1000)
        root.attributes["active_duration_ms"] = accumulated
        root.end_time = end
        root.duration_ms = accumulated
        root.status = status  # type: ignore[assignment]


@contextmanager
def observe_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[TraceSpan | None]:
    collector = current_trace.get()
    if collector is None:
        yield None
        return
    with collector.span(name, attributes) as span:
        yield span


def record_token_usage(usage: TokenUsage) -> None:
    collector = current_trace.get()
    if collector is not None:
        collector.record_token_usage(usage)


def record_tool_metric(name: str, status: str, duration_ms: float) -> None:
    collector = current_trace.get()
    if collector is not None:
        collector.record_tool_metric(name, status, duration_ms)
