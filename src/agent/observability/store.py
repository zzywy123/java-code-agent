"""Persistent trace storage and aggregate metrics."""

from __future__ import annotations

import logging
from pathlib import Path

from agent.observability.models import GlobalMetrics, ToolAggregate, TraceTree
from agent.observability.tracer import TraceCollector

logger = logging.getLogger(__name__)


class TraceStore:
    def __init__(self, persist_dir: Path) -> None:
        self._dir = persist_dir.resolve()
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, collector: TraceCollector) -> TraceTree:
        tree = collector.to_tree()
        path = self._dir / f"{tree.trace_id}.json"
        path.write_text(tree.model_dump_json(indent=2), encoding="utf-8")
        return tree

    def load_for_session(
        self,
        session_id: str,
        project_id: str | None = None,
    ) -> TraceCollector | None:
        trees = [tree for tree in self.list_traces() if tree.session_id == session_id]
        if not trees:
            return None
        if project_id is not None:
            scoped = [tree for tree in trees if tree.project_id == project_id]
            legacy = [tree for tree in trees if tree.project_id is None]
            trees = scoped or legacy
            if not trees:
                return None
        latest = max(trees, key=lambda tree: tree.root.start_time)
        return TraceCollector.from_tree(latest)

    def get_for_session(
        self,
        session_id: str,
        project_id: str | None = None,
    ) -> TraceTree | None:
        collector = self.load_for_session(session_id, project_id)
        return collector.to_tree() if collector is not None else None

    def delete_for_session(self, session_id: str) -> int:
        """Delete all valid persisted traces owned by one Session."""
        deleted = 0
        for path in self._dir.glob("*.json"):
            try:
                tree = TraceTree.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Skipping invalid trace file %s: %s", path, exc)
                continue
            if tree.session_id == session_id:
                path.unlink()
                deleted += 1
        return deleted

    def list_traces(self) -> list[TraceTree]:
        traces: list[TraceTree] = []
        for path in self._dir.glob("*.json"):
            try:
                traces.append(TraceTree.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Skipping invalid trace file %s: %s", path, exc)
        return traces

    def get_metrics(
        self,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> GlobalMetrics:
        traces = self.list_traces()
        if session_id is not None:
            traces = [trace for trace in traces if trace.session_id == session_id]
        if project_id is not None:
            traces = [trace for trace in traces if trace.project_id == project_id]
        tools: dict[str, ToolAggregate] = {}
        all_costs: list[float] = []
        all_usage_count = 0
        metrics = GlobalMetrics(
            trace_count=len(traces),
            session_count=len({trace.session_id for trace in traces}),
        )
        for trace in traces:
            metrics.average_trace_duration_ms += trace.total_duration_ms
            for usage in trace.token_usages:
                all_usage_count += 1
                metrics.llm_calls += 1
                metrics.input_tokens += usage.input_tokens
                metrics.output_tokens += usage.output_tokens
                metrics.total_tokens += usage.total_tokens
                if usage.cost is not None:
                    all_costs.append(usage.cost)
            for tool in trace.tool_metrics:
                metrics.tool_calls += 1
                aggregate = tools.setdefault(tool.name, ToolAggregate())
                aggregate.calls += 1
                aggregate.total_duration_ms += tool.duration_ms
                if tool.status == "success":
                    metrics.tool_successes += 1
                    aggregate.successes += 1
                else:
                    metrics.tool_failures += 1
                    aggregate.failures += 1
        if traces:
            metrics.average_trace_duration_ms /= len(traces)
        if all_usage_count and len(all_costs) == all_usage_count:
            metrics.total_cost = sum(all_costs)
        metrics.tools = tools
        return metrics
