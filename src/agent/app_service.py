"""UI-independent application service for the coding workflow."""

from __future__ import annotations

import json
import hashlib
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command

from agent.config import load_observability_config
from agent.models import (
    ApprovalDecision,
    PatchRecord,
    SessionState,
    SessionSummary,
    StreamEvent,
    StreamEventType,
    SubmitResult,
)
from agent.session import SessionManager
from agent.observability.context import activate_trace
from agent.observability.models import GlobalMetrics, TraceTree
from agent.observability.store import TraceStore
from agent.observability.tracer import TraceCollector, observe_span
from agent.workflow import initial_workflow_state

logger = logging.getLogger(__name__)


class AppService:
    """The only workflow API used by CLI and future web frontends."""

    def __init__(
        self,
        workflow: Any,
        session_manager: SessionManager,
        project_root: Path | str | None = None,
    ) -> None:
        self._workflow = workflow
        self._sessions = session_manager
        self._project_id = self._make_project_id(project_root)
        self._event_dir = session_manager.get_storage_dir() / "events"
        self._event_dir.mkdir(parents=True, exist_ok=True)
        self._events: dict[str, list[StreamEvent]] = {}
        self._correlations: dict[str, set[tuple[str, str]]] = {}
        observability = load_observability_config()
        trace_dir = Path(observability.trace_dir).expanduser()
        if not trace_dir.is_absolute():
            trace_dir = session_manager.get_storage_dir().parent / trace_dir
        self._observability_enabled = observability.enabled
        self._trace_store = TraceStore(trace_dir)
        self._active_traces: dict[str, TraceCollector] = {}

    def create_session(self, name: str = "") -> str:
        session_id = self._sessions.create_session(name)
        self._ensure_events_loaded(session_id)
        return session_id

    def list_sessions(self) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for item in self._sessions.list_sessions():
            session_id = item["session_id"]
            self._ensure_events_loaded(session_id)
            summaries.append(SessionSummary(
                session_id=session_id,
                name=item.get("name", ""),
                event_count=len(self._events[session_id]),
            ))
        return summaries

    def delete_session(self, session_id: str) -> str:
        """Delete a Session and return the active replacement Session ID."""
        self._sessions.delete_session(session_id)
        self._event_path(session_id).unlink(missing_ok=True)
        self._events.pop(session_id, None)
        self._correlations.pop(session_id, None)
        self._active_traces.pop(session_id, None)
        self._trace_store.delete_for_session(session_id)
        replacement = self._sessions.get_or_create_active_session()
        self._ensure_events_loaded(replacement)
        return replacement

    def submit(self, session_id: str, query: str) -> SubmitResult:
        """Run a new user query until completion or an approval interrupt."""
        if not query.strip():
            return self._error_result(session_id, "查询不能为空")
        try:
            self._sessions.get_thread_config(session_id)
        except KeyError as exc:
            return self._error_result(session_id, str(exc), persist=False)
        collector = (
            TraceCollector(session_id, project_id=self._project_id)
            if self._observability_enabled
            else None
        )
        if collector is not None:
            self._active_traces[session_id] = collector
        return self._run(
            session_id,
            initial_workflow_state(query.strip(), session_id),
            collector=collector,
        )

    def resume(
        self,
        session_id: str,
        decision: ApprovalDecision | bool,
    ) -> SubmitResult:
        """Resume the current LangGraph interrupt with an approval decision."""
        if isinstance(decision, bool):
            decision = ApprovalDecision(approved=decision)
        try:
            snapshot = self._workflow.get_state(
                self._sessions.get_thread_config(session_id)
            )
        except KeyError as exc:
            return self._error_result(session_id, str(exc), persist=False)
        if not getattr(snapshot, "interrupts", ()):
            return self._error_result(session_id, "当前会话没有待审批操作")
        collector = None
        if self._observability_enabled:
            collector = self._active_traces.get(session_id)
            if collector is None:
                collector = self._trace_store.load_for_session(
                    session_id,
                    self._project_id,
                )
            if collector is None:
                collector = TraceCollector(session_id, project_id=self._project_id)
            else:
                if collector.project_id is None:
                    collector.project_id = self._project_id
                collector.resume()
            self._active_traces[session_id] = collector
        return self._run(
            session_id,
            Command(resume={
                "approved": decision.approved,
                "reason": decision.reason,
            }),
            collector=collector,
        )

    def get_session(self, session_id: str) -> SessionState:
        config = self._sessions.get_thread_config(session_id)
        snapshot = self._workflow.get_state(config)
        state = dict(snapshot.values or {})
        approval_data = self._approval_from_interrupts(
            getattr(snapshot, "interrupts", ())
        )
        self._ensure_events_loaded(session_id)
        return SessionState(
            session_id=session_id,
            messages=[self._serialize_message(message) for message in state.get("messages", [])],
            patches=self._coerce_patches(state.get("patches", [])),
            final_answer=state.get("final_answer"),
            error=state.get("error"),
            needs_approval=approval_data is not None,
            approval_data=approval_data,
            event_count=len(self._events[session_id]),
        )

    def stream_events(
        self,
        session_id: str,
        after_event_id: str | None = None,
    ) -> list[StreamEvent]:
        """Return persisted events, optionally only those after an event ID."""
        self._sessions.get_thread_config(session_id)
        self._ensure_events_loaded(session_id)
        events = self._events[session_id]
        if after_event_id is None:
            return list(events)
        for index, event in enumerate(events):
            if event.event_id == after_event_id:
                return list(events[index + 1:])
        return list(events)

    def get_trace(self, session_id: str) -> TraceTree | None:
        """Return the latest persisted trace for a Session."""
        self._sessions.get_thread_config(session_id)
        collector = self._active_traces.get(session_id)
        if collector is not None:
            return collector.to_tree()
        return self._trace_store.get_for_session(session_id, self._project_id)

    def get_metrics(
        self,
        scope: str = "all",
        session_id: str | None = None,
    ) -> GlobalMetrics:
        """Aggregate metrics for the selected Session, project or all traces."""
        if scope == "session":
            if session_id is None:
                raise ValueError("当前会话指标需要 session_id")
            self._sessions.get_thread_config(session_id)
            return self._trace_store.get_metrics(session_id=session_id)
        if scope == "project":
            if self._project_id is None:
                return GlobalMetrics()
            return self._trace_store.get_metrics(project_id=self._project_id)
        if scope == "all":
            return self._trace_store.get_metrics()
        raise ValueError(f"未知指标范围: {scope}")

    def remember_project_fact(self, key: str, memory_type: str, content: str) -> None:
        """Explicitly persist reusable project knowledge for later Sessions."""
        self._sessions.remember_project_fact(key, memory_type, content)

    def close(self) -> None:
        self._sessions.close()

    def _run(
        self,
        session_id: str,
        graph_input: Any,
        *,
        collector: TraceCollector | None,
    ) -> SubmitResult:
        self._ensure_events_loaded(session_id)
        event_start = len(self._events[session_id])
        config = self._sessions.get_thread_config(session_id)
        status = "completed"
        approval_data: dict[str, Any] | None = None
        token_start = collector.token_usage_count() if collector is not None else 0
        trace_context = (
            activate_trace(collector, session_id)
            if collector is not None
            else nullcontext()
        )
        trace_context.__enter__()

        try:
            with observe_span("workflow.run", {
                "resume": isinstance(graph_input, Command),
            }):
                for namespace, update in self._workflow.stream(
                    graph_input,
                    config=config,
                    stream_mode="updates",
                    subgraphs=True,
                ):
                    root_update = not namespace
                    if root_update and "__interrupt__" in update:
                        interrupts = update.get("__interrupt__", ())
                        approval_data = self._approval_from_interrupts(interrupts)
                        interrupt_id = self._interrupt_id(interrupts)
                        if approval_data is not None:
                            self._append_event(
                                session_id,
                                "approval_request",
                                approval_data,
                                correlation_id=interrupt_id,
                            )
                        status = "interrupted"
                        continue
                    self._consume_update(session_id, namespace, update)

            snapshot = self._workflow.get_state(config)
            state = dict(snapshot.values or {})
            if approval_data is None:
                approval_data = self._approval_from_interrupts(
                    getattr(snapshot, "interrupts", ())
                )
            if approval_data is not None:
                status = "interrupted"
            elif state.get("error"):
                status = "error"
                self._append_event(session_id, "error", {"message": str(state["error"])})
            elif not self._has_new_event(session_id, event_start, "done"):
                self._append_event(session_id, "done", {
                    "final_answer": state.get("final_answer"),
                    "patch_count": len(state.get("patches", [])),
                })
        except Exception as exc:
            logger.exception("Workflow execution failed for session %s", session_id)
            status = "error"
            approval_data = None
            self._append_event(session_id, "error", {"message": str(exc)})
            state = {}
        finally:
            trace_context.__exit__(None, None, None)

        if collector is not None:
            if status == "interrupted":
                collector.interrupt()
            else:
                collector.finish("error" if status == "error" else "ok")
                self._active_traces.pop(session_id, None)
            for index, usage in enumerate(
                collector.token_usages_since(token_start),
                start=token_start,
            ):
                self._append_event(
                    session_id,
                    "token_usage",
                    usage.model_dump(mode="json"),
                    correlation_id=f"{collector.trace_id}:{index}",
                )
            self._trace_store.save(collector)

        new_events = list(self._events[session_id][event_start:])
        return SubmitResult(
            session_id=session_id,
            status=status,
            events=new_events,
            final_answer=state.get("final_answer"),
            patches=self._coerce_patches(state.get("patches", [])),
            needs_approval=status == "interrupted",
            approval_data=approval_data,
            error=str(state.get("error") or "") or (
                self._last_error(new_events) if status == "error" else None
            ),
        )

    def _consume_update(
        self,
        session_id: str,
        namespace: tuple[str, ...],
        update: dict[str, Any],
    ) -> None:
        for node_name, payload in update.items():
            if payload is None or node_name == "__interrupt__":
                continue
            payload = payload if isinstance(payload, dict) else {}

            if namespace and node_name == "agent":
                for message in payload.get("messages", []):
                    if isinstance(message, AIMessage):
                        self._append_event(
                            session_id,
                            "agent_thinking",
                            {"agent": "coder"},
                            correlation_id=f"coder:{message.id}" if message.id else None,
                        )
                        for tool_call in message.tool_calls:
                            self._append_event(
                                session_id,
                                "tool_call",
                                {
                                    "agent": "coder",
                                    "name": tool_call.get("name", ""),
                                    "arguments": tool_call.get("args", {}),
                                    "tool_call_id": tool_call.get("id", ""),
                                },
                                correlation_id=str(tool_call.get("id") or "") or None,
                            )
                continue

            if namespace and node_name in {"tool_executor", "request_approval"}:
                for message in payload.get("messages", []):
                    if isinstance(message, ToolMessage):
                        content = str(message.content)
                        self._append_event(
                            session_id,
                            "tool_result",
                            {
                                "name": message.name or "",
                                "tool_call_id": message.tool_call_id,
                                "content": content,
                                "status": (
                                    "denied"
                                    if node_name == "request_approval"
                                    else self._tool_status_from_content(content)
                                ),
                            },
                            correlation_id=str(message.tool_call_id),
                        )
                for patch in self._coerce_patches(payload.get("patches", [])):
                    self._append_event(
                        session_id,
                        "patch_applied",
                        patch.model_dump(mode="json"),
                        correlation_id=patch.content_hash_after,
                    )
                continue

            if namespace:
                continue

            if node_name == "supervisor":
                route = payload.get("route", "researcher")
                self._append_event(session_id, "agent_switch", {
                    "agent": "coder" if route == "coding_workflow" else route,
                    "route": route,
                    "trace_id": payload.get("trace_id"),
                })
            elif node_name == "researcher":
                artifact = payload.get("search_artifact")
                if artifact is not None:
                    self._append_event(session_id, "rag_retrieval", artifact.model_dump(mode="json"))
            elif node_name == "tester":
                result = payload.get("test_result")
                if result is not None:
                    self._append_event(session_id, "agent_switch", {"agent": "tester"})
                    self._append_event(session_id, "test_result", result.model_dump(mode="json"))
            elif node_name == "verifier":
                result = payload.get("review_result")
                if result is not None:
                    self._append_event(session_id, "agent_switch", {"agent": "verifier"})
                    self._append_event(session_id, "review_result", result.model_dump(mode="json"))
            elif node_name == "rework":
                self._append_event(session_id, "rework", {
                    "rework_count": payload.get("rework_count", 0),
                })
            elif node_name == "memory":
                result = payload.get("memory_result")
                if result and result.get("saved"):
                    self._append_event(
                        session_id,
                        "memory_saved",
                        result,
                        correlation_id=str(result.get("key") or "") or None,
                    )
                self._append_event(session_id, "done", {
                    "final_answer": payload.get("final_answer"),
                })

    def _append_event(
        self,
        session_id: str,
        event_type: StreamEventType,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> StreamEvent | None:
        if correlation_id:
            key = (event_type, correlation_id)
            if key in self._correlations[session_id]:
                return None
            self._correlations[session_id].add(key)
        event = StreamEvent(
            session_id=session_id,
            event_type=event_type,
            data=data,
            correlation_id=correlation_id,
        )
        self._events[session_id].append(event)
        path = self._event_path(session_id)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.model_dump_json() + "\n")
        return event

    def _ensure_events_loaded(self, session_id: str) -> None:
        if session_id in self._events:
            return
        events: list[StreamEvent] = []
        path = self._event_path(session_id)
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    events.append(StreamEvent.model_validate_json(line))
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid event log line %s:%d: %s",
                        path,
                        line_number,
                        exc,
                    )
        self._events[session_id] = events
        self._correlations[session_id] = {
            (event.event_type, event.correlation_id)
            for event in events
            if event.correlation_id
        }

    def _event_path(self, session_id: str) -> Path:
        safe_id = "".join(char for char in session_id if char.isalnum() or char in "-_")
        return self._event_dir / f"{safe_id}.jsonl"

    def _error_result(
        self,
        session_id: str,
        message: str,
        *,
        persist: bool = True,
    ) -> SubmitResult:
        events: list[StreamEvent] = []
        if persist:
            self._ensure_events_loaded(session_id)
            event = self._append_event(session_id, "error", {"message": message})
            if event is not None:
                events.append(event)
        return SubmitResult(
            session_id=session_id,
            status="error",
            events=events,
            error=message,
        )

    @staticmethod
    def _coerce_patches(values: list[Any]) -> list[PatchRecord]:
        return [
            value if isinstance(value, PatchRecord) else PatchRecord.model_validate(value)
            for value in values
        ]

    @staticmethod
    def _serialize_message(message: BaseMessage) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": message.type,
            "id": message.id,
            "content": message.content,
        }
        render_hint = message.additional_kwargs.get("render_hint")
        if render_hint in {"diff", "text"}:
            data["render_hint"] = render_hint
        if isinstance(message, AIMessage) and message.tool_calls:
            data["tool_calls"] = message.tool_calls
        if isinstance(message, ToolMessage):
            data["tool_call_id"] = message.tool_call_id
            data["name"] = message.name
        return data

    @staticmethod
    def _approval_from_interrupts(interrupts: Any) -> dict[str, Any] | None:
        if not interrupts:
            return None
        value = getattr(interrupts[0], "value", None)
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _interrupt_id(interrupts: Any) -> str | None:
        if not interrupts:
            return None
        return str(getattr(interrupts[0], "id", "") or "") or None

    def _has_new_event(
        self,
        session_id: str,
        start: int,
        event_type: StreamEventType,
    ) -> bool:
        return any(
            event.event_type == event_type
            for event in self._events[session_id][start:]
        )

    @staticmethod
    def _last_error(events: list[StreamEvent]) -> str | None:
        for event in reversed(events):
            if event.event_type == "error":
                return str(event.data.get("message") or "") or None
        return None

    @staticmethod
    def _tool_status_from_content(content: str) -> str:
        content_lower = content.lower()
        for status in (
            "denied",
            "timeout",
            "execution_error",
            "invalid_argument",
            "not_found",
            "error",
            "success",
        ):
            if f"[{status}]" in content_lower:
                return status
        return "success"

    @staticmethod
    def _make_project_id(project_root: Path | str | None) -> str | None:
        if project_root is None:
            return None
        resolved = Path(project_root).expanduser().resolve()
        return hashlib.sha256(
            str(resolved).casefold().encode("utf-8")
        ).hexdigest()[:16]
