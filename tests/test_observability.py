"""Phase 3C observability tests."""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agent.config import LLMConfig, ObservabilityConfig
from agent.llm_client import ObservableChatModel
from agent.models import ApprovalDecision, ToolResult, ToolStatus
from agent.observability.context import activate_trace
from agent.observability.models import TokenUsage, TraceTree
from agent.observability.sanitizer import sanitize
from agent.observability.sanitizer import sanitize_text
from agent.observability.store import TraceStore
from agent.observability.token_counter import TokenCounter
from agent.observability.tracer import (
    TraceCollector,
    observe_span,
    record_tool_metric,
)
from agent.tools.base import BaseTool, ToolRegistry
from tests.test_app_service import build_service
from tests.test_workflow import ScriptedLLM


def flatten_span_names(span):
    names = [span.name]
    for child in span.children:
        names.extend(flatten_span_names(child))
    return names


def flatten_spans(span):
    spans = [span]
    for child in span.children:
        spans.extend(flatten_spans(child))
    return spans


def test_recursive_log_sanitization():
    sanitized = sanitize({
        "api_key": "secret-value",
        "nested": {
            "authorization": "Bearer abc",
            "prompt": "full source prompt",
            "total_tokens": 42,
        },
        "items": [{"password": "pw"}],
    })

    assert sanitized["api_key"] == "***REDACTED***"
    assert sanitized["nested"]["authorization"] == "***REDACTED***"
    assert sanitized["nested"]["prompt"] == "[18 chars]"
    assert sanitized["nested"]["total_tokens"] == 42
    assert sanitized["items"][0]["password"] == "***REDACTED***"
    rendered = sanitize_text("authorization=Bearer abc secret=hidden sk-1234567890")
    assert "abc" not in rendered
    assert "hidden" not in rendered
    assert "sk-1234567890" not in rendered


def test_token_counter_prefers_real_usage_metadata():
    counter = TokenCounter("openai", "gpt-test", ObservabilityConfig())
    response = AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        },
    )

    usage = counter.measure([HumanMessage(content="question")], response, 12.5)

    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.total_tokens == 18
    assert usage.estimated is False
    assert usage.cost is None


def test_token_counter_estimates_and_uses_explicit_rates():
    counter = TokenCounter(
        "custom",
        "unknown-model",
        ObservabilityConfig(
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        ),
    )

    usage = counter.measure(
        [HumanMessage(content="estimate this input")],
        AIMessage(content="estimated output"),
        5.0,
    )

    assert usage.estimated is True
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.cost == (
        usage.input_tokens / 1_000_000
        + usage.output_tokens * 2 / 1_000_000
    )


def test_trace_hierarchy_persistence_and_metrics(tmp_path: Path):
    collector = TraceCollector("session-1")
    with activate_trace(collector, "session-1"):
        with observe_span("parent"):
            with observe_span("child"):
                record_tool_metric("read_file", "success", 4.5)
    collector.finish()

    store = TraceStore(tmp_path / "traces")
    saved = store.save(collector)
    loaded = store.get_for_session("session-1")
    metrics = store.get_metrics()

    assert loaded is not None
    assert loaded.trace_id == saved.trace_id
    assert flatten_span_names(loaded.root) == ["workflow.session", "parent", "child"]
    assert loaded.root.status == "ok"
    assert metrics.trace_count == 1
    assert metrics.session_count == 1
    assert metrics.tool_calls == 1
    assert metrics.tool_successes == 1
    assert metrics.tools["read_file"].calls == 1


def test_metrics_filter_by_session_project_and_include_legacy_in_all(tmp_path: Path):
    store = TraceStore(tmp_path / "traces")
    specs = [
        ("session-1", "project-a", 10),
        ("session-2", "project-a", 20),
        ("session-3", "project-b", 30),
        ("session-1", None, 1),
    ]
    for session_id, project_id, tokens in specs:
        collector = TraceCollector(session_id, project_id=project_id)
        collector.record_token_usage(TokenUsage(
            input_tokens=tokens,
            output_tokens=0,
            total_tokens=tokens,
            model="test",
            provider="test",
        ))
        with activate_trace(collector, session_id):
            record_tool_metric("read_file", "success", 1.0)
        collector.finish()
        store.save(collector)

    session_metrics = store.get_metrics(session_id="session-1")
    project_metrics = store.get_metrics(project_id="project-a")
    all_metrics = store.get_metrics()

    assert session_metrics.trace_count == 2
    assert session_metrics.total_tokens == 11
    assert project_metrics.trace_count == 2
    assert project_metrics.total_tokens == 30
    assert all_metrics.trace_count == 4
    assert all_metrics.total_tokens == 61
    assert any(trace.project_id is None for trace in store.list_traces())

    legacy_payload = store.list_traces()[-1].model_dump(exclude={"project_id"})
    assert TraceTree.model_validate(legacy_payload).project_id is None


class StatusTool(BaseTool):
    description = "status tool"
    parameters_schema = {"type": "object", "properties": {}}

    def __init__(self, repo_root: Path, name: str, status: ToolStatus):
        super().__init__(repo_root)
        self.name = name
        self._status = status

    def execute(self, tool_call_id: str = "", **kwargs):
        return ToolResult(
            tool_call_id=tool_call_id,
            name=self.name,
            status=self._status,
            output=self._status.value,
        )


def test_tool_registry_records_failure_and_timeout_status(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(StatusTool(tmp_path, "slow", ToolStatus.TIMEOUT))
    registry.register(StatusTool(tmp_path, "broken", ToolStatus.EXECUTION_ERROR))
    collector = TraceCollector("session-tools")

    with activate_trace(collector, "session-tools"):
        registry.execute("slow", "call-1")
        registry.execute("broken", "call-2")
    collector.finish()
    tree = collector.to_tree()

    assert [metric.status for metric in tree.tool_metrics] == ["timeout", "execution_error"]
    statuses = {
        span.name: span.status
        for span in flatten_spans(tree.root)
        if span.name.startswith("tool.")
    }
    assert statuses == {"tool.slow": "timeout", "tool.broken": "error"}


class UsageScriptedLLM(ScriptedLLM):
    def invoke(self, messages):
        response = super().invoke(messages)
        return AIMessage(
            content=response.content,
            tool_calls=response.tool_calls,
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )


def test_app_service_keeps_trace_across_interrupt_and_resume(tmp_path: Path):
    observable_llm = ObservableChatModel(
        UsageScriptedLLM(),
        LLMConfig(provider="ollama"),
    )
    service, _, _ = build_service(tmp_path, observable_llm)
    session_id = service.create_session()

    interrupted = service.submit(session_id, "请修复 calculateTotal，并运行测试")
    interrupted_trace = service.get_trace(session_id)

    assert interrupted.status == "interrupted"
    assert interrupted_trace is not None
    assert interrupted_trace.root.status == "interrupted"
    assert any(
        span.name == "coder.approval" and span.status == "interrupted"
        for span in flatten_spans(interrupted_trace.root)
    )
    trace_id = interrupted_trace.trace_id
    assert any(event.event_type == "token_usage" for event in interrupted.events)

    completed = service.resume(session_id, ApprovalDecision(approved=True))
    completed_trace = service.get_trace(session_id)
    metrics = service.get_metrics()

    assert completed.status == "completed"
    assert completed_trace is not None
    assert completed_trace.trace_id == trace_id
    assert completed_trace.root.status == "ok"
    assert completed_trace.root.attributes["resume_count"] == 1
    span_names = flatten_span_names(completed_trace.root)
    assert "supervisor.route" in span_names
    assert "researcher.retrieve" in span_names
    assert "coder.agent" in span_names
    assert "tester.run" in span_names
    assert "verifier.review" in span_names
    assert completed_trace.total_tokens > 0
    assert completed_trace.tool_metrics
    assert metrics.llm_calls > 0
    assert metrics.tool_calls > 0
    assert metrics.total_tokens == completed_trace.total_tokens
    service.close()


def test_interrupted_trace_resumes_after_process_restart(tmp_path: Path):
    first_llm = ObservableChatModel(
        UsageScriptedLLM(),
        LLMConfig(provider="ollama"),
    )
    first_service, _, _ = build_service(tmp_path, first_llm)
    session_id = first_service.create_session("restart")
    interrupted = first_service.submit(session_id, "请修复 calculateTotal，并运行测试")
    first_trace = first_service.get_trace(session_id)
    first_service.close()

    second_llm = ObservableChatModel(
        UsageScriptedLLM(),
        LLMConfig(provider="ollama"),
    )
    second_service, _, _ = build_service(tmp_path, second_llm)
    completed = second_service.resume(session_id, ApprovalDecision(approved=True))
    resumed_trace = second_service.get_trace(session_id)

    assert interrupted.status == "interrupted"
    assert completed.status == "completed"
    assert first_trace is not None
    assert resumed_trace is not None
    assert resumed_trace.trace_id == first_trace.trace_id
    assert resumed_trace.root.status == "ok"
    assert resumed_trace.root.attributes["resume_count"] == 1
    assert resumed_trace.total_tokens > first_trace.total_tokens
    second_service.close()
