"""Application service integration tests."""

from pathlib import Path

import pytest

from agent.app_service import AppService
from agent.config import AgentConfig, LLMConfig, MemoryConfig, RAGConfig, WorkflowConfig
from agent.models import ApprovalDecision
from agent.session import SessionManager
from agent.workflow import create_workflow
from tests.test_workflow import AlreadyFixedLLM, EmptySearch, ScriptedLLM, build_registry


def build_service(tmp_path: Path, llm=None) -> tuple[AppService, SessionManager, MemoryConfig]:
    llm = llm or ScriptedLLM()
    memory_config = MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
    )
    manager = SessionManager(memory_config, llm=llm)
    graph = create_workflow(
        llm=llm,
        llm_config=LLMConfig(provider="ollama"),
        agent_config=AgentConfig(),
        workflow_config=WorkflowConfig(max_rework=1),
        rag_config=RAGConfig(),
        tool_registry=build_registry(tmp_path),
        search_engine=EmptySearch(),
        session_manager=manager,
        repo_root=tmp_path,
    )
    return AppService(graph, manager, project_root=tmp_path), manager, memory_config


def test_submit_interrupt_and_resume_emit_real_events_without_replay(tmp_path):
    service, _, _ = build_service(tmp_path)
    session_id = service.create_session("coding")

    interrupted = service.submit(session_id, "请修复 calculateTotal，并运行测试")

    assert interrupted.status == "interrupted"
    assert interrupted.needs_approval is True
    assert interrupted.approval_data["type"] == "approval_request"
    assert [event.event_type for event in interrupted.events].count("tool_call") == 1
    assert [event.event_type for event in interrupted.events].count("approval_request") == 1

    completed = service.resume(session_id, ApprovalDecision(approved=True))
    event_types = [event.event_type for event in completed.events]

    assert completed.status == "completed"
    assert completed.needs_approval is False
    assert "tool_call" not in event_types
    assert "tool_result" in event_types
    assert "patch_applied" in event_types
    assert "test_result" in event_types
    assert "review_result" in event_types
    assert event_types[-1] == "done"
    assert len(completed.patches) == 1

    all_events = service.stream_events(session_id)
    assert [event.event_type for event in all_events].count("tool_call") == 1
    assert [event.event_type for event in all_events].count("approval_request") == 1
    state = service.get_session(session_id)
    assert state.needs_approval is False
    assert state.event_count == len(all_events)
    service.close()


def test_already_fixed_submit_completes_without_patch_but_runs_tests(tmp_path):
    service, _, _ = build_service(tmp_path, AlreadyFixedLLM())
    session_id = service.create_session()

    result = service.submit(
        session_id,
        "请修复 OrderService.calculateTotal 的 Bug，并运行测试",
    )

    event_types = [event.event_type for event in result.events]
    assert result.status == "completed"
    assert result.patches == []
    assert "approval_request" not in event_types
    assert "test_result" in event_types
    assert "review_result" in event_types
    assert event_types[-1] == "done"
    service.close()


def test_rejected_approval_emits_denied_result_without_patch(tmp_path):
    service, _, _ = build_service(tmp_path)
    session_id = service.create_session()
    interrupted = service.submit(session_id, "请修复 calculateTotal，并运行测试")

    result = service.resume(
        session_id,
        ApprovalDecision(approved=False, reason="暂不修改"),
    )

    denied_results = [
        event for event in result.events
        if event.event_type == "tool_result" and event.data.get("status") == "denied"
    ]
    assert interrupted.status == "interrupted"
    assert result.status == "completed"
    assert result.patches == []
    assert denied_results
    assert not any(event.event_type == "patch_applied" for event in result.events)
    service.close()


def test_event_log_and_session_list_survive_service_restart(tmp_path):
    service, _, memory_config = build_service(tmp_path, AlreadyFixedLLM())
    session_id = service.create_session("persistent")
    result = service.submit(session_id, "请修复 calculateTotal，并运行测试")
    event_count = len(result.events)
    service.close()

    llm = AlreadyFixedLLM()
    manager = SessionManager(memory_config, llm=llm)
    graph = create_workflow(
        llm=llm,
        llm_config=LLMConfig(provider="ollama"),
        agent_config=AgentConfig(),
        workflow_config=WorkflowConfig(max_rework=1),
        rag_config=RAGConfig(),
        tool_registry=build_registry(tmp_path),
        search_engine=EmptySearch(),
        session_manager=manager,
        repo_root=tmp_path,
    )
    reopened = AppService(graph, manager, project_root=tmp_path)

    summaries = reopened.list_sessions()
    assert any(
        summary.session_id == session_id
        and summary.name == "persistent"
        and summary.event_count == event_count
        for summary in summaries
    )
    assert len(reopened.stream_events(session_id)) == event_count
    reopened.close()


def test_resume_without_pending_approval_returns_error(tmp_path):
    service, _, _ = build_service(tmp_path, AlreadyFixedLLM())
    session_id = service.create_session()

    result = service.resume(session_id, True)

    assert result.status == "error"
    assert "没有待审批" in result.error
    assert result.events[-1].event_type == "error"
    service.close()


def test_delete_session_removes_events_traces_and_creates_replacement(tmp_path):
    service, manager, _ = build_service(tmp_path, AlreadyFixedLLM())
    session_id = service.create_session("delete me")
    result = service.submit(session_id, "请修复 calculateTotal，并运行测试")
    event_path = manager.get_storage_dir() / "events" / f"{session_id}.jsonl"

    assert result.status == "completed"
    assert event_path.exists()
    assert service.get_trace(session_id) is not None

    replacement = service.delete_session(session_id)

    assert replacement != session_id
    assert not event_path.exists()
    assert session_id not in service._events
    assert session_id not in service._correlations
    assert session_id not in service._active_traces
    assert all(
        trace.session_id != session_id
        for trace in service._trace_store.list_traces()
    )
    assert service.get_session(replacement).session_id == replacement
    assert all(item.session_id != session_id for item in service.list_sessions())
    service.close()


def test_delete_unknown_session_fails_without_creating_replacement(tmp_path):
    service, _, _ = build_service(tmp_path, AlreadyFixedLLM())

    with pytest.raises(KeyError, match="会话不存在"):
        service.delete_session("missing")

    assert service.list_sessions() == []
    service.close()


def test_submit_git_diff_returns_real_tool_output(tmp_path):
    service, _, _ = build_service(tmp_path, AlreadyFixedLLM())
    session_id = service.create_session("git read")

    result = service.submit(session_id, "git diff")

    assert result.status == "completed"
    assert result.final_answer.startswith("diff --git")
    assert "当前快照" not in result.final_answer
    state = service.get_session(session_id)
    assert state.messages[-1]["render_hint"] == "diff"
    service.close()


def test_metrics_can_filter_current_session_and_project(tmp_path):
    service, _, _ = build_service(tmp_path, AlreadyFixedLLM())
    first = service.create_session("first")
    second = service.create_session("second")
    service.submit(first, "git diff")
    service.submit(second, "git status")

    session_metrics = service.get_metrics(scope="session", session_id=first)
    project_metrics = service.get_metrics(scope="project")
    all_metrics = service.get_metrics(scope="all")

    assert session_metrics.trace_count == 1
    assert session_metrics.session_count == 1
    assert project_metrics.trace_count == 2
    assert project_metrics.session_count == 2
    assert all_metrics == project_metrics
    with pytest.raises(ValueError, match="session_id"):
        service.get_metrics(scope="session")
    with pytest.raises(ValueError, match="未知指标范围"):
        service.get_metrics(scope="invalid")
    service.close()
