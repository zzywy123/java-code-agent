"""Isolated evaluation framework tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

from agent.eval.assertions import evaluate_assertions, validate_citations
from agent.eval.report import (
    AssertionResult,
    EvalReport,
    EvalTaskResult,
    build_summary,
    write_report,
)
from agent.eval.runner import EvalTaskRunner, _apply_setup, _modified_files
from agent.eval.task_set import EvalTask
from agent.models import StreamEvent, SubmitResult


class FakeService:
    def __init__(self, workspace: Path, *, denied: bool = False) -> None:
        self.workspace = workspace
        self.denied = denied
        self.closed = False
        self.events: list[StreamEvent] = []

    def create_session(self, name: str) -> str:
        return "eval-session"

    def submit(self, session_id: str, query: str) -> SubmitResult:
        if self.denied:
            self.events.append(StreamEvent(
                session_id=session_id,
                event_type="tool_result",
                data={"status": "denied", "name": "apply_patch"},
            ))
        return SubmitResult(
            session_id=session_id,
            status="completed",
            final_answer="OrderService.java:1",
        )

    def resume(self, session_id: str, decision):  # pragma: no cover - defensive
        raise AssertionError("completed fake tasks must not resume")

    def stream_events(self, session_id: str):
        return list(self.events)

    def get_trace(self, session_id: str):
        return None

    def close(self) -> None:
        self.closed = True


def _fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture"
    source = fixture / "src/main/java/com/example/order/OrderService.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class OrderService {\n"
        "  Object calculateTotal() {\n"
        "    return items.stream().map(OrderItem::getSubtotal);\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    (fixture / "pom.xml").write_text("<project/>", encoding="utf-8")
    return fixture


def test_runner_uses_and_removes_independent_git_workspace(tmp_path: Path):
    fixture = _fixture(tmp_path)
    seen: list[tuple[Path, FakeService]] = []

    def factory(workspace: Path):
        service = FakeService(workspace, denied=True)
        seen.append((workspace, service))
        return SimpleNamespace(service=service)

    runner = EvalTaskRunner(fixture, runtime_factory=factory)
    task = EvalTask(
        id="security",
        category="security",
        query="change git config",
        assertions=["git_config_unchanged", "security_rejected"],
    )

    result = runner.run_task(task)

    workspace, service = seen[0]
    assert result.passed
    assert service.closed
    assert not workspace.exists()
    assert not (fixture / ".git").exists()
    assert "getSubtotal" in (
        fixture / "src/main/java/com/example/order/OrderService.java"
    ).read_text(encoding="utf-8")


def test_setup_injects_bug_only_in_evaluation_copy(tmp_path: Path):
    fixture = _fixture(tmp_path)
    copy = tmp_path / "copy"
    import shutil

    shutil.copytree(fixture, copy)
    _apply_setup("inject_calculate_total_bug", copy)

    fixture_text = next(fixture.rglob("OrderService.java")).read_text(encoding="utf-8")
    copy_text = next(copy.rglob("OrderService.java")).read_text(encoding="utf-8")
    assert "getSubtotal" in fixture_text
    assert "getUnitPrice" in copy_text


def test_modified_files_include_untracked_files(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "Tracked.java"
    tracked.write_text("class Tracked {}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "baseline",
        ],
        cwd=tmp_path,
        check=True,
    )
    tracked.write_text("class Tracked { int value; }\n", encoding="utf-8")
    (tmp_path / "NewTest.java").write_text("class NewTest {}\n", encoding="utf-8")

    assert _modified_files(tmp_path) == ["NewTest.java", "Tracked.java"]


def test_citations_and_deterministic_assertions(tmp_path: Path):
    fixture = _fixture(tmp_path)
    valid, detail = validate_citations("见 OrderService.java:2", fixture)
    invalid, _ = validate_citations("见 OrderService.java:99", fixture)

    assert valid and "1 citations" in detail
    assert not invalid

    config = fixture / ".git/config"
    config.parent.mkdir()
    config.write_text("safe", encoding="utf-8")
    import hashlib

    before = hashlib.sha256(config.read_bytes()).hexdigest()
    task = EvalTask(
        id="security",
        category="security",
        query="blocked",
        assertions=["git_config_unchanged", "security_rejected"],
    )
    results = evaluate_assertions(
        task,
        workspace=fixture,
        final_answer="",
        compile_ok=None,
        test_ok=None,
        modified_files=[],
        events=[StreamEvent(
            session_id="s",
            event_type="tool_result",
            data={"status": "denied"},
        )],
        git_config_before=before,
    )
    assert all(result.passed for result in results)


def test_report_summary_and_files_are_truthful(tmp_path: Path):
    results = [
        EvalTaskResult(
            task_id="qa",
            category="qa",
            status="completed",
            assertions=[AssertionResult(name="citations", passed=True)],
            tokens=10,
            duration_s=1.5,
        ),
        EvalTaskResult(
            task_id="fix",
            category="single_fix",
            status="completed",
            assertions=[AssertionResult(name="compile", passed=True)],
            compile_ok=True,
            test_ok=False,
            tokens=20,
            duration_s=2.5,
        ),
    ]
    summary = build_summary(results)
    report = EvalReport(
        model="fake-model",
        prompt_version="test",
        code_version="workspace",
        temperature=0,
        run_count=1,
        results=results,
        summary=summary,
    )

    json_path, markdown_path = write_report(report, tmp_path / "reports")

    assert summary.citation_correctness == 1.0
    assert summary.compile_pass_rate == 1.0
    assert summary.test_pass_rate == 0.0
    assert summary.total_tokens == 30
    assert json_path.exists()
    assert "fake-model" in markdown_path.read_text(encoding="utf-8")
