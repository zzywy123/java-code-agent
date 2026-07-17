"""Isolated evaluation runner."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from agent.eval.assertions import evaluate_assertions
from agent.eval.llm_judge import LLMJudge
from agent.eval.report import EvalReport, EvalTaskResult, build_summary, write_report
from agent.eval.task_set import DEFAULT_TASKS, EvalTask
from agent.models import ApprovalDecision
from agent.runtime import AppRuntime, create_app_runtime


class EvalTaskRunner:
    """Run every evaluation task in an independent temporary Git repository."""

    def __init__(
        self,
        fixture_dir: Path,
        runtime_factory: Callable[[Path], AppRuntime] = create_app_runtime,
        judge: LLMJudge | None = None,
        keep_workspaces: bool = False,
    ) -> None:
        self._fixture = fixture_dir.resolve()
        self._runtime_factory = runtime_factory
        self._judge = judge or LLMJudge()
        self._keep = keep_workspaces

    def run_task(self, task: EvalTask) -> EvalTaskResult:
        workspace = Path(tempfile.mkdtemp(prefix=f"eval_{task.id}_"))
        runtime = None
        started = time.perf_counter()
        try:
            shutil.copytree(
                self._fixture,
                workspace,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("target", ".git", ".agent-index", ".checkpoints"),
            )
            _apply_setup(task.setup, workspace)
            _init_git(workspace)
            git_config_before = _hash_file(workspace / ".git/config")
            with _isolated_environment(workspace):
                runtime = self._runtime_factory(workspace)
                session_id = runtime.service.create_session(f"eval:{task.id}")
                submit = runtime.service.submit(session_id, task.query)
                while submit.needs_approval:
                    submit = runtime.service.resume(
                        session_id,
                        ApprovalDecision(approved=True, reason="evaluation auto-approval"),
                    )
                events = runtime.service.stream_events(session_id)
                trace = runtime.service.get_trace(session_id)

            diff = _git(workspace, "diff")
            modified_files = _modified_files(workspace)
            needs_build = any(
                name in task.assertions for name in ("compile", "tests")
            )
            compile_ok = _run_maven(workspace, "compile") if needs_build else None
            test_ok = _run_maven(workspace, "test") if "tests" in task.assertions else None
            assertions = evaluate_assertions(
                task,
                workspace=workspace,
                final_answer=submit.final_answer or "",
                compile_ok=compile_ok,
                test_ok=test_ok,
                modified_files=modified_files,
                events=events,
                git_config_before=git_config_before,
            )
            judge_score = (
                self._judge.judge(task.query, submit.final_answer or "", task.category)
                if task.llm_judge and self._judge.enabled
                else None
            )
            return EvalTaskResult(
                task_id=task.id,
                category=task.category,
                status=submit.status,
                final_answer=submit.final_answer,
                assertions=assertions,
                compile_ok=compile_ok,
                test_ok=test_ok,
                modified_files=modified_files,
                diff=diff,
                trace_id=trace.trace_id if trace else None,
                iterations=_count_spans(trace.root, "coder.agent") if trace else 0,
                tokens=trace.total_tokens if trace else 0,
                duration_s=time.perf_counter() - started,
                cost=trace.total_cost if trace else None,
                judge_score=judge_score,
                error=submit.error,
            )
        except Exception as exc:
            return EvalTaskResult(
                task_id=task.id,
                category=task.category,
                status="error",
                duration_s=time.perf_counter() - started,
                error=str(exc),
            )
        finally:
            if runtime is not None:
                runtime.service.close()
            if not self._keep:
                shutil.rmtree(workspace, onerror=_remove_readonly)


def run_evaluation(
    fixture_dir: Path,
    output_dir: Path,
    tasks: list[EvalTask] | None = None,
    run_count: int = 1,
) -> EvalReport:
    runner = EvalTaskRunner(fixture_dir)
    selected = tasks or DEFAULT_TASKS
    results = [runner.run_task(task) for _ in range(run_count) for task in selected]
    from agent.config import load_config

    llm_config, _, _ = load_config()
    report = EvalReport(
        model=llm_config.model,
        prompt_version="v0.2.0",
        code_version=_git_code_version(Path(__file__).resolve().parents[3]),
        temperature=llm_config.temperature,
        run_count=run_count,
        results=results,
        summary=build_summary(results),
    )
    write_report(report, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Java Coding Agent evaluations")
    parser.add_argument("--fixture", type=Path, default=Path("demo-repo"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--task", action="append", dest="task_ids")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    selected = (
        [task for task in DEFAULT_TASKS if task.id in set(args.task_ids)]
        if args.task_ids
        else DEFAULT_TASKS
    )
    report = run_evaluation(args.fixture, args.output, selected, max(1, args.runs))
    print(report.model_dump_json(indent=2))


def _apply_setup(setup: str | None, workspace: Path) -> None:
    if setup == "inject_calculate_total_bug":
        path = workspace / "src/main/java/com/example/order/OrderService.java"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(".map(OrderItem::getSubtotal)", ".map(OrderItem::getUnitPrice)", 1),
            encoding="utf-8",
        )
    elif setup == "remove_quantity_validation":
        path = workspace / "src/main/java/com/example/order/OrderItem.java"
        text = path.read_text(encoding="utf-8")
        start = text.index("        if (quantity <= 0) {")
        end = text.index("        this.quantity = quantity;", start)
        path.write_text(text[:start] + text[end:], encoding="utf-8")


def _init_git(workspace: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Agent Eval",
            "-c",
            "user.email=eval@example.invalid",
            "commit",
            "-qm",
            "eval baseline",
        ],
        cwd=workspace,
        check=True,
    )


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _run_maven(workspace: Path, goal: str) -> bool:
    wrapper = workspace / ("mvnw.cmd" if os.name == "nt" else "mvnw")
    env = os.environ.copy()
    env.pop("JAVA_HOME", None)
    result = subprocess.run(
        [str(wrapper), "-B", goal],
        cwd=workspace,
        env=env,
        capture_output=True,
        timeout=180,
    )
    return result.returncode == 0


@contextmanager
def _isolated_environment(workspace: Path) -> Iterator[None]:
    overrides = {
        "AGENT_REPO_ROOT": str(workspace),
        "MEMORY_CHECKPOINT_DIR": str(workspace / ".checkpoints"),
        "MEMORY_LONG_TERM_PERSIST_DIR": str(workspace / ".memory"),
        "RAG_INDEX_DIR": str(workspace / ".agent-index"),
        "RAG_ENABLE_VECTOR": "false",
        "MCP_ENABLED": "false",
        "OBSERVABILITY_TRACE_DIR": str(workspace / ".observability/traces"),
    }
    old = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _count_spans(span: Any, name: str) -> int:
    return int(span.name == name) + sum(_count_spans(child, name) for child in span.children)


def _git_code_version(path: Path) -> str:
    try:
        version = _git(path, "rev-parse", "--short", "HEAD").strip()
        dirty = bool(_git(path, "status", "--porcelain").strip())
        return f"{version}-dirty" if dirty else version
    except Exception:
        return "workspace"


def _modified_files(workspace: Path) -> list[str]:
    """Return tracked and untracked working-tree paths from porcelain status."""
    paths: set[str] = set()
    for line in _git(
        workspace,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        if path:
            paths.add(path.strip('"'))
    return sorted(paths)


def _remove_readonly(function: Callable[[str], Any], path: str, _error: Any) -> None:
    """Retry cleanup after clearing Windows Git object read-only flags."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


if __name__ == "__main__":
    main()
