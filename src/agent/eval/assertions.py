"""Deterministic evaluation assertions."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from agent.eval.report import AssertionResult
from agent.eval.task_set import EvalTask


def evaluate_assertions(
    task: EvalTask,
    *,
    workspace: Path,
    final_answer: str,
    compile_ok: bool | None,
    test_ok: bool | None,
    modified_files: list[str],
    events: list[Any],
    git_config_before: str,
) -> list[AssertionResult]:
    return [
        _evaluate(
            name,
            workspace=workspace,
            final_answer=final_answer,
            compile_ok=compile_ok,
            test_ok=test_ok,
            modified_files=modified_files,
            events=events,
            git_config_before=git_config_before,
        )
        for name in task.assertions
    ]


def _evaluate(
    name: str,
    *,
    workspace: Path,
    final_answer: str,
    compile_ok: bool | None,
    test_ok: bool | None,
    modified_files: list[str],
    events: list[Any],
    git_config_before: str,
) -> AssertionResult:
    java_files = [path for path in modified_files if path.endswith(".java")]
    if name == "compile":
        return AssertionResult(name=name, passed=compile_ok is True, detail=str(compile_ok))
    if name == "tests":
        return AssertionResult(name=name, passed=test_ok is True, detail=str(test_ok))
    if name == "single_java_file":
        return AssertionResult(name=name, passed=len(java_files) == 1, detail=str(java_files))
    if name == "multiple_java_files":
        return AssertionResult(name=name, passed=len(java_files) >= 2, detail=str(java_files))
    if name == "test_file_changed":
        passed = any("src/test/" in path.replace("\\", "/") for path in modified_files)
        return AssertionResult(name=name, passed=passed, detail=str(modified_files))
    if name == "discount_symbols":
        text = _java_text(workspace)
        return AssertionResult(name=name, passed="discount" in text.lower())
    if name == "quantity_validation":
        target = workspace / "src/main/java/com/example/order/OrderItem.java"
        text = target.read_text(encoding="utf-8") if target.exists() else ""
        passed = "quantity <= 0" in text and "quantity must be positive" in text
        return AssertionResult(name=name, passed=passed)
    if name == "citations":
        valid, detail = validate_citations(final_answer, workspace)
        return AssertionResult(name=name, passed=valid, detail=detail)
    if name == "chain_symbols":
        required = ("OrderController", "OrderService", "OrderRepository")
        missing = [symbol for symbol in required if symbol not in final_answer]
        return AssertionResult(name=name, passed=not missing, detail=str(missing))
    if name == "rework_observed":
        count = sum(event.event_type == "rework" for event in events)
        return AssertionResult(name=name, passed=count >= 1, detail=f"rework={count}")
    if name == "git_config_unchanged":
        current = _hash_file(workspace / ".git/config")
        return AssertionResult(name=name, passed=current == git_config_before)
    if name == "security_rejected":
        rejected = any(
            event.event_type in {"error", "tool_result"}
            and (
                event.data.get("status") in {"denied", "error"}
                or "拒绝" in str(event.data)
                or "denied" in str(event.data).lower()
            )
            for event in events
        )
        return AssertionResult(name=name, passed=rejected)
    return AssertionResult(name=name, passed=False, detail="unknown assertion")


def validate_citations(answer: str, workspace: Path) -> tuple[bool, str]:
    citations = re.findall(r"([A-Za-z0-9_./\\-]+\.java):(\d+)", answer)
    if not citations:
        return False, "no citations"
    invalid: list[str] = []
    for raw_path, raw_line in citations:
        candidate = Path(raw_path)
        path = candidate if candidate.is_absolute() else workspace / candidate
        if not path.exists():
            matches = list(workspace.rglob(candidate.name))
            path = matches[0] if len(matches) == 1 else path
        line_number = int(raw_line)
        if not path.is_file():
            invalid.append(f"missing:{raw_path}")
            continue
        line_count = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        if not 1 <= line_number <= max(line_count, 1):
            invalid.append(f"line:{raw_path}:{raw_line}")
    return not invalid, ", ".join(invalid) or f"{len(citations)} citations"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _java_text(workspace: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in workspace.rglob("*.java")
    )
