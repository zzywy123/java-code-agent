"""Evaluation result models and report rendering."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class AssertionResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class EvalTaskResult(BaseModel):
    task_id: str
    category: str
    status: str
    final_answer: str | None = None
    assertions: list[AssertionResult] = Field(default_factory=list)
    compile_ok: bool | None = None
    test_ok: bool | None = None
    modified_files: list[str] = Field(default_factory=list)
    diff: str = ""
    trace_id: str | None = None
    iterations: int = 0
    tokens: int = 0
    duration_s: float = 0.0
    cost: float | None = None
    judge_score: float | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "completed" and all(item.passed for item in self.assertions)


class EvalSummary(BaseModel):
    citation_correctness: float = 0.0
    patch_success_rate: float = 0.0
    compile_pass_rate: float = 0.0
    test_pass_rate: float = 0.0
    task_completion_rate: float = 0.0
    avg_iterations: float = 0.0
    total_tokens: int = 0
    total_duration_s: float = 0.0
    avg_cost_per_task: float | None = None


class EvalReport(BaseModel):
    model: str
    prompt_version: str
    code_version: str
    temperature: float
    run_count: int
    results: list[EvalTaskResult] = Field(default_factory=list)
    summary: EvalSummary = Field(default_factory=EvalSummary)


def build_summary(results: list[EvalTaskResult]) -> EvalSummary:
    if not results:
        return EvalSummary()
    citation_tasks = [
        result for result in results
        if any(item.name == "citations" for item in result.assertions)
    ]
    patch_tasks = [
        result for result in results
        if result.category in {"single_fix", "cross_file", "failure_repair", "verifier_rework"}
    ]
    compile_tasks = [result for result in results if result.compile_ok is not None]
    test_tasks = [result for result in results if result.test_ok is not None]
    known_costs = [result.cost for result in results if result.cost is not None]
    return EvalSummary(
        citation_correctness=_ratio(
            sum(_assertion_passed(item, "citations") for item in citation_tasks),
            len(citation_tasks),
        ),
        patch_success_rate=_ratio(sum(item.passed for item in patch_tasks), len(patch_tasks)),
        compile_pass_rate=_ratio(sum(bool(item.compile_ok) for item in compile_tasks), len(compile_tasks)),
        test_pass_rate=_ratio(sum(bool(item.test_ok) for item in test_tasks), len(test_tasks)),
        task_completion_rate=_ratio(sum(item.passed for item in results), len(results)),
        avg_iterations=sum(item.iterations for item in results) / len(results),
        total_tokens=sum(item.tokens for item in results),
        total_duration_s=sum(item.duration_s for item in results),
        avg_cost_per_task=(
            sum(known_costs) / len(known_costs)
            if len(known_costs) == len(results) and known_costs
            else None
        ),
    )


def write_report(report: EvalReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation-report.json"
    markdown_path = output_dir / "evaluation-report.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: EvalReport) -> str:
    summary = report.summary
    lines = [
        "# Evaluation Report",
        "",
        f"- Model: `{report.model}`",
        f"- Prompt version: `{report.prompt_version}`",
        f"- Code version: `{report.code_version}`",
        f"- Runs per task: `{report.run_count}`",
        "",
        "## Summary",
        "",
        f"- Task completion: {summary.task_completion_rate:.1%}",
        f"- Citation correctness: {summary.citation_correctness:.1%}",
        f"- Compile pass: {summary.compile_pass_rate:.1%}",
        f"- Test pass: {summary.test_pass_rate:.1%}",
        f"- Total tokens: {summary.total_tokens}",
        f"- Total duration: {summary.total_duration_s:.2f}s",
        f"- Average cost: {summary.avg_cost_per_task if summary.avg_cost_per_task is not None else 'N/A'}",
        "",
        "## Tasks",
        "",
        "| Task | Status | Passed | Tokens | Duration |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in report.results:
        lines.append(
            f"| {result.task_id} | {result.status} | "
            f"{'yes' if result.passed else 'no'} | {result.tokens} | {result.duration_s:.2f}s |"
        )
    return "\n".join(lines) + "\n"


def _assertion_passed(result: EvalTaskResult, name: str) -> bool:
    return any(item.name == name and item.passed for item in result.assertions)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
