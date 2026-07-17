"""Rich terminal client backed exclusively by the shared AppService."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from agent.app_service import AppService
from agent.models import (
    ApprovalDecision,
    ReviewArtifact,
    SearchArtifact,
    TestResultArtifact,
    ToolStatus,
)
from agent.observability.logger import configure_logging
from agent.runtime import create_app_runtime

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
console = Console(force_terminal=True)


def display_welcome() -> None:
    console.print(Panel(
        "[bold]Java Coding Agent[/bold]\n"
        "[dim]代码检索、审批修改、自动测试、Verifier 审查与会话恢复[/dim]",
        border_style="cyan",
    ))


def display_approval_request(approval_data: dict[str, Any]) -> bool:
    summary = str(approval_data.get("summary") or "代码修改请求")
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("工具")
    table.add_column("参数")
    for call in approval_data.get("tool_calls", []):
        table.add_row(str(call.get("name", "")), str(call.get("arguments", {})))
    console.print(Panel(table, title=summary, border_style="yellow"))
    for diff in approval_data.get("diffs", []):
        console.print(Panel(str(diff), title="Diff", border_style="yellow"))
    return Confirm.ask("批准执行", default=False)


def display_tool_result(name: str, status: ToolStatus, output: str) -> None:
    colors = {
        ToolStatus.SUCCESS: "green",
        ToolStatus.DENIED: "yellow",
        ToolStatus.TIMEOUT: "red",
        ToolStatus.NOT_FOUND: "yellow",
        ToolStatus.INVALID_ARGUMENT: "red",
        ToolStatus.EXECUTION_ERROR: "red",
        ToolStatus.ERROR: "red",
    }
    console.print(Panel(output, title=f"{name}: {status.value}", border_style=colors[status]))


def display_final_answer(answer: str | None, error: str | None, patches: list[Any]) -> None:
    if error:
        console.print(Panel(error, title="执行失败", border_style="red"))
        return
    if answer:
        console.print(Panel(Markdown(answer), title="回答", border_style="green"))
    if patches:
        files = "\n".join(f"- {Path(patch.file_path).as_posix()}" for patch in patches)
        console.print(Panel(files, title=f"已应用 {len(patches)} 个 Patch", border_style="cyan"))


def display_stream_events(events: list[Any]) -> None:
    """Render AppService events without reimplementing workflow behavior."""
    for event in events:
        data = event.data
        if event.event_type == "agent_switch":
            console.print(f"[cyan]Agent -> {data.get('agent', data.get('route', 'unknown'))}[/cyan]")
        elif event.event_type == "rag_retrieval":
            artifact = SearchArtifact.model_validate(data)
            console.print(f"[dim]RAG 检索到 {len(artifact.results)} 条证据[/dim]")
        elif event.event_type == "tool_call":
            console.print(f"[blue]工具调用: {data.get('name', '')} {data.get('arguments', {})}[/blue]")
        elif event.event_type == "tool_result":
            try:
                status = ToolStatus(str(data.get("status") or "success"))
            except ValueError:
                status = ToolStatus.ERROR
            display_tool_result(
                str(data.get("name") or "tool"),
                status,
                str(data.get("content") or data.get("output") or ""),
            )
        elif event.event_type == "patch_applied":
            console.print(f"[green]Patch 已应用: {data.get('file_path', '')}[/green]")
        elif event.event_type == "test_result":
            result = TestResultArtifact.model_validate(data)
            display_tool_result(
                "run_tests",
                ToolStatus.SUCCESS if result.success else ToolStatus.EXECUTION_ERROR,
                result.stdout or result.stderr,
            )
        elif event.event_type == "review_result":
            review = ReviewArtifact.model_validate(data)
            detail = review.summary
            if review.issues:
                detail += "\n" + "\n".join(f"- {issue}" for issue in review.issues)
            console.print(Panel(
                detail,
                title="Verifier",
                border_style="green" if review.approved else "red",
            ))
        elif event.event_type == "memory_saved":
            console.print(
                f"[cyan]已沉淀项目记忆 ({data.get('type', 'decision')}): "
                f"{data.get('content', '')}[/cyan]"
            )
        elif event.event_type == "rework":
            console.print(f"[yellow]进入第 {data.get('rework_count', 0)} 次返工[/yellow]")
        elif event.event_type == "error":
            console.print(Panel(str(data.get("message") or "未知错误"), title="错误", border_style="red"))


def run_app_service_with_approval(
    service: AppService,
    session_id: str,
    query: str,
):
    """Drive one request through completion and any Patch approval interrupt."""
    result = service.submit(session_id, query)
    while True:
        display_stream_events(result.events)
        if not result.needs_approval:
            display_final_answer(result.final_answer, result.error, result.patches)
            return result
        approved = display_approval_request(result.approval_data or {})
        result = service.resume(
            session_id,
            ApprovalDecision(
                approved=approved,
                reason="" if approved else "用户拒绝操作",
            ),
        )


def main() -> None:
    configure_logging()
    display_welcome()
    console.print("[dim]正在初始化运行时和代码索引...[/dim]")
    try:
        runtime = create_app_runtime()
    except Exception as exc:
        console.print(f"[red]运行时初始化失败: {exc}[/red]")
        sys.exit(1)

    service = runtime.service
    session_id = runtime.session_id
    console.print(f"[dim]仓库: {runtime.repo_root}[/dim]")
    console.print(
        f"[dim]模型: {runtime.llm_config.provider.value} / {runtime.llm_config.model}; "
        f"工具: {runtime.tool_count}; 索引: {runtime.chunk_count} chunks / {runtime.search_type} / "
        f"{runtime.index_duration_seconds:.1f}s[/dim]"
    )
    console.print(f"[dim]Session: {session_id}[/dim]")

    try:
        while True:
            try:
                user_input = console.input("\n[bold green]You> [/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                break
            if user_input.lower() in {"new", "新会话", "新任务"}:
                session_id = service.create_session()
                console.print(f"[dim]已创建新 Session: {session_id}[/dim]")
                continue
            if user_input.startswith("/remember "):
                parts = user_input.split(maxsplit=3)
                if len(parts) != 4:
                    console.print("[yellow]格式: /remember <preference|convention|decision> <key> <content>[/yellow]")
                    continue
                try:
                    service.remember_project_fact(parts[2], parts[1], parts[3])
                    console.print(f"[green]已保存项目记忆: {parts[2]}[/green]")
                except ValueError as exc:
                    console.print(f"[yellow]{exc}[/yellow]")
                continue
            run_app_service_with_approval(service, session_id, user_input)
    finally:
        service.close()
        console.print("\n[dim]再见！[/dim]")


if __name__ == "__main__":
    main()
