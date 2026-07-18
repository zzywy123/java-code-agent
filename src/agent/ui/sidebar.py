"""Repository, Session and metric controls."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from agent.repository_import import RepositoryImportError, RepositoryImportService


def render_sidebar(
    runtime: Any,
    current_session_id: str,
    import_service: RepositoryImportService,
    owner_id: str,
) -> tuple[str, str | None]:
    service = runtime.service
    with st.sidebar:
        st.subheader("工作区")
        repo_action = _render_repository_import(
            runtime,
            import_service,
            owner_id,
        )

        st.markdown(
            f'<div class="repo-meta">{escape(runtime.llm_config.provider.value)} / '
            f'{escape(runtime.llm_config.model)}<br>{escape(runtime.search_type)} · '
            f'{runtime.chunk_count} chunks</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("会话")
        summaries = service.list_sessions()
        ids = [summary.session_id for summary in summaries]
        if current_session_id not in ids:
            ids.insert(0, current_session_id)
        selected = st.selectbox(
            "当前会话",
            ids,
            index=ids.index(current_session_id),
            format_func=lambda value: _session_label(value, summaries),
            label_visibility="collapsed",
        )
        if st.button("新建会话", width="stretch", icon=":material/add_comment:"):
            selected = service.create_session("Coding session")

        confirm_key = f"confirm_delete_session_{selected}"
        if st.button("删除会话", width="stretch", icon=":material/delete:"):
            st.session_state[confirm_key] = True
        if st.session_state.get(confirm_key):
            st.warning("将永久删除该会话的消息、事件、Checkpoint 和 Trace。")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button(
                "确认删除",
                type="primary",
                width="stretch",
                key=f"delete_session_{selected}",
            ):
                replacement = service.delete_session(selected)
                st.session_state.pop(confirm_key, None)
                st.session_state.session_id = replacement
                st.rerun()
            if cancel_col.button(
                "取消",
                width="stretch",
                key=f"cancel_delete_session_{selected}",
            ):
                st.session_state.pop(confirm_key, None)
                st.rerun()

        state = service.get_session(selected)
        events = service.stream_events(selected)
        active_agent = _active_agent(events)
        st.divider()
        st.subheader("Agent 状态")
        for agent in ("supervisor", "researcher", "coder", "tester", "verifier"):
            if state.needs_approval and agent == "coder":
                status = "waiting"
                label = "等待审批"
            elif active_agent == agent and not state.final_answer:
                status = "running"
                label = "运行中"
            else:
                status = ""
                label = "空闲"
            st.markdown(
                '<div class="agent-status">'
                f'<span class="status-dot {status}"></span>'
                f'<span>{escape(agent.title())}</span>'
                f'<span class="status-label">{escape(label)}</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("运行指标")
        metrics_scope = st.selectbox(
            "统计范围",
            ("session", "project", "all"),
            format_func={
                "session": "当前会话",
                "project": "当前项目",
                "all": "全部",
            }.get,
        )
        metrics = service.get_metrics(
            scope=metrics_scope,
            session_id=selected,
        )
        left, right = st.columns(2)
        left.metric("Token", f"{metrics.total_tokens:,}")
        right.metric("工具", metrics.tool_calls)
        left.metric("Trace", metrics.trace_count)
        right.metric("失败", metrics.tool_failures)

    return selected, repo_action


def _render_repository_import(
    runtime: Any,
    import_service: RepositoryImportService,
    owner_id: str,
) -> str | None:
    mode = st.segmented_control(
        "导入方式",
        ("本地文件夹", "Git 地址", "服务器路径"),
        default="本地文件夹",
        width="stretch",
    )
    try:
        if mode == "本地文件夹":
            uploaded_files = st.file_uploader(
                "选择本地仓库",
                accept_multiple_files="directory",
                max_upload_size=25,
                key="local_repository_upload",
            )
            if st.button(
                "导入本地仓库",
                width="stretch",
                icon=":material/folder_open:",
                disabled=not uploaded_files,
            ):
                with st.spinner("正在导入仓库"):
                    result = import_service.import_uploaded_directory(
                        uploaded_files,
                        owner_id=owner_id,
                    )
                st.toast("本地仓库已导入", icon=":material/check_circle:")
                return str(result.repo_root)

        elif mode == "Git 地址":
            git_url = st.text_input(
                "Git 仓库地址",
                placeholder="https://github.com/org/repository.git",
            )
            branch = st.text_input("分支", placeholder="main")
            if st.button(
                "克隆仓库",
                width="stretch",
                icon=":material/download:",
                disabled=not git_url.strip(),
            ):
                with st.spinner("正在克隆仓库"):
                    result = import_service.clone_repository(
                        git_url,
                        owner_id=owner_id,
                        branch=branch,
                    )
                st.toast("Git 仓库已克隆", icon=":material/check_circle:")
                return str(result.repo_root)

        else:
            repo_value = st.text_input(
                "服务器仓库路径",
                value=str(runtime.repo_root),
                key=f"server_repo_path_{hash(str(runtime.repo_root))}",
            )
            if st.button(
                "加载服务器仓库",
                width="stretch",
                icon=":material/folder_managed:",
            ):
                result = import_service.validate_server_repository(repo_value)
                return str(result.repo_root)
    except RepositoryImportError as exc:
        st.error(str(exc))
    except (OSError, ValueError) as exc:
        st.error(f"仓库导入失败：{exc}")
    return None


def _session_label(session_id: str, summaries: list[Any]) -> str:
    for summary in summaries:
        if summary.session_id == session_id:
            return f"{summary.name or 'Session'} · {session_id[:8]}"
    return f"Session · {session_id[:8]}"


def _active_agent(events: list[Any]) -> str:
    for event in reversed(events):
        if event.event_type == "agent_switch":
            return str(event.data.get("agent", ""))
    return "supervisor"
