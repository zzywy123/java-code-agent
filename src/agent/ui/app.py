"""Streamlit entry point for the Java Coding Agent console."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import streamlit as st

from agent.runtime import create_app_runtime
from agent.repository_import import RepositoryImportService
from agent.ui.chat import render_chat
from agent.ui.panels import render_panels
from agent.ui.sidebar import render_sidebar
from agent.ui.styles import APP_CSS


def main() -> None:
    st.set_page_config(
        page_title="Java Coding Agent",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    owner_id = _get_or_create_owner_id()
    runtime = _get_or_create_runtime()
    session_id = st.session_state.get("session_id", runtime.session_id)
    selected_session, repo_action = render_sidebar(
        runtime,
        session_id,
        _get_or_create_import_service(),
        owner_id,
    )
    st.session_state.session_id = selected_session

    if repo_action and Path(repo_action) != runtime.repo_root:
        with st.spinner("正在加载仓库和索引"):
            runtime.service.close()
            st.session_state.runtime = create_app_runtime(
                repo_action,
                storage_namespace=_storage_namespace_for(repo_action, owner_id),
            )
            st.session_state.session_id = st.session_state.runtime.session_id
        st.rerun()

    title_col, status_col = st.columns([4, 1])
    title_col.title("Java Coding Agent")
    state = runtime.service.get_session(st.session_state.session_id)
    status_col.metric("状态", "等待审批" if state.needs_approval else "就绪")

    chat_col, panel_col = st.columns([1.35, 1], gap="large")
    with chat_col:
        render_chat(runtime.service, st.session_state.session_id)
    with panel_col:
        render_panels(runtime.service, st.session_state.session_id)


def _get_or_create_runtime():
    if "runtime" not in st.session_state:
        default_repo = os.environ.get("AGENT_REPO_ROOT", "./demo-repo")
        with st.spinner("正在初始化 Agent"):
            # Keep the original storage location so existing sessions remain visible.
            st.session_state.runtime = create_app_runtime(default_repo)
            st.session_state.session_id = st.session_state.runtime.session_id
    return st.session_state.runtime


def _get_or_create_owner_id() -> str:
    if "workspace_owner_id" not in st.session_state:
        st.session_state.workspace_owner_id = uuid.uuid4().hex
    return str(st.session_state.workspace_owner_id)


def _storage_namespace_for(repo_path: Path | str, owner_id: str) -> str | None:
    default_repo = Path(
        os.environ.get("AGENT_REPO_ROOT", "./demo-repo")
    ).expanduser().resolve()
    return None if Path(repo_path).expanduser().resolve() == default_repo else owner_id


def _get_or_create_import_service() -> RepositoryImportService:
    if "repository_import_service" not in st.session_state:
        st.session_state.repository_import_service = (
            RepositoryImportService.from_environment()
        )
    return st.session_state.repository_import_service


if __name__ == "__main__":
    main()
