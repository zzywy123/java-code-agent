"""Chat history and approval interactions."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from agent.models import ApprovalDecision


def render_chat(service: Any, session_id: str) -> None:
    state = service.get_session(session_id)
    displayed_answers: set[str] = set()
    for message in state.messages:
        message_type = message.get("type")
        content = str(message.get("content") or "").strip()
        if not content or message_type not in {"human", "ai"}:
            continue
        role = "user" if message_type == "human" else "assistant"
        with st.chat_message(role):
            _render_content(content, message.get("render_hint"))
        if role == "assistant":
            displayed_answers.add(content)

    if state.final_answer and state.final_answer not in displayed_answers:
        with st.chat_message("assistant"):
            st.markdown(state.final_answer)

    if state.needs_approval:
        approval = state.approval_data or {}
        st.markdown(
            '<div class="approval-band"><strong>等待审批</strong><br>'
            f'{escape(str(approval.get("summary", "代码修改请求")))}</div>',
            unsafe_allow_html=True,
        )
        for tool_call in approval.get("tool_calls", []):
            with st.expander(str(tool_call.get("name", "tool")), expanded=False):
                st.json(tool_call.get("arguments", {}))
        approve_col, reject_col, _ = st.columns([1, 1, 3])
        if approve_col.button("批准", type="primary", use_container_width=True):
            with st.spinner("继续执行"):
                service.resume(session_id, ApprovalDecision(approved=True))
            st.rerun()
        if reject_col.button("拒绝", use_container_width=True):
            with st.spinner("记录拒绝"):
                service.resume(
                    session_id,
                    ApprovalDecision(approved=False, reason="用户在操作台拒绝"),
                )
            st.rerun()

    query = st.chat_input("输入代码问题或工程任务", disabled=state.needs_approval)
    if query:
        with st.spinner("Agent 正在处理"):
            service.submit(session_id, query)
        st.rerun()


def _render_content(content: str, render_hint: Any) -> None:
    if render_hint == "diff":
        st.code(content, language="diff")
    elif render_hint == "text":
        st.code(content, language="text")
    else:
        st.markdown(content)
