"""Operational event, Diff, test, Token and Trace panels."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

import streamlit as st


EVENT_LABELS = {
    "agent_switch": "Agent 切换",
    "rag_retrieval": "代码检索",
    "tool_call": "工具调用",
    "tool_result": "工具结果",
    "approval_request": "审批请求",
    "patch_applied": "Patch 已应用",
    "test_result": "测试完成",
    "review_result": "Verifier 审查",
    "memory_saved": "长期记忆",
    "rework": "返工",
    "token_usage": "Token 统计",
    "error": "错误",
    "done": "完成",
}


def render_panels(service: Any, session_id: str) -> None:
    events = service.stream_events(session_id)
    trace = service.get_trace(session_id)
    timeline_tab, diff_tab, test_tab, token_tab, trace_tab = st.tabs(
        ["时间线", "Diff", "测试", "Token", "Trace"]
    )

    with timeline_tab:
        if not events:
            st.caption("暂无事件")
        for event in events[-40:]:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            detail = _event_detail(event)
            st.markdown(
                '<div class="event-line">'
                f'<div class="event-time">{timestamp}</div>'
                f'<div class="event-name">'
                f'{escape(EVENT_LABELS.get(event.event_type, event.event_type))}</div>'
                f'<div class="repo-meta">{escape(detail)}</div></div>',
                unsafe_allow_html=True,
            )

    with diff_tab:
        patches = [event.data for event in events if event.event_type == "patch_applied"]
        if not patches:
            st.caption("暂无代码变更")
        for patch in patches:
            st.markdown(f"**{patch.get('file_path', 'file')}**")
            st.code(str(patch.get("unified_diff", "")), language="diff")

    with test_tab:
        test_events = [event.data for event in events if event.event_type == "test_result"]
        if not test_events:
            st.caption("尚未运行测试")
        for result in test_events:
            status = "通过" if result.get("success") else "失败"
            cols = st.columns(3)
            cols[0].metric("状态", status)
            cols[1].metric("通过", result.get("tests_passed", 0))
            cols[2].metric("失败", result.get("tests_failed", 0))
            output = result.get("stdout") or result.get("stderr")
            if output:
                st.code(str(output)[-8000:], language="text")

    with token_tab:
        token_events = [event.data for event in events if event.event_type == "token_usage"]
        total_in = sum(int(item.get("input_tokens", 0)) for item in token_events)
        total_out = sum(int(item.get("output_tokens", 0)) for item in token_events)
        cols = st.columns(3)
        cols[0].metric("输入", f"{total_in:,}")
        cols[1].metric("输出", f"{total_out:,}")
        cols[2].metric("总计", f"{total_in + total_out:,}")
        if token_events:
            st.dataframe(token_events, use_container_width=True, hide_index=True)
        else:
            st.caption("暂无 Token 数据")

    with trace_tab:
        if trace is None:
            st.caption("暂无 Trace")
        else:
            st.json(trace.model_dump(mode="json"), expanded=False)


def _event_detail(event: Any) -> str:
    data = event.data
    if event.event_type == "agent_switch":
        return str(data.get("agent", ""))
    if event.event_type in {"tool_call", "tool_result"}:
        return str(data.get("name", ""))
    if event.event_type == "rag_retrieval":
        return f"{len(data.get('results', []))} results"
    if event.event_type == "test_result":
        return "通过" if data.get("success") else "失败"
    if event.event_type == "review_result":
        return str(data.get("summary", ""))
    if event.event_type == "memory_saved":
        return str(data.get("content", ""))
    if event.event_type == "error":
        return str(data.get("message", ""))
    return ""
