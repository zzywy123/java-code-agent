"""Streamlit console boundary and rendering smoke tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest


UI_DIR = Path(__file__).parents[1] / "src" / "agent" / "ui"


def test_ui_respects_application_service_boundary():
    forbidden = (
        "agent.agent_graph",
        "agent.workflow",
        "agent.tools",
        "agent.rag",
        "agent.indexing",
    )
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in UI_DIR.glob("*.py")
    )

    assert "from agent.runtime import create_app_runtime" in sources
    assert all(name not in sources for name in forbidden)
    assert ".submit(" in sources
    assert ".resume(" in sources


def test_unsafe_html_paths_escape_dynamic_content():
    sidebar = (UI_DIR / "sidebar.py").read_text(encoding="utf-8")
    chat = (UI_DIR / "chat.py").read_text(encoding="utf-8")
    panels = (UI_DIR / "panels.py").read_text(encoding="utf-8")

    assert "escape(runtime.llm_config.model)" in sidebar
    assert "escape(str(approval.get(" in chat
    assert "escape(detail)" in panels


def test_session_delete_requires_explicit_confirmation():
    sidebar = (UI_DIR / "sidebar.py").read_text(encoding="utf-8")

    assert 'st.button("删除会话"' in sidebar
    assert '"确认删除"' in sidebar
    assert '"取消"' in sidebar
    assert "service.delete_session(selected)" in sidebar
    assert sidebar.index('st.button("删除会话"') < sidebar.index(
        "service.delete_session(selected)"
    )


def test_streamlit_app_starts_with_mocked_runtime(monkeypatch, tmp_path: Path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from agent.config import LLMConfig
    from agent.models import SessionState
    from agent import runtime as runtime_module

    class FakeService:
        def __init__(self):
            self.deleted_sessions = []
            self.metrics_requests = []

        def list_sessions(self):
            return []

        def get_session(self, session_id):
            return SessionState(
                session_id=session_id,
                messages=[{
                    "type": "ai",
                    "content": 'Git Diff:\n-exec "$JAVACMD"\n+exec "$JAVACMD"',
                    "render_hint": "diff",
                }],
            )

        def stream_events(self, session_id):
            return []

        def get_metrics(self, scope="all", session_id=None):
            self.metrics_requests.append((scope, session_id))
            return SimpleNamespace(
                total_tokens=0,
                tool_calls=0,
                trace_count=0,
                tool_failures=0,
            )

        def get_trace(self, session_id):
            return None

        def create_session(self, name):
            return "new-session"

        def delete_session(self, session_id):
            self.deleted_sessions.append(session_id)
            return "replacement-session"

    fake_service = FakeService()
    fake_runtime = SimpleNamespace(
        service=fake_service,
        session_id="session-1",
        repo_root=tmp_path,
        llm_config=LLMConfig(provider="ollama"),
        search_type="BM25",
        chunk_count=0,
    )
    monkeypatch.setattr(runtime_module, "create_app_runtime", lambda root: fake_runtime)

    app_path = UI_DIR / "app.py"
    monkeypatch.setenv("AGENT_REPO_ROOT", str(tmp_path))
    app = AppTest.from_file(str(app_path), default_timeout=10)

    app.run()

    assert not app.exception
    assert app.title[0].value == "Java Coding Agent"
    assert app.chat_input[0].placeholder == "输入代码问题或工程任务"
    assert '$JAVACMD' in app.code[0].value
    assert app.code[0].language == "diff"
    assert fake_service.metrics_requests[-1] == ("session", "session-1")
    scope_select = next(item for item in app.selectbox if item.label == "统计范围")
    assert scope_select.options == ["当前会话", "当前项目", "全部"]

    scope_select.select("project").run()
    assert fake_service.metrics_requests[-1] == ("project", "session-1")
    next(item for item in app.selectbox if item.label == "统计范围").select("all").run()
    assert fake_service.metrics_requests[-1] == ("all", "session-1")

    next(button for button in app.button if button.label == "删除会话").click().run()
    assert fake_service.deleted_sessions == []
    assert any(warning.value.startswith("将永久删除") for warning in app.warning)

    next(button for button in app.button if button.label == "确认删除").click().run()
    assert not app.exception
    assert fake_service.deleted_sessions == ["session-1"]
