"""Persistent sessions and memory context for the coding workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.config import MemoryConfig
from agent.memory.long_term import LongTermMemory
from agent.memory.short_term import ShortTermMemory
from agent.memory.summary import SummaryMemory

logger = logging.getLogger(__name__)

DECISION_CAPTURE_PROMPT = """\
你是项目长期记忆提取器。判断一次已完成工作流是否形成了以后仍应遵守的稳定信息。

只允许保存：
- preference：用户明确表达的长期偏好
- convention：项目长期约定、编码规范或协作规则
- decision：已确认的架构/技术决策及关键理由

禁止保存：当前代码内容、文件位置、Bug 状态、测试结果、临时任务过程、未经批准的建议，
以及能从仓库重新读取的事实。没有值得长期保存的信息时 save 必须为 false。

只输出 JSON：
{"save": true/false, "type": "preference|convention|decision", "content": "简洁且自包含的内容"}
"""


class SessionManager:
    """Own one shared SQLite checkpointer and isolated conversation memories."""

    def __init__(self, config: MemoryConfig, llm=None) -> None:
        checkpoint_dir = Path(config.checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir
        self._metadata_path = checkpoint_dir / "sessions.json"
        self._connection = sqlite3.connect(
            checkpoint_dir / "checkpoints.sqlite",
            check_same_thread=False,
        )
        self._checkpointer = SqliteSaver(self._connection)
        self._checkpointer.setup()
        self._sessions = self._load_sessions()
        self._active_session_id: str | None = self._sessions.pop("_active", None)
        self._short_term: dict[str, ShortTermMemory] = {}
        self._summaries: dict[str, SummaryMemory] = {}
        self._seen_messages: dict[str, set[str]] = {}
        self._llm = llm
        self._config = config
        self.long_term = LongTermMemory(Path(config.long_term_persist_dir).resolve())

    def create_session(self, name: str = "") -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {"name": name or "Coding session"}
        self._active_session_id = session_id
        self._save_sessions()
        return session_id

    def get_or_create_active_session(self) -> str:
        if self._active_session_id in self._sessions:
            return self._active_session_id  # type: ignore[return-value]
        return self.create_session()

    def list_sessions(self) -> list[dict[str, str]]:
        return [
            {"session_id": session_id, "name": data.get("name", "")}
            for session_id, data in self._sessions.items()
        ]

    def delete_session(self, session_id: str) -> None:
        """Delete one Session and all of its checkpoint-backed state."""
        if session_id not in self._sessions:
            raise KeyError(f"会话不存在: {session_id}")
        self._checkpointer.delete_thread(session_id)
        self._sessions.pop(session_id)
        self._short_term.pop(session_id, None)
        self._summaries.pop(session_id, None)
        self._seen_messages.pop(session_id, None)
        if self._active_session_id == session_id:
            self._active_session_id = next(iter(self._sessions), None)
        self._save_sessions()

    def get_thread_config(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise KeyError(f"会话不存在: {session_id}")
        return {"configurable": {"thread_id": session_id}}

    def get_checkpointer(self) -> BaseCheckpointSaver:
        return self._checkpointer

    def get_storage_dir(self) -> Path:
        """Return the persistent directory shared by session-level services."""
        return self._checkpoint_dir

    def remember_project_fact(self, key: str, memory_type: str, content: str) -> None:
        """Persist one explicit reusable project preference, convention or decision."""
        key = key.strip()
        content = content.strip()
        if not key or not content:
            raise ValueError("记忆 key 和 content 不能为空")
        if memory_type not in {"preference", "convention", "decision"}:
            raise ValueError("记忆类型必须是 preference、convention 或 decision")
        self.long_term.store(key, {"type": memory_type, "content": content})

    def capture_workflow_decision(
        self,
        task: str,
        final_answer: str,
        *,
        approved: bool = True,
    ) -> dict[str, object] | None:
        """Extract and persist one durable decision from a successful workflow."""
        if (
            not self._config.auto_capture_decisions
            or self._llm is None
            or not approved
            or not final_answer.strip()
            or not self._looks_like_durable_decision(task, final_answer)
        ):
            return None

        limit = self._config.auto_capture_max_chars
        prompt = (
            f"任务：\n{task[:limit]}\n\n"
            f"最终结果：\n{final_answer[:limit]}"
        )
        try:
            response = self._llm.invoke([
                SystemMessage(content=DECISION_CAPTURE_PROMPT),
                HumanMessage(content=prompt),
            ])
            raw = str(response.content or "")
            start = raw.find("{")
            end = raw.rfind("}")
            if start < 0 or end < start:
                return None
            data = json.loads(raw[start:end + 1])
        except Exception as exc:
            logger.warning("Automatic decision capture failed: %s", exc)
            return None

        if data.get("save") is not True:
            return None
        memory_type = str(data.get("type") or "")
        content = " ".join(str(data.get("content") or "").split())
        if memory_type not in {"preference", "convention", "decision"} or len(content) < 8:
            return None

        digest = hashlib.sha256(
            f"{memory_type}:{content.casefold()}".encode("utf-8")
        ).hexdigest()[:16]
        key = f"workflow-{digest}"
        if self.long_term.recall(key) is not None:
            return {
                "saved": False,
                "key": key,
                "type": memory_type,
                "content": content,
            }

        self.long_term.store(key, {
            "type": memory_type,
            "content": content,
            "source": "workflow",
        })
        return {
            "saved": True,
            "key": key,
            "type": memory_type,
            "content": content,
        }

    @staticmethod
    def _looks_like_durable_decision(task: str, final_answer: str) -> bool:
        """Cheaply reject ordinary Q&A, bug facts and test-only outcomes."""
        text = f"{task}\n{final_answer}".lower()
        markers = (
            "决定", "采用", "统一", "约定", "规范", "架构", "技术选型",
            "方案", "策略", "取舍", "迁移", "替换", "长期", "偏好",
            "继续使用", "仍使用",
            "decision", "adopt", "convention", "architecture", "strategy",
            "migrate", "replace", "prefer",
        )
        return any(marker in text for marker in markers)

    def build_context(
        self,
        session_id: str,
        messages: list[BaseMessage],
        task: str,
    ) -> list[BaseMessage]:
        """Return bounded conversation context plus validated project memories."""
        short = self._short_term.setdefault(
            session_id,
            ShortTermMemory(self._config.short_term_window),
        )
        summary = self._summaries.setdefault(
            session_id,
            SummaryMemory(
                llm=self._llm,
                trigger_count=self._config.summary_trigger,
                max_summary_tokens=self._config.max_summary_tokens,
            ),
        )
        seen = self._seen_messages.setdefault(session_id, set())
        for message in messages:
            key = f"{message.type}:{message.id}:{message.content}"
            if key in seen:
                continue
            seen.add(key)
            short.add(message)
            summary.add(message)

        context: list[BaseMessage] = []
        if summary.get_summary():
            context.append(SystemMessage(content=f"对话摘要：{summary.get_summary()}"))

        memories = []
        for entry in self.long_term.search(task, top_k=5):
            content = entry.get("content", {})
            if content.get("type") in {"preference", "convention", "decision"}:
                memories.append(str(content.get("content") or content.get("description") or ""))
        memories = [item for item in memories if item]
        if memories:
            context.append(SystemMessage(content="已确认的项目记忆：\n- " + "\n- ".join(memories)))

        # Keep the state's message order. A sliding-window implementation can
        # otherwise detach a ToolMessage from its preceding tool-call message.
        context.extend(self._safe_message_window(messages))
        return context

    def _safe_message_window(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Trim history without emitting orphaned OpenAI tool messages."""
        window = max(self._config.short_term_window, 20)
        candidates = list(messages[-window:])
        result: list[BaseMessage] = []
        index = 0

        while index < len(candidates):
            message = candidates[index]
            if isinstance(message, AIMessage) and message.tool_calls:
                expected = {
                    str(call.get("id"))
                    for call in message.tool_calls
                    if call.get("id")
                }
                following: list[ToolMessage] = []
                cursor = index + 1
                while cursor < len(candidates) and isinstance(candidates[cursor], ToolMessage):
                    following.append(candidates[cursor])
                    cursor += 1
                actual = {str(tool.tool_call_id) for tool in following}
                if expected and expected == actual:
                    result.append(message)
                    result.extend(following)
                    index = cursor
                    continue

                # The checkpoint was truncated or came from an older format.
                # Preserve the assistant text, but remove invalid tool_calls.
                result.append(AIMessage(content=str(message.content or "工具调用历史")))
                index += 1
                continue

            if isinstance(message, ToolMessage):
                # An orphan result cannot be sent with role=tool. Keep it as
                # ordinary context so the model can still understand the fact.
                result.append(SystemMessage(content=f"历史工具结果：{message.content}"))
            else:
                result.append(message)
            index += 1

        return result

    def close(self) -> None:
        self._connection.close()

    def _load_sessions(self) -> dict:
        if not self._metadata_path.exists():
            return {}
        try:
            return json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_sessions(self) -> None:
        data = dict(self._sessions)
        data["_active"] = self._active_session_id
        self._metadata_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
