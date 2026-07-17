"""Summary memory: compresses old conversation into a summary.

When messages exceed the trigger count, older messages are summarized
using LLM and replaced with a single summary message.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """\
请将以下对话历史压缩为一段简洁的摘要，保留关键信息：
- 用户的主要问题和需求
- Agent 的重要发现和结论
- 已完成的代码修改和测试结果
- 未解决的问题

摘要应该用中文撰写，不超过 500 字。
"""


class SummaryMemory:
    """Summary-based memory that compresses old messages.

    When messages exceed trigger_count, older messages are summarized
    and replaced with a single summary message.
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        trigger_count: int = 30,
        max_summary_tokens: int = 1000,
    ) -> None:
        self._llm = llm
        self._trigger_count = trigger_count
        self._max_summary_tokens = max_summary_tokens
        self._messages: list[BaseMessage] = []
        self._summary: str = ""

    def add(self, message: BaseMessage) -> None:
        """Add a message and trigger summarization if needed."""
        self._messages.append(message)
        if len(self._messages) > self._trigger_count:
            self._summarize()

    def add_many(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages."""
        self._messages.extend(messages)
        if len(self._messages) > self._trigger_count:
            self._summarize()

    def get_summary(self) -> str:
        """Get the current summary."""
        return self._summary

    def get_messages(self) -> list[BaseMessage]:
        """Get all messages (summary + recent)."""
        result: list[BaseMessage] = []
        if self._summary:
            result.append(SystemMessage(content=f"对话摘要：{self._summary}"))
        result.extend(self._messages)
        return result

    def get_context(self, max_tokens: int = 4000) -> list[BaseMessage]:
        """Get context that fits within a token budget."""
        result: list[BaseMessage] = []
        char_budget = max_tokens * 4
        used = 0

        # Summary first
        if self._summary:
            summary_msg = SystemMessage(content=f"对话摘要：{self._summary}")
            summary_chars = len(self._summary)
            if summary_chars <= char_budget:
                result.append(summary_msg)
                used += summary_chars

        # Then recent messages (newest first)
        for msg in reversed(self._messages):
            msg_chars = len(str(msg.content)) if msg.content else 0
            if used + msg_chars > char_budget:
                break
            result.insert(len(result) - (1 if self._summary else 0), msg)
            used += msg_chars

        return result

    def clear(self) -> None:
        """Clear all messages and summary."""
        self._messages.clear()
        self._summary = ""

    def count(self) -> int:
        """Return the number of messages (excluding summary)."""
        return len(self._messages)

    def _summarize(self) -> None:
        """Summarize older messages using LLM."""
        if not self._llm:
            # Without LLM, just keep recent messages
            self._messages = self._messages[-self._trigger_count:]
            return

        # Take the older half for summarization
        split_point = len(self._messages) // 2
        old_messages = self._messages[:split_point]
        recent_messages = self._messages[split_point:]

        # Build summarization prompt
        old_text = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content[:200]}"
            for m in old_messages
            if m.content
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=SUMMARY_PROMPT),
                HumanMessage(content=old_text),
            ])
            new_summary = response.content.strip()

            # Combine with existing summary
            if self._summary:
                self._summary = f"{self._summary}\n\n{new_summary}"
            else:
                self._summary = new_summary

            # Keep only recent messages
            self._messages = recent_messages
            logger.info("Summarized %d messages, kept %d", split_point, len(recent_messages))

        except Exception as e:
            logger.warning("Summarization failed: %s", e)
            # On failure, just trim
            self._messages = self._messages[-self._trigger_count:]
