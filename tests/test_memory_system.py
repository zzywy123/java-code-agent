"""Tests for Memory system: ShortTerm, Summary, LongTerm, ThreadManager.

Validates:
- Short-term sliding window
- Summary memory with LLM compression
- Long-term persistent storage
- Thread isolation and checkpointer
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.memory.long_term import LongTermMemory
from agent.memory.short_term import ShortTermMemory
from agent.memory.summary import SummaryMemory
from agent.memory.thread_manager import ThreadManager


class TestShortTermMemory:
    """Sliding window memory tests."""

    def test_add_and_count(self):
        mem = ShortTermMemory(window_size=10)
        mem.add(HumanMessage(content="hello"))
        assert mem.count() == 1

    def test_window_size_limit(self):
        mem = ShortTermMemory(window_size=3)
        for i in range(5):
            mem.add(HumanMessage(content=f"msg {i}"))
        assert mem.count() == 3

    def test_keeps_recent_messages(self):
        mem = ShortTermMemory(window_size=2)
        mem.add(HumanMessage(content="old"))
        mem.add(HumanMessage(content="new"))
        mem.add(HumanMessage(content="newer"))
        messages = mem.get_messages()
        contents = [m.content for m in messages]
        assert "newer" in contents
        assert "old" not in contents

    def test_preserves_system_messages(self):
        mem = ShortTermMemory(window_size=2)
        mem.add(SystemMessage(content="system"))
        mem.add(HumanMessage(content="msg1"))
        mem.add(HumanMessage(content="msg2"))
        messages = mem.get_messages()
        assert any(isinstance(m, SystemMessage) for m in messages)

    def test_get_context_respects_token_budget(self):
        mem = ShortTermMemory(window_size=100)
        for i in range(10):
            mem.add(HumanMessage(content=f"message {i} " * 100))
        context = mem.get_context(max_tokens=50)
        # Should return fewer messages than total
        assert len(context) < 10

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add(HumanMessage(content="test"))
        mem.clear()
        assert mem.count() == 0

    def test_add_many(self):
        mem = ShortTermMemory(window_size=10)
        messages = [HumanMessage(content=f"msg {i}") for i in range(5)]
        mem.add_many(messages)
        assert mem.count() == 5


class TestSummaryMemory:
    """Summary memory tests."""

    def test_add_without_llm(self):
        mem = SummaryMemory(llm=None, trigger_count=3)
        mem.add(HumanMessage(content="hello"))
        assert mem.count() == 1

    def test_summarize_without_llm_trims(self):
        mem = SummaryMemory(llm=None, trigger_count=2)
        for i in range(5):
            mem.add(HumanMessage(content=f"msg {i}"))
        # Without LLM, should trim to trigger_count
        assert mem.count() <= 2

    def test_summarize_with_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "这是对话摘要"
        mem = SummaryMemory(llm=mock_llm, trigger_count=2)

        for i in range(4):
            mem.add(HumanMessage(content=f"message {i}"))

        summary = mem.get_summary()
        assert "摘要" in summary or len(summary) > 0

    def test_get_messages_includes_summary(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "summary text"
        mem = SummaryMemory(llm=mock_llm, trigger_count=2)

        for i in range(4):
            mem.add(HumanMessage(content=f"msg {i}"))

        messages = mem.get_messages()
        assert any("摘要" in str(m.content) for m in messages if m.content)

    def test_clear(self):
        mem = SummaryMemory()
        mem.add(HumanMessage(content="test"))
        mem.clear()
        assert mem.count() == 0
        assert mem.get_summary() == ""


class TestLongTermMemory:
    """Long-term persistent memory tests."""

    def test_store_and_recall(self, tmp_path: Path):
        mem = LongTermMemory(tmp_path / "memory")
        mem.store("key1", {"type": "decision", "content": "use RRF fusion"})
        result = mem.recall("key1")
        assert result is not None
        assert result["content"] == "use RRF fusion"

    def test_recall_nonexistent(self, tmp_path: Path):
        mem = LongTermMemory(tmp_path / "memory")
        assert mem.recall("missing") is None

    def test_search(self, tmp_path: Path):
        mem = LongTermMemory(tmp_path / "memory")
        mem.store("bm25", {"description": "BM25 inverted index for keyword search"})
        mem.store("vector", {"description": "vector similarity search with ChromaDB"})
        mem.store("rrf", {"description": "reciprocal rank fusion algorithm"})

        results = mem.search("BM25 search")
        assert len(results) > 0

    def test_persistence(self, tmp_path: Path):
        dir_path = tmp_path / "memory"
        mem1 = LongTermMemory(dir_path)
        mem1.store("persist", {"data": "test"})

        # Create new instance — should load from disk
        mem2 = LongTermMemory(dir_path)
        result = mem2.recall("persist")
        assert result is not None

    def test_delete(self, tmp_path: Path):
        mem = LongTermMemory(tmp_path / "memory")
        mem.store("key", {"data": "value"})
        assert mem.delete("key") is True
        assert mem.recall("key") is None

    def test_count(self, tmp_path: Path):
        mem = LongTermMemory(tmp_path / "memory")
        mem.store("a", {"x": 1})
        mem.store("b", {"x": 2})
        assert mem.count() == 2

    def test_clear(self, tmp_path: Path):
        mem = LongTermMemory(tmp_path / "memory")
        mem.store("a", {"x": 1})
        mem.clear()
        assert mem.count() == 0


class TestThreadManager:
    """Thread management tests."""

    def test_create_thread(self, tmp_path: Path):
        tm = ThreadManager(checkpoint_dir=tmp_path / "threads")
        thread_id = tm.create_thread("Test Thread")
        assert thread_id
        assert tm.get_thread(thread_id)["name"] == "Test Thread"

    def test_list_threads(self, tmp_path: Path):
        tm = ThreadManager(checkpoint_dir=tmp_path / "threads")
        tm.create_thread("Thread 1")
        tm.create_thread("Thread 2")
        assert len(tm.list_threads()) == 2

    def test_delete_thread(self, tmp_path: Path):
        tm = ThreadManager(checkpoint_dir=tmp_path / "threads")
        tid = tm.create_thread()
        assert tm.delete_thread(tid) is True
        assert tm.get_thread(tid) is None

    def test_thread_persistence(self, tmp_path: Path):
        dir_path = tmp_path / "threads"
        tm1 = ThreadManager(checkpoint_dir=dir_path)
        tid = tm1.create_thread("Persistent")

        tm2 = ThreadManager(checkpoint_dir=dir_path)
        assert tm2.get_thread(tid) is not None

    def test_get_checkpointer(self, tmp_path: Path):
        tm = ThreadManager()
        cp = tm.get_checkpointer()
        assert cp is not None

    def test_get_thread_config(self, tmp_path: Path):
        tm = ThreadManager()
        tid = tm.create_thread()
        config = tm.get_thread_config(tid)
        assert config["configurable"]["thread_id"] == tid

    def test_increment_message_count(self, tmp_path: Path):
        tm = ThreadManager(checkpoint_dir=tmp_path / "threads")
        tid = tm.create_thread()
        tm.increment_message_count(tid, 5)
        assert tm.get_thread(tid)["message_count"] == 5
