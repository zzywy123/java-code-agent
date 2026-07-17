"""Embedding initialization and offline fallback tests."""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import pytest

from agent.config import EmbeddingConfig
from agent.indexing.embedding import EmbeddingService


class FakeSentenceTransformer:
    def get_embedding_dimension(self) -> int:
        return 512


def test_local_embedding_uses_cache_only_by_default(monkeypatch):
    calls: list[tuple[str, bool]] = []

    def factory(model_name: str, *, local_files_only: bool):
        calls.append((model_name, local_files_only))
        return FakeSentenceTransformer()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=factory),
    )
    service = EmbeddingService(EmbeddingConfig(local_files_only=True))

    service.initialize()

    assert calls == [("BAAI/bge-small-zh-v1.5", True)]


def test_concurrent_embedding_initialization_loads_model_once(monkeypatch):
    calls = 0
    call_lock = threading.Lock()

    def factory(model_name: str, *, local_files_only: bool):
        nonlocal calls
        with call_lock:
            calls += 1
        time.sleep(0.05)
        return FakeSentenceTransformer()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=factory),
    )
    service = EmbeddingService(EmbeddingConfig(local_files_only=True))
    errors: list[Exception] = []

    def initialize() -> None:
        try:
            service.initialize()
        except Exception as exc:  # pragma: no cover - assertion captures failures
            errors.append(exc)

    threads = [threading.Thread(target=initialize) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert calls == 1


def test_embedding_initialization_failure_is_not_retried(monkeypatch):
    calls = 0

    def factory(model_name: str, *, local_files_only: bool):
        nonlocal calls
        calls += 1
        raise OSError("model cache is incomplete")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=factory),
    )
    service = EmbeddingService(EmbeddingConfig(local_files_only=True))

    with pytest.raises(RuntimeError, match="model cache is incomplete"):
        service.initialize()
    with pytest.raises(RuntimeError, match="model cache is incomplete"):
        service.initialize()

    assert calls == 1


def test_online_embedding_sets_bounded_hugging_face_timeouts(monkeypatch):
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: FakeSentenceTransformer()),
    )
    service = EmbeddingService(EmbeddingConfig(
        local_files_only=False,
        hub_timeout_seconds=7,
    ))

    service.initialize()

    assert __import__("os").environ["HF_HUB_ETAG_TIMEOUT"] == "7"
    assert __import__("os").environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "7"
