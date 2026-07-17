"""Embedding service for code chunks.

Supports two providers:
1. LOCAL: sentence-transformers with BAAI/bge-small-zh-v1.5 (default)
2. OPENAI: OpenAI text-embedding-3-small

The local provider requires sentence-transformers to be installed.
The OpenAI provider requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from agent.config import EmbeddingConfig, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding service that converts text to vector representations.

    Supports local (sentence-transformers) and OpenAI providers.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: Any = None
        self._openai_client: Any = None
        self._initialized = False
        self._initialization_error: RuntimeError | None = None
        self._initialization_lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize exactly once so concurrent queries cannot duplicate model loading."""
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the embedding model."""
        if self._initialized:
            return
        if self._initialization_error is not None:
            raise self._initialization_error

        with self._initialization_lock:
            if self._initialized:
                return
            if self._initialization_error is not None:
                raise self._initialization_error
            try:
                if self._config.provider == EmbeddingProvider.LOCAL:
                    self._init_local()
                elif self._config.provider == EmbeddingProvider.OPENAI:
                    self._init_openai()
                else:
                    raise ValueError(f"Unknown embedding provider: {self._config.provider}")
            except Exception as exc:
                error = (
                    exc if isinstance(exc, RuntimeError)
                    else RuntimeError(f"Failed to initialize embedding service: {exc}")
                )
                self._initialization_error = error
                raise error from exc
            self._initialized = True

    def _init_local(self) -> None:
        """Initialize sentence-transformers model."""
        try:
            if not self._config.local_files_only:
                timeout = str(self._config.hub_timeout_seconds)
                os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", timeout)
                os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", timeout)
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._config.model_name)
            self._model = SentenceTransformer(
                self._config.model_name,
                local_files_only=self._config.local_files_only,
            )
            # bge-small-zh-v1.5 outputs 512-dim vectors
            try:
                actual_dim = self._model.get_embedding_dimension()
            except AttributeError:
                actual_dim = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding model loaded, dimension=%d", actual_dim)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def _init_openai(self) -> None:
        """Initialize OpenAI embedding client."""
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI embedding client initialized: %s", self._config.openai_model)
        except ImportError:
            raise RuntimeError("openai package not installed")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            return []

        self._ensure_initialized()

        if self._config.provider == EmbeddingProvider.LOCAL:
            return self._embed_local(texts)
        elif self._config.provider == EmbeddingProvider.OPENAI:
            return self._embed_openai(texts)
        else:
            raise ValueError(f"Unknown provider: {self._config.provider}")

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        For retrieval, queries may be embedded differently than documents
        (e.g., with a query prefix for bge models).
        """
        if not query:
            return []

        self._ensure_initialized()

        if self._config.provider == EmbeddingProvider.LOCAL:
            # bge models benefit from a query prefix
            prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
            try:
                result = self._model.encode(
                    [prefixed],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception:
                # Fallback: encode without prefix
                result = self._model.encode(
                    [query],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return result[0].tolist()
        elif self._config.provider == EmbeddingProvider.OPENAI:
            return self._embed_openai([query])[0]
        else:
            raise ValueError(f"Unknown provider: {self._config.provider}")

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._config.dimension

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embed using sentence-transformers."""
        batch_size = self._config.batch_size
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            all_embeddings.extend(e.tolist() for e in embeddings)

        return all_embeddings

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed using OpenAI API."""
        batch_size = self._config.batch_size
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._openai_client.embeddings.create(
                model=self._config.openai_model,
                input=batch,
            )
            for item in response.data:
                all_embeddings.append(item.embedding)

        return all_embeddings
