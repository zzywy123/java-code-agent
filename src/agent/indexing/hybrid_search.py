"""Hybrid search engine combining vector search and BM25.

Fusion algorithm: Reciprocal Rank Fusion (RRF).
    score = 1 / (k + rank)
    where k=60 (configurable), rank is the 1-based position in each result list.

Reranking: Uses cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers
CrossEncoder for precise reranking of the fused candidates.
Falls back to score-based ordering if cross-encoder is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.config import RAGConfig
from agent.indexing.bm25_index import BM25Index
from agent.indexing.vector_store import VectorStore
from agent.models import CodeChunk, SearchResult

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search combining vector similarity and BM25 keyword matching.

    Pipeline:
    1. Vector search (ChromaDB cosine similarity)
    2. BM25 search (keyword matching with Java tokenizer)
    3. Reciprocal Rank Fusion (RRF)
    4. Reranking (cross-encoder)
    """

    def __init__(
        self,
        config: RAGConfig,
        vector_store: VectorStore | None,
        bm25_index: BM25Index,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._reranker: Any = None
        self._reranker_initialized = False
        self._vector_failed = False

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Execute hybrid search.

        Args:
            query: The search query
            top_k: Number of final results (default from config)

        Returns:
            List of SearchResult objects sorted by combined relevance
        """
        if top_k is None:
            top_k = self._config.rerank_top_k

        # Step 1: Vector search
        vector_results: list[SearchResult] = []
        if self._vector_store is not None and not self._vector_failed:
            try:
                vector_results = self._vector_store.query(
                    query_text=query,
                    top_k=self._config.vector_top_k,
                )
            except Exception as exc:
                self._vector_failed = True
                logger.warning("Vector search unavailable; using BM25 only: %s", exc)

        # Step 2: BM25 search
        bm25_results = self._bm25_index.search(
            query=query,
            top_k=self._config.bm25_top_k,
        )

        # Step 3: Fusion
        fused = self._fusion_rrf(vector_results, bm25_results)

        # Step 4: Rerank (optional)
        if self._config.rerank_enabled:
            fused = self._rerank(fused, query)

        # Truncate to top_k
        results = fused[:top_k]

        # Update ranks
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def _fusion_rrf(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> list[SearchResult]:
        """Merge results using weighted Reciprocal Rank Fusion.

        RRF score = sum over all lists: weight / (k + rank_i)
        BM25 weight = 2.0 (keyword matching is more precise for code search)
        Vector weight = 1.0 (semantic similarity is a supplement)
        """
        k = self._config.rrf_k
        bm25_weight = 2.0
        vector_weight = 1.0

        # chunk_id → merged score
        scores: dict[str, float] = {}
        # chunk_id → SearchResult (keep the one with more info)
        chunk_map: dict[str, SearchResult] = {}

        for r in vector_results:
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + vector_weight / (k + r.rank)
            if cid not in chunk_map:
                chunk_map[cid] = r

        for r in bm25_results:
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + bm25_weight / (k + r.rank)
            if cid not in chunk_map:
                chunk_map[cid] = r

        # Build fused results
        fused: list[SearchResult] = []
        for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            original = chunk_map[cid]
            fused.append(SearchResult(
                chunk=original.chunk,
                score=score,
                source="hybrid",
                rank=0,  # Will be set after reranking
            ))

        # Trim to fusion_top_k
        return fused[:self._config.fusion_top_k]

    def _rerank(
        self,
        results: list[SearchResult],
        query: str,
    ) -> list[SearchResult]:
        """Rerank results using a cross-encoder model.

        Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for precise reranking.
        Falls back to score-based ordering if the model is unavailable.
        """
        if not results:
            return results

        reranker = self._get_reranker()
        if reranker is None:
            # Fallback: just return in current order
            return results

        # Build query-document pairs
        pairs = []
        for r in results:
            # Combine method name, class, and content for reranking
            doc_text = str(
                f"{r.chunk.slice.class_name}.{r.chunk.slice.method_name} "
                f"{r.chunk.slice.content[:500]}"
            )
            pairs.append([str(query), doc_text])

        try:
            # Ensure all pairs are string tuples (some CrossEncoders require this)
            clean_pairs = [(str(a), str(b)) for a, b in pairs]
            scores = reranker.predict(clean_pairs)
            # Assign reranker scores
            for i, score in enumerate(scores):
                results[i].score = float(score)
            # Sort by reranker score
            results.sort(key=lambda r: r.score, reverse=True)
        except Exception as e:
            logger.warning("Reranking failed, using fusion order: %s", e)

        return results

    def _get_reranker(self) -> Any:
        """Lazy-load the cross-encoder reranker model."""
        if self._reranker_initialized:
            return self._reranker

        self._reranker_initialized = True
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", self._config.rerank_model)
            self._reranker = CrossEncoder(self._config.rerank_model)
            logger.info("Reranker loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not available, reranking disabled"
            )
            self._reranker = None
        except Exception as e:
            logger.warning("Failed to load reranker: %s, reranking disabled", e)
            self._reranker = None

        return self._reranker

    def stats(self) -> dict[str, Any]:
        """Return search engine statistics."""
        return {
            "vector_count": self._vector_store.count() if self._vector_store is not None else 0,
            "bm25_count": self._bm25_index.count(),
            "rerank_enabled": self._config.rerank_enabled,
            "rerank_model": self._config.rerank_model,
        }
