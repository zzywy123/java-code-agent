"""Tests for BM25 index, vector store, and hybrid search.

Validates:
- BM25 tokenization (camelCase, underscore, dot splitting, stop words)
- BM25 search correctness
- Vector store add/query/delete
- Hybrid search RRF fusion
- Reranking
- Search result structure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.indexing.bm25_index import BM25Index, _split_camel_case, _tokenize_java_text
from agent.indexing.chunk_store import compute_chunk_id, compute_file_hash
from agent.models import CodeChunk, CodeSlice, SearchResult


# ============================================================
# Helpers
# ============================================================

def _make_slice(
    class_name: str = "Main",
    method_name: str = "main",
    content: str = "public static void main(String[] args) {}",
    file_path: str = "src/Main.java",
    symbol_signature: str = "com.example.Main.main",
) -> CodeSlice:
    return CodeSlice(
        module="test", package="com.example",
        class_name=class_name, method_name=method_name,
        file_path=file_path, start_line=1, end_line=10,
        content=content, symbol_signature=symbol_signature,
    )


def _make_chunk(
    class_name: str = "Main",
    method_name: str = "main",
    content: str = "public static void main(String[] args) {}",
    file_path: str = "src/Main.java",
    symbol_signature: str = "com.example.Main.main",
) -> CodeChunk:
    s = _make_slice(class_name, method_name, content, file_path, symbol_signature)
    return CodeChunk(
        chunk_id=compute_chunk_id(file_path, symbol_signature, content),
        slice=s,
        file_hash=compute_file_hash(content),
    )


# ============================================================
# BM25 Tokenizer Tests
# ============================================================

class TestTokenizeJavaText:
    """Java-specific tokenization."""

    def test_camel_case_splitting(self):
        tokens = _tokenize_java_text("calculateTotal")
        assert "calculate" in tokens
        assert "total" in tokens

    def test_pascal_case_splitting(self):
        tokens = _tokenize_java_text("OrderService")
        assert "order" in tokens
        assert "service" in tokens

    def test_underscore_splitting(self):
        tokens = _tokenize_java_text("my_method_name")
        assert "my" in tokens
        assert "method" in tokens
        assert "name" in tokens

    def test_dot_splitting(self):
        tokens = _tokenize_java_text("com.example.order")
        assert "com" in tokens
        assert "example" in tokens
        assert "order" in tokens

    def test_mixed_splitting(self):
        tokens = _tokenize_java_text("OrderService.calculateTotal")
        assert "order" in tokens
        assert "service" in tokens
        assert "calculate" in tokens
        assert "total" in tokens

    def test_stop_word_removal(self):
        tokens = _tokenize_java_text("public void return new this")
        # All Java stop words should be removed
        assert len(tokens) == 0

    def test_short_tokens_removed(self):
        tokens = _tokenize_java_text("a bb ccc dddd")
        # "a" is 1 char, should be removed
        assert "a" not in tokens
        assert "bb" in tokens

    def test_lowercase_normalization(self):
        tokens = _tokenize_java_text("MyClass")
        assert all(t == t.lower() for t in tokens)

    def test_acronym_handling(self):
        tokens = _tokenize_java_text("getHTTPResponse")
        # "get" is a Java stop word, so it's removed
        assert "http" in tokens
        assert "response" in tokens

    def test_empty_input(self):
        assert _tokenize_java_text("") == []
        assert _tokenize_java_text("   ") == []


class TestSplitCamelCase:
    """camelCase splitting unit tests."""

    def test_simple_camel(self):
        assert _split_camel_case("calculateTotal") == ["calculate", "total"]

    def test_pascal(self):
        assert _split_camel_case("OrderService") == ["order", "service"]

    def test_all_upper(self):
        parts = _split_camel_case("HTTP")
        assert "http" in [p.lower() for p in parts]

    def test_single_word(self):
        assert _split_camel_case("order") == ["order"]

    def test_multi_camel(self):
        parts = _split_camel_case("getOrderByCustomerId")
        assert "get" in [p.lower() for p in parts]
        assert "order" in [p.lower() for p in parts]


# ============================================================
# BM25 Index Tests
# ============================================================

class TestBM25Index:
    """BM25 search functionality."""

    def test_add_and_count(self):
        index = BM25Index()
        chunks = [_make_chunk(content="calculate total amount")]
        index.add(chunks)
        assert index.count() == 1

    def test_search_relevant_result(self):
        index = BM25Index()
        c1 = _make_chunk(
            class_name="OrderService", method_name="calculateTotal",
            content="calculate total amount for order items",
        )
        c2 = _make_chunk(
            class_name="UserService", method_name="findUser",
            content="find user by id from database",
            file_path="src/UserService.java",
            symbol_signature="UserService.findUser",
        )
        index.add([c1, c2])

        results = index.search("calculate total", top_k=5)
        assert len(results) > 0
        # calculateTotal should be ranked first (higher BM25 score)
        assert results[0].chunk.slice.method_name == "calculateTotal"

    def test_search_returns_search_results(self):
        index = BM25Index()
        index.add([_make_chunk(content="test content")])
        results = index.search("test content", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].source == "bm25"

    def test_search_empty_index(self):
        index = BM25Index()
        results = index.search("anything")
        assert results == []

    def test_remove_by_file(self):
        index = BM25Index()
        c1 = _make_chunk(file_path="src/A.java", content="a", symbol_signature="A")
        c2 = _make_chunk(file_path="src/B.java", content="b", symbol_signature="B")
        index.add([c1, c2])

        removed = index.remove_by_file("src/A.java")
        assert removed == 1
        assert index.count() == 1

    def test_clear(self):
        index = BM25Index()
        index.add([_make_chunk()])
        index.clear()
        assert index.count() == 0


# ============================================================
# Hybrid Search RRF Tests
# ============================================================

class TestRRFFusion:
    """Reciprocal Rank Fusion algorithm."""

    def test_rrf_merges_results(self):
        """RRF should merge results from both sources."""
        from agent.config import RAGConfig
        from agent.indexing.hybrid_search import HybridSearchEngine

        config = RAGConfig(rrf_k=60, fusion_top_k=10, rerank_enabled=False)
        engine = HybridSearchEngine(config, MagicMock(), MagicMock())

        c1 = _make_chunk(class_name="A", method_name="foo", content="foo content",
                         file_path="src/A.java", symbol_signature="A.foo")
        c2 = _make_chunk(class_name="B", method_name="bar", content="bar content",
                         file_path="src/B.java", symbol_signature="B.bar")

        vector_results = [
            SearchResult(chunk=c1, score=0.9, source="vector", rank=1),
            SearchResult(chunk=c2, score=0.7, source="vector", rank=2),
        ]
        bm25_results = [
            SearchResult(chunk=c2, score=5.0, source="bm25", rank=1),
            SearchResult(chunk=c1, score=3.0, source="bm25", rank=2),
        ]

        fused = engine._fusion_rrf(vector_results, bm25_results)

        # Both chunks should appear
        chunk_ids = {r.chunk.chunk_id for r in fused}
        assert c1.chunk_id in chunk_ids
        assert c2.chunk_id in chunk_ids

        # All should be marked as hybrid
        assert all(r.source == "hybrid" for r in fused)

    def test_rrf_ranking_formula(self):
        """Verify weighted RRF score = weight/(k+rank) for each list."""
        from agent.config import RAGConfig
        from agent.indexing.hybrid_search import HybridSearchEngine

        config = RAGConfig(rrf_k=60, fusion_top_k=10, rerank_enabled=False)
        engine = HybridSearchEngine(config, MagicMock(), MagicMock())

        c = _make_chunk()
        vector = [SearchResult(chunk=c, score=0.9, source="vector", rank=1)]
        bm25 = [SearchResult(chunk=c, score=5.0, source="bm25", rank=1)]

        fused = engine._fusion_rrf(vector, bm25)
        assert len(fused) == 1
        # vector_weight=1.0, bm25_weight=2.0
        expected_score = 1.0 / (60 + 1) + 2.0 / (60 + 1)  # rank 1 in both
        assert abs(fused[0].score - expected_score) < 1e-6

    def test_rrf_respects_fusion_top_k(self):
        from agent.config import RAGConfig
        from agent.indexing.hybrid_search import HybridSearchEngine

        config = RAGConfig(rrf_k=60, fusion_top_k=2, rerank_enabled=False)
        engine = HybridSearchEngine(config, MagicMock(), MagicMock())

        chunks = [
            _make_chunk(class_name=f"C{i}", method_name=f"m{i}",
                        content=f"content {i}", file_path=f"src/C{i}.java",
                        symbol_signature=f"C{i}.m{i}")
            for i in range(5)
        ]
        vector = [SearchResult(chunk=c, score=0.9 - i * 0.1, source="vector", rank=i + 1)
                  for i, c in enumerate(chunks)]

        fused = engine._fusion_rrf(vector, [])
        assert len(fused) <= 2


# ============================================================
# Vector Store Tests (mocked ChromaDB)
# ============================================================

class TestVectorStoreMocked:
    """Vector store tests with mocked ChromaDB."""

    def test_add_calls_collection(self):
        from agent.indexing.vector_store import VectorStore
        from agent.config import RAGConfig

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 512

        store = VectorStore(RAGConfig(), mock_embedding)
        # Manually set up the mock collection
        store._collection = MagicMock()
        store._collection.count.return_value = 0
        store._initialized = True

        chunk = _make_chunk()
        chunk.embedding = [0.1] * 512
        store.add([chunk])

        store._collection.upsert.assert_called_once()

    def test_query_returns_search_results(self):
        from agent.indexing.vector_store import VectorStore
        from agent.config import RAGConfig

        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1] * 512

        store = VectorStore(RAGConfig(), mock_embedding)
        store._collection = MagicMock()
        store._collection.count.return_value = 1
        store._initialized = True

        store._collection.query.return_value = {
            "ids": [["chunk-1"]],
            "distances": [[0.3]],
            "metadatas": [[{
                "file_path": "src/Main.java",
                "class_name": "Main",
                "method_name": "main",
                "symbol_signature": "Main.main",
                "package": "com.example",
                "start_line": 1,
                "end_line": 10,
                "file_hash": "abc123",
            }]],
            "documents": [["public static void main()"]],
        }

        results = store.query("main method", top_k=1)
        assert len(results) == 1
        assert results[0].source == "vector"
        assert results[0].chunk.slice.method_name == "main"
