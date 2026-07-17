"""Tests for Agentic RAG: QueryRewriter, EvidenceJudge, AgenticRAG.

Validates:
- Query rewriting (with and without LLM)
- Evidence judgment (rule-based and LLM-based)
- Multi-round retrieval loop
- Degradation behavior
- Refinement query generation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.config import RAGConfig
from agent.indexing.chunk_store import compute_chunk_id, compute_file_hash
from agent.models import CodeChunk, CodeSlice, RAGResult, SearchResult
from agent.rag.agentic_rag import AgenticRAG
from agent.rag.evidence_judge import EvidenceJudge
from agent.rag.query_rewriter import QueryRewriter


# ============================================================
# Helpers
# ============================================================

def _make_search_result(
    class_name: str = "Main",
    method_name: str = "main",
    content: str = "test content",
    file_path: str = "src/Main.java",
    score: float = 0.8,
) -> SearchResult:
    s = CodeSlice(
        module="test", package="com.example",
        class_name=class_name, method_name=method_name,
        file_path=file_path, start_line=1, end_line=10,
        content=content, symbol_signature=f"{class_name}.{method_name}",
    )
    chunk = CodeChunk(
        chunk_id=compute_chunk_id(file_path, f"{class_name}.{method_name}", content),
        slice=s, file_hash=compute_file_hash(content),
    )
    return SearchResult(chunk=chunk, score=score, source="hybrid")


# ============================================================
# QueryRewriter Tests
# ============================================================

class TestQueryRewriter:
    """Query rewriting functionality."""

    def test_returns_original_without_llm(self):
        rewriter = QueryRewriter(llm=None)
        result = rewriter.rewrite("OrderService calculateTotal bug")
        assert result == ["OrderService calculateTotal bug"]

    def test_returns_original_for_empty_query(self):
        rewriter = QueryRewriter(llm=None)
        assert rewriter.rewrite("") == [""]

    def test_llm_rewrite_returns_list(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = '["query1", "query2"]'
        rewriter = QueryRewriter(llm=mock_llm)
        result = rewriter.rewrite("test query")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_llm_rewrite_includes_original(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = '["sub1", "sub2"]'
        rewriter = QueryRewriter(llm=mock_llm)
        result = rewriter.rewrite("original query")
        assert "original query" in result

    def test_llm_rewrite_handles_invalid_json(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "not json at all"
        rewriter = QueryRewriter(llm=mock_llm)
        result = rewriter.rewrite("test query")
        assert result == ["test query"]

    def test_llm_rewrite_limits_count(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = '["a", "b", "c", "d", "e"]'
        rewriter = QueryRewriter(llm=mock_llm)
        result = rewriter.rewrite("test", max_queries=3)
        assert len(result) <= 3

    def test_llm_failure_falls_back(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        rewriter = QueryRewriter(llm=mock_llm)
        result = rewriter.rewrite("test query")
        assert result == ["test query"]


# ============================================================
# EvidenceJudge Tests
# ============================================================

class TestEvidenceJudge:
    """Evidence sufficiency judgment."""

    def test_empty_results_insufficient(self):
        judge = EvidenceJudge(threshold=0.6)
        sufficient, conf, reason = judge.judge("test", [])
        assert not sufficient
        assert conf == 0.0

    def test_matching_results_sufficient(self):
        judge = EvidenceJudge(threshold=0.3)
        results = [_make_search_result(
            class_name="OrderService",
            method_name="calculateTotal",
            content="calculate total amount for order items",
        )]
        sufficient, conf, reason = judge.judge("OrderService calculateTotal", results)
        assert sufficient

    def test_non_matching_results_insufficient(self):
        judge = EvidenceJudge(threshold=0.8)
        results = [_make_search_result(
            class_name="UserService",
            method_name="findUser",
            content="find user by id",
        )]
        sufficient, conf, reason = judge.judge("OrderService calculateTotal bug", results)
        assert not sufficient

    def test_llm_judge_used_when_uncertain(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = '{"sufficient": true, "confidence": 0.85, "reason": "good match"}'
        judge = EvidenceJudge(llm=mock_llm, threshold=0.6)
        # Use a query that partially matches to get uncertain rule-based confidence
        results = [_make_search_result(
            class_name="OrderService",
            method_name="payOrder",
            content="process payment for order",
        )]
        sufficient, conf, reason = judge.judge("OrderService calculateTotal bug", results)
        # LLM should be called since rule-based is uncertain
        mock_llm.invoke.assert_called_once()
        assert sufficient
        assert conf == 0.85

    def test_llm_judge_falls_back_on_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        judge = EvidenceJudge(llm=mock_llm, threshold=0.3)
        results = [_make_search_result(
            class_name="OrderService",
            method_name="calculateTotal",
            content="calculate total amount",
        )]
        sufficient, conf, reason = judge.judge("OrderService calculateTotal", results)
        # Should fall back to rule-based
        assert isinstance(sufficient, bool)


# ============================================================
# AgenticRAG Tests
# ============================================================

class TestAgenticRAG:
    """Agentic RAG retrieval loop."""

    def test_single_round_sufficient(self):
        config = RAGConfig(max_retrieval_rounds=3, rerank_top_k=5)
        mock_search = MagicMock()
        mock_search.search.return_value = [_make_search_result()]

        rewriter = QueryRewriter(llm=None)
        judge = EvidenceJudge(threshold=0.1)  # Low threshold → always sufficient
        rag = AgenticRAG(config, mock_search, rewriter, judge)

        result = rag.retrieve("test query")
        assert result.evidence_sufficient
        assert result.rounds_used == 1
        assert not result.degraded

    def test_multi_round_retrieval(self):
        config = RAGConfig(max_retrieval_rounds=3, rerank_top_k=5)
        mock_search = MagicMock()
        # First call returns poor results, second returns good results
        good_result = _make_search_result(
            class_name="OrderService", method_name="calculateTotal",
            content="calculate total amount for order",
        )
        mock_search.search.side_effect = [
            [_make_search_result(class_name="Other", content="unrelated")],
            [good_result],
        ]

        rewriter = QueryRewriter(llm=None)
        # First call insufficient, second call sufficient
        judge = MagicMock()
        judge.judge.side_effect = [
            (False, 0.3, "not enough"),
            (True, 0.8, "good match"),
        ]
        rag = AgenticRAG(config, mock_search, rewriter, judge)

        result = rag.retrieve("OrderService calculateTotal")
        assert result.evidence_sufficient
        assert result.rounds_used == 2

    def test_degrades_after_max_rounds(self):
        config = RAGConfig(max_retrieval_rounds=2, rerank_top_k=5)
        mock_search = MagicMock()
        mock_search.search.return_value = [_make_search_result()]

        rewriter = QueryRewriter(llm=None)
        judge = EvidenceJudge(threshold=0.99)  # Very high threshold → never sufficient
        rag = AgenticRAG(config, mock_search, rewriter, judge)

        result = rag.retrieve("test query")
        assert not result.evidence_sufficient
        assert result.degraded
        assert result.rounds_used == 2

    def test_tracks_all_queries(self):
        config = RAGConfig(max_retrieval_rounds=2, rerank_top_k=5)
        mock_search = MagicMock()
        mock_search.search.return_value = [_make_search_result()]

        rewriter = QueryRewriter(llm=None)
        judge = EvidenceJudge(threshold=0.99)
        rag = AgenticRAG(config, mock_search, rewriter, judge)

        result = rag.retrieve("original query")
        assert "original query" in result.queries_used

    def test_deduplicates_results(self):
        config = RAGConfig(max_retrieval_rounds=2, rerank_top_k=5)
        same_result = _make_search_result()
        mock_search = MagicMock()
        mock_search.search.return_value = [same_result]

        rewriter = QueryRewriter(llm=None)
        judge = EvidenceJudge(threshold=0.99)
        rag = AgenticRAG(config, mock_search, rewriter, judge)

        result = rag.retrieve("test")
        # Same chunk should appear only once
        chunk_ids = [r.chunk.chunk_id for r in result.sources]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_refinement_query_includes_context(self):
        rag = AgenticRAG.__new__(AgenticRAG)  # Create without __init__
        results = [_make_search_result(class_name="OrderService", method_name="calculateTotal")]
        refined = rag._build_refinement_query("what is the bug?", results)
        assert "OrderService" in refined
        assert "calculateTotal" in refined

    def test_follows_java_method_reference_before_judging(self):
        config = RAGConfig(max_retrieval_rounds=1, rerank_top_k=5)
        calculate = _make_search_result(
            class_name="OrderService",
            method_name="calculateTotal",
            content="items.stream().map(OrderItem::getSubtotal)",
            score=0.05,
        )
        subtotal = _make_search_result(
            class_name="OrderItem",
            method_name="getSubtotal",
            content="return unitPrice.multiply(quantity);",
            file_path="src/OrderItem.java",
            score=0.04,
        )
        search = MagicMock()
        search.search.side_effect = [[calculate], [subtotal]]
        judge = MagicMock()
        judge.judge.return_value = (True, 0.9, "dependency resolved")
        rag = AgenticRAG(config, search, QueryRewriter(llm=None), judge)

        result = rag.retrieve("OrderService.calculateTotal ignores quantity")

        assert "OrderItem getSubtotal" in result.queries_used
        assert any(item.chunk.slice.method_name == "getSubtotal" for item in result.sources)
        judge.judge.assert_called_once()
