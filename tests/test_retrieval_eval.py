"""Retrieval evaluation set and metrics for RAG quality.

Evaluation Set: 12 queries about the demo-repo Order Service.
Each query has expected relevant chunks identified by (file_path, method_name).

Metrics:
- Recall@K: fraction of queries where at least one relevant chunk appears in top-K
- MRR (Mean Reciprocal Rank): average of 1/rank of first relevant result
- Citation Correctness: fraction of results where file_path:line references are valid

This test uses BM25-only search (no embedding model required) to validate
the retrieval pipeline end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.indexing.bm25_index import BM25Index
from agent.indexing.chunk_store import ChunkStore, compute_chunk_id, compute_file_hash
from agent.indexing.java_slicer import JavaSlicer
from agent.models import SearchResult


# ============================================================
# Evaluation Dataset
# ============================================================

# Each entry: (query, expected_matches)
# expected_matches: list of (file_contains, method_name) tuples
# file_contains: substring that should appear in the chunk's file_path
EVAL_DATASET: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "OrderService 的 calculateTotal 方法有什么 bug？",
        [("OrderService.java", "calculateTotal")],
    ),
    (
        "OrderService createOrder 如何创建新订单？",
        [("OrderService.java", "createOrder"), ("OrderController.java", "createOrder")],
    ),
    (
        "Money add multiply 方法有哪些？",
        [("Money.java", "<class>"), ("Money.java", "add"), ("Money.java", "multiply")],
    ),
    (
        "OrderService payOrder 订单支付流程",
        [("OrderService.java", "payOrder"), ("OrderController.java", "payOrder")],
    ),
    (
        "OrderItem getSubtotal 小计怎么计算？",
        [("OrderItem.java", "getSubtotal")],
    ),
    (
        "OrderRepository findByCustomerId 查找客户订单",
        [("OrderRepository.java", "findByCustomerId")],
    ),
    (
        "OrderStatus 订单的状态有哪些？",
        [("OrderStatus.java", "<class>")],
    ),
    (
        "OrderService addItem 如何给订单添加商品？",
        [("OrderService.java", "addItem"), ("OrderController.java", "addItem")],
    ),
    (
        "OrderController REST 接口有哪些？",
        [("OrderController.java", "<class>")],
    ),
    (
        "OrderService 依赖 OrderRepository",
        [("OrderService.java", "<class>"), ("OrderApplication.java", "orderService")],
    ),
    (
        "OrderService getOrder 如何获取订单详情？",
        [("OrderService.java", "getOrder"), ("OrderController.java", "getOrder")],
    ),
    (
        "Order Order 构造函数做了什么？",
        [("Order.java", "Order")],
    ),
]


# ============================================================
# Metrics
# ============================================================

def recall_at_k(
    results_list: list[list[SearchResult]],
    expected_list: list[list[tuple[str, str]]],
    k: int,
) -> float:
    """Recall@K: fraction of queries with ≥1 relevant result in top-K.

    A result is relevant if its file_path contains the expected file substring
    AND its method_name matches the expected method name.
    """
    if not results_list:
        return 0.0

    hits = 0
    for results, expected in zip(results_list, expected_list):
        top_k = results[:k]
        for r in top_k:
            for file_sub, method in expected:
                if file_sub in r.chunk.slice.file_path and r.chunk.slice.method_name == method:
                    hits += 1
                    break
            else:
                continue
            break

    return hits / len(results_list)


def mean_reciprocal_rank(
    results_list: list[list[SearchResult]],
    expected_list: list[list[tuple[str, str]]],
) -> float:
    """MRR: average of 1/rank of the first relevant result.

    Returns 0 for queries with no relevant result in the result list.
    """
    if not results_list:
        return 0.0

    rr_sum = 0.0
    for results, expected in zip(results_list, expected_list):
        for i, r in enumerate(results):
            for file_sub, method in expected:
                if file_sub in r.chunk.slice.file_path and r.chunk.slice.method_name == method:
                    rr_sum += 1.0 / (i + 1)
                    break
            else:
                continue
            break

    return rr_sum / len(results_list)


def citation_correctness(
    results_list: list[list[SearchResult]],
) -> float:
    """Citation correctness: fraction of results where file_path exists and
    start_line > 0 (i.e., the citation is valid and pointable).
    """
    total = 0
    correct = 0
    for results in results_list:
        for r in results:
            total += 1
            if r.chunk.slice.file_path and r.chunk.slice.start_line > 0:
                correct += 1

    return correct / total if total > 0 else 0.0


# ============================================================
# Evaluation Tests
# ============================================================

@pytest.fixture
def eval_index() -> BM25Index:
    """Build a BM25 index from the demo repo for evaluation."""
    demo_dir = Path("demo-repo/src/main/java/com/example/order")
    if not demo_dir.exists():
        pytest.skip("Demo repo not available")

    slicer = JavaSlicer()
    chunks = []
    for f in sorted(demo_dir.glob("*.java")):
        slices = slicer.slice_file(f, module="order-service")
        for s in slices:
            chunk_id = compute_chunk_id(s.file_path, s.symbol_signature, s.content)
            from agent.models import CodeChunk
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                slice=s,
                file_hash=compute_file_hash(s.content),
            ))

    index = BM25Index()
    index.add(chunks)
    return index


class TestRetrievalEval:
    """Run the evaluation dataset against BM25 search."""

    def test_all_queries_return_results(self, eval_index: BM25Index):
        """Every query should produce at least one result."""
        for query, _ in EVAL_DATASET:
            results = eval_index.search(query, top_k=10)
            assert len(results) > 0, f"No results for query: {query}"

    def test_recall_at_1(self, eval_index: BM25Index):
        """Recall@1: at least 50% of queries have a relevant result at rank 1."""
        results_list = []
        expected_list = []
        for query, expected in EVAL_DATASET:
            results_list.append(eval_index.search(query, top_k=10))
            expected_list.append(expected)

        r1 = recall_at_k(results_list, expected_list, k=1)
        assert r1 >= 0.5, f"Recall@1 = {r1:.2f}, expected >= 0.5"

    def test_recall_at_3(self, eval_index: BM25Index):
        """Recall@3: at least 70% of queries have a relevant result in top 3."""
        results_list = []
        expected_list = []
        for query, expected in EVAL_DATASET:
            results_list.append(eval_index.search(query, top_k=10))
            expected_list.append(expected)

        r3 = recall_at_k(results_list, expected_list, k=3)
        assert r3 >= 0.7, f"Recall@3 = {r3:.2f}, expected >= 0.7"

    def test_recall_at_5(self, eval_index: BM25Index):
        """Recall@5: at least 80% of queries have a relevant result in top 5."""
        results_list = []
        expected_list = []
        for query, expected in EVAL_DATASET:
            results_list.append(eval_index.search(query, top_k=10))
            expected_list.append(expected)

        r5 = recall_at_k(results_list, expected_list, k=5)
        assert r5 >= 0.8, f"Recall@5 = {r5:.2f}, expected >= 0.8"

    def test_mrr(self, eval_index: BM25Index):
        """MRR: mean reciprocal rank should be at least 0.5."""
        results_list = []
        expected_list = []
        for query, expected in EVAL_DATASET:
            results_list.append(eval_index.search(query, top_k=10))
            expected_list.append(expected)

        mrr = mean_reciprocal_rank(results_list, expected_list)
        assert mrr >= 0.5, f"MRR = {mrr:.2f}, expected >= 0.5"

    def test_citation_correctness(self, eval_index: BM25Index):
        """All returned results should have valid citations."""
        results_list = []
        for query, _ in EVAL_DATASET:
            results_list.append(eval_index.search(query, top_k=10))

        acc = citation_correctness(results_list)
        assert acc >= 0.95, f"Citation correctness = {acc:.2f}, expected >= 0.95"

    def test_eval_dataset_coverage(self):
        """Evaluation dataset should have at least 12 queries."""
        assert len(EVAL_DATASET) >= 12, f"Only {len(EVAL_DATASET)} eval queries"
