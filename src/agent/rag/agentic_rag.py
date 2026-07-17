"""Agentic RAG: multi-round retrieval with query rewriting and evidence judgment.

Pipeline:
1. Query Rewrite: generate multiple retrieval queries
2. Hybrid Search: execute search for each query
3. Evidence Judge: assess if results are sufficient
4. If insufficient and rounds remaining: refine query and retry
5. If max rounds reached: degrade gracefully (return best results so far)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.config import RAGConfig
from agent.indexing.hybrid_search import HybridSearchEngine
from agent.models import RAGResult, SearchResult
from agent.observability.tracer import observe_span
from agent.rag.evidence_judge import EvidenceJudge
from agent.rag.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class AgenticRAG:
    """Agentic RAG controller with multi-round retrieval.

    Orchestrates QueryRewriter, HybridSearchEngine, and EvidenceJudge
    into a retrieval loop with configurable max rounds and degradation.
    """

    def __init__(
        self,
        config: RAGConfig,
        search_engine: HybridSearchEngine,
        query_rewriter: QueryRewriter,
        evidence_judge: EvidenceJudge,
    ) -> None:
        self._config = config
        self._search = search_engine
        self._rewriter = query_rewriter
        self._judge = evidence_judge

    def retrieve(self, query: str, max_rounds: int | None = None) -> RAGResult:
        """Execute retrieval inside one observable RAG span."""
        with observe_span("rag.retrieve", {"query_length": len(query)}):
            return self._retrieve(query, max_rounds)

    def _retrieve(self, query: str, max_rounds: int | None = None) -> RAGResult:
        """Execute agentic RAG retrieval.

        Args:
            query: The user's question
            max_rounds: Max retrieval rounds (default from config)

        Returns:
            RAGResult with sources, round count, and sufficiency flag
        """
        if max_rounds is None:
            max_rounds = self._config.max_retrieval_rounds

        all_results: list[SearchResult] = []
        all_queries: list[str] = []
        seen_chunk_ids: set[str] = set()

        for round_num in range(max_rounds):
            with observe_span("rag.round", {"round": round_num + 1}) as round_span:
                # Step 1: Generate queries
                if round_num == 0:
                    # Most code questions already contain a searchable symbol.
                    # Try it directly before spending a model call on rewriting.
                    queries = [query]
                elif round_num == 1:
                    queries = self._rewriter.rewrite(query)
                else:
                    refinement = self._build_refinement_query(query, all_results)
                    queries = [refinement]

                queries = [item for item in queries if item not in all_queries]
                if not queries:
                    queries = [self._build_refinement_query(query, all_results)]

                all_queries.extend(queries)

                # Step 2: Search for each query
                for q in queries:
                    results = self._search_query(q, round_num + 1)
                    for r in results:
                        if r.chunk.chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(r.chunk.chunk_id)
                            all_results.append(r)

                # Follow explicit Java method references such as
                # OrderItem::getSubtotal before judging evidence sufficiency.
                dependency_queries = self._extract_dependency_queries(all_results)
                for dependency_query in dependency_queries:
                    if dependency_query in all_queries:
                        continue
                    all_queries.append(dependency_query)
                    results = self._search_query(dependency_query, round_num + 1)
                    for r in results:
                        if r.chunk.chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(r.chunk.chunk_id)
                            all_results.append(r)

                # Step 3: Judge evidence sufficiency
                sufficient, confidence, reason = self._judge.judge(query, all_results)
                if round_span is not None:
                    round_span.attributes.update({
                        "query_count": len(queries) + len(dependency_queries),
                        "result_count": len(all_results),
                        "sufficient": sufficient,
                        "confidence": confidence,
                    })

                logger.info(
                    "RAG round %d/%d: %d results, sufficient=%s (conf=%.2f: %s)",
                    round_num + 1, max_rounds, len(all_results),
                    sufficient, confidence, reason,
                )

            if sufficient:
                return RAGResult(
                    answer="",  # Answer generation is done by the agent, not here
                    sources=sorted(all_results, key=lambda r: r.score, reverse=True)[:self._config.rerank_top_k],
                    rounds_used=round_num + 1,
                    evidence_sufficient=True,
                    degraded=False,
                    queries_used=all_queries,
                )

        # Max rounds reached — degrade gracefully
        logger.warning(
            "RAG degraded after %d rounds: evidence insufficient",
            max_rounds,
        )
        return RAGResult(
            answer="",
            sources=sorted(all_results, key=lambda r: r.score, reverse=True)[:self._config.rerank_top_k],
            rounds_used=max_rounds,
            evidence_sufficient=False,
            degraded=True,
            queries_used=all_queries,
        )

    def _build_refinement_query(
        self,
        original_query: str,
        existing_results: list[SearchResult],
    ) -> str:
        """Build a refinement query based on existing results.

        Extracts class/method names from existing results to narrow the search.
        """
        # Gather class names and method names from top results
        classes = set()
        methods = set()
        for r in existing_results[:5]:
            if r.chunk.slice.class_name:
                classes.add(r.chunk.slice.class_name)
            if r.chunk.slice.method_name and r.chunk.slice.method_name != "<class>":
                methods.add(r.chunk.slice.method_name)

        # Build a more specific query
        parts = [original_query]
        if classes:
            parts.append(" ".join(classes))
        if methods:
            parts.append(" ".join(methods))

        return " ".join(parts)

    @staticmethod
    def _extract_dependency_queries(results: list[SearchResult]) -> list[str]:
        """Extract a small set of Class::method references from code chunks."""
        queries: list[str] = []
        seen: set[str] = set()
        for result in results[:5]:
            references = re.findall(
                r"\b([A-Z][A-Za-z0-9_$]*)::([A-Za-z_$][A-Za-z0-9_$]*)\b",
                result.chunk.slice.content,
            )
            for class_name, method_name in references:
                query = f"{class_name} {method_name}"
                if query not in seen:
                    seen.add(query)
                    queries.append(query)
                if len(queries) >= 4:
                    return queries
        return queries

    def _search_query(self, query: str, round_number: int) -> list[SearchResult]:
        with observe_span("rag.search", {
            "round": round_number,
            "query_length": len(query),
        }) as span:
            results = self._search.search(query, top_k=self._config.rerank_top_k)
            if span is not None:
                span.attributes["result_count"] = len(results)
            return results
