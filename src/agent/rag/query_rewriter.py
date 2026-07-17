"""Query Rewriter for Agentic RAG.

Uses LLM to rewrite user queries into multiple retrieval-friendly sub-queries.
Handles:
- Java terminology expansion (e.g., "bug" → "calculateTotal incorrect logic")
- Code-specific reformulation
- Breaking complex questions into focused sub-queries
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """\
你是一个代码检索查询改写专家。你的任务是将用户的自然语言问题改写为多个更适合代码检索的子查询。

规则：
1. 每个子查询应该聚焦于一个具体的代码概念
2. 保留原始查询中的类名、方法名等代码标识符
3. 将中文描述转换为包含英文代码关键词的查询
4. 每个查询应该独立可检索
5. 返回 2-4 个子查询

输出格式（严格 JSON 数组）：
["query1", "query2", "query3"]

不要输出其他内容，只输出 JSON 数组。
"""


class QueryRewriter:
    """Rewrites user queries into multiple retrieval-friendly sub-queries.

    Uses LLM to generate focused queries that better match code chunks.
    Falls back to the original query if LLM is unavailable.
    """

    def __init__(self, llm: ChatOpenAI | None = None) -> None:
        self._llm = llm

    def rewrite(self, query: str, max_queries: int = 4) -> list[str]:
        """Rewrite a query into multiple sub-queries.

        Args:
            query: The original user query
            max_queries: Maximum number of sub-queries to generate

        Returns:
            List of rewritten queries (always includes the original)
        """
        if not query.strip():
            return [query]

        # If no LLM available, return the original query
        if self._llm is None:
            return [query]

        try:
            return self._rewrite_with_llm(query, max_queries)
        except Exception as e:
            logger.warning("Query rewrite failed, using original: %s", e)
            return [query]

    def _rewrite_with_llm(self, query: str, max_queries: int) -> list[str]:
        """Use LLM to rewrite the query."""
        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{query}\n\n请改写为最多 {max_queries} 个检索子查询："),
        ]

        response = self._llm.invoke(messages)
        content = response.content.strip()

        # Parse JSON array from response
        import json
        # Try to extract JSON array from the response
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            json_str = content[start:end + 1]
            queries = json.loads(json_str)
            if isinstance(queries, list):
                # Filter empty strings and limit count
                valid = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
                # Always include the original query
                if query not in valid:
                    valid.insert(0, query)
                return valid[:max_queries]

        # Fallback: return original query
        return [query]
