"""Evidence sufficiency judge for Agentic RAG.

Determines whether retrieved code chunks provide sufficient evidence
to answer the user's question. Uses both rule-based heuristics and
optional LLM-based judgment.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.models import SearchResult

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
你是一个代码检索证据评估专家。判断检索到的代码片段是否足以回答用户问题。

评估标准：
1. 检索结果是否包含问题涉及的核心类或方法
2. 代码片段是否提供了足够的上下文来理解业务逻辑
3. 是否覆盖了问题的各个方面

输出格式（严格 JSON）：
{"sufficient": true/false, "confidence": 0.0-1.0, "reason": "简短原因"}

不要输出其他内容，只输出 JSON。
"""


class EvidenceJudge:
    """Judges whether retrieved evidence is sufficient to answer a query.

    Two modes:
    1. Rule-based: checks if query keywords appear in retrieved chunks
    2. LLM-based: uses LLM to assess evidence quality
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        threshold: float = 0.6,
    ) -> None:
        self._llm = llm
        self._threshold = threshold

    def judge(
        self,
        query: str,
        results: list[SearchResult],
    ) -> tuple[bool, float, str]:
        """Judge whether the results are sufficient to answer the query.

        Args:
            query: The original user query
            results: Retrieved search results

        Returns:
            (sufficient, confidence, reason) tuple
        """
        if not results:
            return False, 0.0, "没有检索到任何结果"

        # Rule-based check
        rule_sufficient, rule_conf, rule_reason = self._rule_based_judge(query, results)

        # Exact identifier coverage is deterministic evidence. Ask the LLM
        # only when the rule-based result remains insufficient but uncertain.
        if rule_sufficient:
            return rule_sufficient, rule_conf, rule_reason
        if self._llm is not None and 0.3 < rule_conf < 0.8:
            try:
                return self._llm_judge(query, results)
            except Exception as e:
                logger.warning("LLM judge failed, using rule-based: %s", e)

        return rule_sufficient, rule_conf, rule_reason

    def _rule_based_judge(
        self,
        query: str,
        results: list[SearchResult],
    ) -> tuple[bool, float, str]:
        """Simple rule-based evidence sufficiency check."""
        if not results:
            return False, 0.0, "无检索结果"

        # Code identifiers are more stable than whitespace tokens for mixed
        # Chinese/Java queries. Split qualified names into their components.
        identifiers = re.findall(r"[A-Za-z_$][A-Za-z0-9_$.]*", query)
        query_tokens = {
            part.lower()
            for identifier in identifiers
            for part in identifier.split(".")
            if len(part) >= 2 and part.lower() not in {"bug", "java", "code"}
        }

        # Pure Chinese queries produce no code identifiers.  Return a
        # mid-range confidence so the optional LLM judge gets a chance to
        # evaluate, instead of always falling through to "degraded".
        if not query_tokens:
            return False, 0.5, "纯中文查询，无代码标识符，需 LLM 判断"

        # Check how many query tokens appear in results
        result_text = " ".join(
            f"{r.chunk.slice.class_name} {r.chunk.slice.method_name} {r.chunk.slice.content[:200]}"
            for r in results[:5]
        ).lower()

        matched = sum(1 for t in query_tokens if t in result_text)
        total = len(query_tokens) if query_tokens else 1

        match_ratio = matched / total

        # RRF scores are intentionally small (often around 0.05), so rank and
        # exact symbol coverage must be used instead of an absolute score.
        query_lower = query.lower()
        has_class_match = any(
            r.chunk.slice.class_name.lower() in query_lower
            for r in results[:3]
            if r.chunk.slice.class_name
        )
        has_method_match = any(
            r.chunk.slice.method_name.lower() in query_lower
            for r in results[:3]
            if r.chunk.slice.method_name not in {"", "<class>", "<file>"}
        )
        has_top_rank = any(0 < r.rank <= 3 for r in results[:3])

        # Combine signals
        confidence = match_ratio * 0.5
        if has_class_match:
            confidence += 0.2
        if has_method_match:
            confidence += 0.2
        if has_top_rank:
            confidence += 0.1

        confidence = min(confidence, 1.0)
        sufficient = confidence >= self._threshold

        if sufficient:
            reason = f"匹配度 {match_ratio:.0%}，{matched}/{total} 个关键词命中"
        else:
            reason = f"匹配度不足 {match_ratio:.0%}，需要更多上下文"

        return sufficient, confidence, reason

    def _llm_judge(
        self,
        query: str,
        results: list[SearchResult],
    ) -> tuple[bool, float, str]:
        """Use LLM to judge evidence sufficiency."""
        # Build context from top results
        context_parts = []
        for i, r in enumerate(results[:5], 1):
            s = r.chunk.slice
            context_parts.append(
                f"[{i}] {s.file_path}:{s.start_line}-{s.end_line} "
                f"{s.class_name}.{s.method_name}\n{s.content[:300]}"
            )
        context = "\n\n".join(context_parts)

        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{query}\n\n检索到的代码：\n{context}\n\n请评估证据充分性："),
        ]

        response = self._llm.invoke(messages)
        content = response.content.strip()

        # Parse JSON response
        import json
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(content[start:end + 1])
                return (
                    data.get("sufficient", False),
                    float(data.get("confidence", 0.5)),
                    data.get("reason", "LLM 评估完成"),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback
        return False, 0.5, "LLM 返回格式异常"
