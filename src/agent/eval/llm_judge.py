"""Optional LLM-as-a-Judge implementation."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


class LLMJudge:
    """Judge answers only when an explicit LLM instance is supplied."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    @property
    def enabled(self) -> bool:
        return self._llm is not None

    def judge(self, query: str, answer: str, rubric: str) -> float | None:
        if self._llm is None:
            return None
        response = self._llm.invoke([
            SystemMessage(content=(
                "Evaluate the answer against the rubric. Return JSON only: "
                '{"score": 0.0, "reason": "..."}. Do not execute tools.'
            )),
            HumanMessage(content=f"Query:\n{query}\n\nAnswer:\n{answer}\n\nRubric:\n{rubric}"),
        ])
        content = str(response.content)
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            score = float(json.loads(content[start:end + 1]).get("score"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return min(1.0, max(0.0, score))
