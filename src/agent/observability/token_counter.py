"""Token accounting with response metadata and deterministic estimation."""

from __future__ import annotations

import json
from typing import Any

from agent.config import ObservabilityConfig
from agent.observability.models import TokenUsage


class TokenCounter:
    def __init__(
        self,
        provider: str,
        model: str,
        config: ObservabilityConfig,
    ) -> None:
        self._provider = provider
        self._model = model
        self._config = config

    def measure(
        self,
        messages: Any,
        response: Any,
        duration_ms: float,
    ) -> TokenUsage:
        input_tokens, output_tokens, total_tokens = self._read_usage(response)
        estimated = input_tokens is None or output_tokens is None
        if estimated:
            input_tokens = self._estimate_tokens(self._render_input(messages))
            output_tokens = self._estimate_tokens(self._render_response(response))
            total_tokens = input_tokens + output_tokens
        elif total_tokens is None:
            total_tokens = input_tokens + output_tokens

        cost = self._calculate_cost(input_tokens, output_tokens)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=self._model,
            provider=self._provider,
            estimated=estimated,
            cost=cost,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _read_usage(response: Any) -> tuple[int | None, int | None, int | None]:
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            return (
                TokenCounter._as_int(usage.get("input_tokens")),
                TokenCounter._as_int(usage.get("output_tokens")),
                TokenCounter._as_int(usage.get("total_tokens")),
            )

        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            token_usage = metadata.get("token_usage") or metadata.get("usage")
            if isinstance(token_usage, dict):
                return (
                    TokenCounter._as_int(token_usage.get("prompt_tokens")),
                    TokenCounter._as_int(token_usage.get("completion_tokens")),
                    TokenCounter._as_int(token_usage.get("total_tokens")),
                )
        return None, None, None

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(self._model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _render_input(messages: Any) -> str:
        if isinstance(messages, (str, bytes)):
            return str(messages)
        if isinstance(messages, list):
            parts: list[str] = []
            for message in messages:
                parts.append(str(getattr(message, "content", message)))
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))
            return "\n".join(parts)
        return str(messages)

    @staticmethod
    def _render_response(response: Any) -> str:
        parts = [str(getattr(response, "content", response))]
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))
        return "\n".join(parts)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        input_rate = self._config.input_cost_per_million
        output_rate = self._config.output_cost_per_million
        if input_rate is None or output_rate is None:
            return None
        return (
            input_tokens * input_rate / 1_000_000
            + output_tokens * output_rate / 1_000_000
        )

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
