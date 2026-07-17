"""LLM client using OpenAI-compatible API.

Supports DeepSeek, OpenAI, and Ollama providers.
Uses langchain-openai for integration with LangGraph.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from agent.config import LLMConfig, LLMProvider, load_observability_config
from agent.observability.context import current_trace
from agent.observability.token_counter import TokenCounter
from agent.observability.tracer import observe_span, record_token_usage


class ObservableChatModel:
    """Transparent invoke/bind_tools wrapper that records LLM usage."""

    def __init__(self, delegate: Any, config: LLMConfig) -> None:
        self._delegate = delegate
        self._config = config
        self._counter = TokenCounter(
            provider=config.provider.value,
            model=config.model,
            config=load_observability_config(),
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ObservableChatModel":
        return ObservableChatModel(self._delegate.bind_tools(tools, **kwargs), self._config)

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        if current_trace.get() is None:
            return self._delegate.invoke(input, *args, **kwargs)

        import time

        started = time.perf_counter()
        with observe_span("llm.invoke", {
            "provider": self._config.provider.value,
            "model": self._config.model,
        }) as span:
            response = self._delegate.invoke(input, *args, **kwargs)
            duration_ms = (time.perf_counter() - started) * 1000
            usage = self._counter.measure(input, response, duration_ms)
            record_token_usage(usage)
            if span is not None:
                span.attributes.update({
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "estimated": usage.estimated,
                })
            return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def create_llm(config: LLMConfig) -> Any:
    """Create a ChatOpenAI instance from LLM configuration.

    Supports DeepSeek, OpenAI, and Ollama providers.

    Args:
        config: LLM configuration

    Returns:
        Configured ChatOpenAI instance
    """
    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        # Tool calling support
        model_kwargs={
            "tool_choice": "auto",
        },
    )
    observability = load_observability_config()
    return ObservableChatModel(llm, config) if observability.enabled else llm


def create_llm_with_tools(
    config: LLMConfig,
    tools: list[dict[str, Any]],
) -> Any:
    """Create a ChatOpenAI instance bound with tools.

    Args:
        config: LLM configuration
        tools: List of OpenAI-format tool definitions

    Returns:
        ChatOpenAI instance with tools bound
    """
    llm = create_llm(config)
    return llm.bind_tools(tools)
