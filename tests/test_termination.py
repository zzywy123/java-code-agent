"""Tests for agent termination conditions.

Verifies that the agent terminates correctly when:
- No tool calls are returned (normal completion)
- Maximum iterations reached
- Consecutive failures exceed limit
- Error state is set
"""

from __future__ import annotations

import pytest

from agent.config import AgentConfig


def should_end(state: dict, config: AgentConfig) -> bool:
    """Replicate the router's termination logic for testing."""
    # 1. Error state
    if state.get("error"):
        return True
    # 2. Max iterations
    if state["iteration"] >= config.max_iterations:
        return True
    # 3. Consecutive failures
    if state["consecutive_failures"] >= config.max_consecutive_failures:
        return True
    # 4. No pending tool calls
    if not state.get("pending_tool_calls"):
        return True
    return False


class TestTerminationConditions:
    """Tests for termination logic."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(max_iterations=15, max_consecutive_failures=3)

    @pytest.fixture
    def base_state(self) -> dict:
        return {
            "iteration": 0,
            "consecutive_failures": 0,
            "pending_tool_calls": [],
            "error": None,
        }

    def test_no_tool_calls_terminates(self, config, base_state):
        """Agent should terminate when no tool calls are pending."""
        assert should_end(base_state, config) is True

    def test_with_tool_calls_continues(self, config, base_state):
        """Agent should continue when tool calls are pending."""
        from agent.models import ToolCallRequest
        base_state["pending_tool_calls"] = [
            ToolCallRequest(id="c1", name="search_code", arguments={"query": "test"}),
        ]
        assert should_end(base_state, config) is False

    def test_max_iterations_terminates(self, config, base_state):
        """Agent should terminate at max iterations."""
        from agent.models import ToolCallRequest
        base_state["iteration"] = 15
        base_state["pending_tool_calls"] = [
            ToolCallRequest(id="c1", name="search_code", arguments={"query": "test"}),
        ]
        assert should_end(base_state, config) is True

    def test_below_max_iterations_continues(self, config, base_state):
        """Agent should continue below max iterations."""
        from agent.models import ToolCallRequest
        base_state["iteration"] = 14
        base_state["pending_tool_calls"] = [
            ToolCallRequest(id="c1", name="search_code", arguments={"query": "test"}),
        ]
        assert should_end(base_state, config) is False

    def test_consecutive_failures_terminates(self, config, base_state):
        """Agent should terminate when consecutive failures exceed limit."""
        from agent.models import ToolCallRequest
        base_state["consecutive_failures"] = 3
        base_state["pending_tool_calls"] = [
            ToolCallRequest(id="c1", name="search_code", arguments={"query": "test"}),
        ]
        assert should_end(base_state, config) is True

    def test_below_failure_limit_continues(self, config, base_state):
        """Agent should continue below failure limit."""
        from agent.models import ToolCallRequest
        base_state["consecutive_failures"] = 2
        base_state["pending_tool_calls"] = [
            ToolCallRequest(id="c1", name="search_code", arguments={"query": "test"}),
        ]
        assert should_end(base_state, config) is False

    def test_error_state_terminates(self, config, base_state):
        """Agent should terminate on error state."""
        from agent.models import ToolCallRequest
        base_state["error"] = "Something went wrong"
        base_state["pending_tool_calls"] = [
            ToolCallRequest(id="c1", name="search_code", arguments={"query": "test"}),
        ]
        assert should_end(base_state, config) is True

    def test_custom_config_limits(self):
        """Agent should respect custom config limits."""
        config = AgentConfig(max_iterations=5, max_consecutive_failures=1)
        state = {
            "iteration": 5,
            "consecutive_failures": 0,
            "pending_tool_calls": [],
            "error": None,
        }
        assert should_end(state, config) is True

        state["iteration"] = 4
        from agent.models import ToolCallRequest
        state["pending_tool_calls"] = [
            ToolCallRequest(id="c1", name="test", arguments={}),
        ]
        assert should_end(state, config) is False
