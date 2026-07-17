"""Base tool class and tool registry.

All tools inherit from BaseTool and are registered in the global ToolRegistry.
Tools produce ToolResult objects with structured status.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Any

from agent.models import ToolResult, ToolStatus
from agent.observability.tracer import observe_span, record_tool_metric
from agent.security.path_guard import PathViolationError, normalize_and_validate


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    parameters_schema: dict[str, Any]  # JSON Schema for LLM function calling

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        Returns:
            ToolResult with structured status and output.
        """
        ...

    def validate_path(self, path: str) -> Path:
        """Validate and normalize a path against repo_root.

        Args:
            path: Relative or absolute path

        Returns:
            Resolved absolute path

        Raises:
            PathViolationError: If path violates security constraints
        """
        return normalize_and_validate(path, self.repo_root)

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def _error_result(
        self,
        tool_call_id: str,
        status: ToolStatus,
        message: str,
    ) -> ToolResult:
        """Helper to create an error ToolResult."""
        return ToolResult(
            tool_call_id=tool_call_id,
            name=self.name,
            status=status,
            output=message,
        )

    def _success_result(
        self,
        tool_call_id: str,
        output: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Helper to create a success ToolResult."""
        return ToolResult(
            tool_call_id=tool_call_id,
            name=self.name,
            status=ToolStatus.SUCCESS,
            output=output,
            metadata=metadata or {},
        )


class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, tool_call_id: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            tool_call_id: ID from the LLM tool call
            **kwargs: Tool arguments

        Returns:
            ToolResult from tool execution
        """
        started = time.perf_counter()
        with observe_span(f"tool.{name}", {
            "tool": name,
            "argument_keys": sorted(kwargs.keys()),
        }) as span:
            tool = self._tools.get(name)
            if tool is None:
                result = ToolResult(
                    tool_call_id=tool_call_id,
                    name=name,
                    status=ToolStatus.NOT_FOUND,
                    output=f"工具不存在: {name}",
                )
            else:
                try:
                    result = tool.execute(tool_call_id=tool_call_id, **kwargs)
                except PathViolationError as e:
                    result = ToolResult(
                        tool_call_id=tool_call_id,
                        name=name,
                        status=ToolStatus.DENIED,
                        output=f"安全拒绝: {e}",
                    )
                except ValueError as e:
                    result = ToolResult(
                        tool_call_id=tool_call_id,
                        name=name,
                        status=ToolStatus.INVALID_ARGUMENT,
                        output=f"参数错误: {e}",
                    )
                except Exception as e:
                    result = ToolResult(
                        tool_call_id=tool_call_id,
                        name=name,
                        status=ToolStatus.EXECUTION_ERROR,
                        output=f"执行异常: {e}",
                    )

            duration_ms = (time.perf_counter() - started) * 1000
            record_tool_metric(name, result.status.value, duration_ms)
            if span is not None:
                span.attributes.update({
                    "status": result.status.value,
                    "duration_ms": duration_ms,
                })
                if result.status == ToolStatus.TIMEOUT:
                    span.status = "timeout"
                elif result.status != ToolStatus.SUCCESS:
                    span.status = "error"
            return result

    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def restricted(self, names: set[str]) -> "ToolRegistry":
        """Create a role-scoped registry backed by the same tool instances."""
        registry = ToolRegistry()
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                registry.register(tool)
        return registry

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
