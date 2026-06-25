"""Tool execution boundary for model-requested tool calls."""

from collections.abc import Sequence
from dataclasses import dataclass

from ori.types.tools import JsonObject, ToolDefinition, ToolFunction, ToolResult
from ori.events import ToolExecutionOutcome


@dataclass(frozen=True)
class ToolExecutionRequest:
    """Model-requested tool execution."""

    call_id: str
    tool_name: str
    arguments: JsonObject


class ToolExecutor:
    """Executes model-requested tools and normalizes outcomes."""

    def __init__(self, tools: Sequence[ToolDefinition] = ()) -> None:
        """Create an executor with model-callable tool definitions."""

        self._tools = tuple(tools)

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        """Return model-visible tool definitions."""

        return self._tools

    async def execute(
        self,
        request: ToolExecutionRequest,
    ) -> ToolExecutionOutcome:
        """Execute one tool request and return a normalized outcome."""

        result, is_error = await self._call_tool(
            request.tool_name,
            request.arguments,
        )
        return ToolExecutionOutcome.from_result(
            call_id=request.call_id,
            tool_name=request.tool_name,
            result=result,
            is_error=is_error,
        )

    async def _call_tool(
        self,
        tool_name: str,
        arguments: JsonObject,
    ) -> tuple[ToolResult, bool]:
        """Resolve and call a tool while normalizing tool failures."""

        try:
            tool = self._get_tool(tool_name)
            if tool is None:
                return ToolResult.text(f"Tool '{tool_name}' not found"), True
            return await tool(**arguments), False
        except Exception as error:
            return ToolResult.text(str(error)), True

    def _get_tool(self, tool_name: str) -> ToolFunction | None:
        """Find a registered tool implementation by name."""

        normalized_tool_name = tool_name.lower().strip()
        for tool in self._tools:
            if tool.name == normalized_tool_name:
                return tool.fn
        return None
