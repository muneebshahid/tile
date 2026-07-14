"""Tool execution boundary for model-requested tool calls."""

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from tile.types.tool_execution import (
    ToolExecutionOutcome,
    ToolInputIssue,
    ToolInputValidationFailure,
    ToolInvocationFailure,
)
from tile.types.tools import JsonObject, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# Placeholder used only to validate function signatures without constructing input.
_SIGNATURE_ARGUMENT = object()


@dataclass(frozen=True)
class _ValidatedArguments:
    """Successfully validated model-controlled arguments."""

    value: BaseModel


class ToolExecutor:
    """Executes model-requested tools and normalizes outcomes."""

    def __init__(self, tools: Sequence[ToolDefinition] = ()) -> None:
        """Create an executor with model-callable tool definitions."""

        self._tools = tuple(tools)
        seen: set[str] = set()
        for tool in self._tools:
            _require_unique_name(tool, seen)
            _require_invocable_signature(tool)

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        """Return model-visible tool definitions."""

        return self._tools

    async def execute(
        self,
        *,
        call_id: str,
        tool_name: str,
        arguments: JsonObject,
    ) -> ToolExecutionOutcome:
        """Execute one tool request and return a normalized outcome."""

        result = await self._call_tool(
            tool_name,
            arguments,
        )
        return ToolExecutionOutcome.from_result(
            call_id=call_id,
            tool_name=tool_name,
            result=result,
        )

    async def _call_tool(
        self,
        tool_name: str,
        arguments: JsonObject,
    ) -> ToolResult:
        """Resolve and call a tool while normalizing tool failures."""

        tool = self._get_tool(tool_name)
        if tool is None:
            return ToolResult.error(f"Tool '{tool_name}' not found")

        validated = self._validate_arguments(tool, arguments)
        if isinstance(validated, ToolResult):
            return validated

        return await self._invoke_tool(tool, validated.value)

    def _get_tool(self, tool_name: str) -> ToolDefinition | None:
        """Find a registered tool definition by case-insensitive name."""

        normalized_tool_name = tool_name.lower().strip()
        for tool in self._tools:
            if tool.name.lower() == normalized_tool_name:
                return tool
        return None

    @staticmethod
    def _validate_arguments(
        tool: ToolDefinition,
        arguments: JsonObject,
    ) -> _ValidatedArguments | ToolResult:
        """Validate model arguments or return a model-correctable error."""

        try:
            return _ValidatedArguments(tool.input_model.model_validate(arguments))
        except ValidationError as error:
            issues = _validation_issues(error)
            return ToolResult.error(
                _format_validation_failure(tool.name, issues),
                details=ToolInputValidationFailure(
                    tool_name=tool.name,
                    issues=issues,
                ),
            )

    @staticmethod
    async def _invoke_tool(
        tool: ToolDefinition,
        arguments: BaseModel,
    ) -> ToolResult:
        """Invoke validated tool code and normalize escaped exceptions."""

        try:
            return await tool.fn(arguments)
        except Exception as error:
            logger.debug("Tool '%s' invocation failed", tool.name, exc_info=True)
            return ToolResult.error(
                str(error),
                details=ToolInvocationFailure(
                    tool_name=tool.name,
                    exception_type=type(error).__name__,
                    message=str(error),
                ),
            )


def _validation_issues(error: ValidationError) -> list[ToolInputIssue]:
    """Convert Pydantic errors into stable runtime issue metadata."""

    return [
        ToolInputIssue(
            location=tuple(item["loc"]),
            code=item["type"],
            message=item["msg"],
        )
        for item in error.errors(include_input=False, include_url=False)
    ]


def _format_validation_failure(
    tool_name: str,
    issues: list[ToolInputIssue],
) -> str:
    """Render validation issues as concise model-visible correction text."""

    lines = [f"Invalid arguments for tool '{tool_name}':"]
    lines.extend(
        f"- {_format_location(issue.location)}: {issue.message}" for issue in issues
    )
    return "\n".join(lines)


def _format_location(location: tuple[str | int, ...]) -> str:
    """Render a Pydantic issue location as a familiar field path."""

    rendered = ""
    for part in location:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}" if rendered else part
    return rendered or "arguments"


def _require_unique_name(tool: ToolDefinition, seen: set[str]) -> None:
    """Reject case-insensitive duplicate tool names."""

    name = tool.name.lower()
    if name in seen:
        raise ValueError(f"Duplicate tool name: {tool.name}")
    seen.add(name)


def _require_invocable_signature(tool: ToolDefinition) -> None:
    """Reject functions that cannot receive one validated input model."""

    try:
        inspect.signature(tool.fn).bind(_SIGNATURE_ARGUMENT)
    except TypeError as error:
        raise ValueError(
            f"Tool '{tool.name}' function must accept one validated input model "
            f"as a positional argument: {error}"
        ) from error
