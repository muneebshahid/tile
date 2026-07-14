"""Runtime contracts for model-requested tool executions."""

from pydantic import BaseModel, SerializeAsAny

from tile.types.conversation import ToolResultTurn
from tile.types.tools import ToolDetails, ToolResult


class ToolExecutionOutcome(BaseModel):
    """Full runtime outcome for a tool execution."""

    tool_result_turn: ToolResultTurn
    details: SerializeAsAny[ToolDetails] | None = None
    terminate: bool = False

    @property
    def result(self) -> ToolResult:
        """Return the full tool result including non-replay metadata."""

        return ToolResult(
            content=self.tool_result_turn.content,
            details=self.details,
            terminate=self.terminate,
        )

    @classmethod
    def from_result(
        cls,
        *,
        call_id: str,
        tool_name: str,
        result: ToolResult,
        is_error: bool,
    ) -> "ToolExecutionOutcome":
        """Build an execution outcome from a raw tool result."""

        return cls(
            tool_result_turn=ToolResultTurn(
                call_id=call_id,
                tool_name=tool_name,
                content=result.content,
                is_error=is_error,
            ),
            details=result.details,
            terminate=result.terminate and not is_error,
        )
