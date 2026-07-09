"""Typed run outcomes and the result tools that report them."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, JsonValue, SerializeAsAny

from tile.types.tools import ToolDefinition, ToolResult

COMPLETE_TOOL_NAME = "complete"
FAIL_TOOL_NAME = "fail"
MAX_RESULT_FOLLOW_UPS = 8

RESULT_CONTRACT = f"""\
This run must end with a result tool call.
- When the task is complete, call `{COMPLETE_TOOL_NAME}` with your final result as \
its arguments. The arguments are validated against the required schema; validation \
errors come back as tool errors you can correct.
- If you cannot complete the task or cannot produce a conforming result, call \
`{FAIL_TOOL_NAME}` with a clear `reason`.
- Plain text does not end the run. Only a `{COMPLETE_TOOL_NAME}` or \
`{FAIL_TOOL_NAME}` call does."""

RESULT_FOLLOW_UP = (
    f"You ended your turn without calling `{COMPLETE_TOOL_NAME}` or "
    f"`{FAIL_TOOL_NAME}`. Call `{COMPLETE_TOOL_NAME}` with your final result, "
    f"or `{FAIL_TOOL_NAME}` with a reason if you cannot."
)

NO_RESULT_REASON = (
    f"The model ended the run without calling `{COMPLETE_TOOL_NAME}` or "
    f"`{FAIL_TOOL_NAME}`."
)


class Completed(BaseModel):
    """Terminal outcome for a run that delivered its result."""

    type: Literal["completed"] = "completed"
    value: SerializeAsAny[BaseModel] | None = None
    output_text: str = ""


class Failed(BaseModel):
    """Terminal outcome for a run that reported it cannot deliver."""

    type: Literal["failed"] = "failed"
    reason: str
    output_text: str = ""


RunOutcome: TypeAlias = Completed | Failed


class ResultRecorder:
    """Capture the first result reported through the injected result tools."""

    def __init__(self, result_type: type[BaseModel]) -> None:
        """Create a recorder that validates results against one schema."""

        self._result_type = result_type
        self.value: BaseModel | None = None
        self.reason: str | None = None

    @property
    def has_outcome(self) -> bool:
        """Return whether a terminal result has been recorded."""

        return self.value is not None or self.reason is not None

    def tool_definitions(self) -> tuple[ToolDefinition, ToolDefinition]:
        """Return the complete and fail tool definitions bound to this recorder."""

        complete = ToolDefinition(
            name=COMPLETE_TOOL_NAME,
            description=(
                "Report the final result and end the run. The arguments are "
                "validated against the required result schema; validation errors "
                "are returned for correction. Call this exactly once, when the "
                "task is done."
            ),
            input_schema=self._result_type.model_json_schema(),
            fn=self._complete,
        )
        fail = ToolDefinition(
            name=FAIL_TOOL_NAME,
            description=(
                "Report that the task cannot be completed and end the run. "
                "Provide a clear reason naming what is missing or impossible."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the task cannot be completed.",
                    }
                },
                "required": ["reason"],
            },
            fn=self._fail,
        )
        return complete, fail

    async def _complete(self, **arguments: JsonValue) -> ToolResult:
        """Validate and record the run's final result."""

        self._reject_repeat_result()
        self.value = self._result_type.model_validate(arguments)
        return ToolResult.text("Result recorded.")

    async def _fail(self, reason: str) -> ToolResult:
        """Record the model's reason for not delivering a result."""

        if not isinstance(reason, str):
            raise ValueError("`reason` must be a string.")
        self._reject_repeat_result()
        self.reason = reason
        return ToolResult.text("Failure recorded.")

    def _reject_repeat_result(self) -> None:
        """Refuse a second terminal result for the same run."""

        if self.has_outcome:
            raise RuntimeError("A result was already recorded for this run.")
