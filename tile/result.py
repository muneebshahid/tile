"""Output contract protocol: tool names, prompt text, and run outcomes."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, SerializeAsAny

from tile.types.tools import JsonObject

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
    """Terminal outcome for a run that delivered its result.

    ``value`` carries the validated result instance of the winning
    `complete` call, and None for runs without an output contract. An
    outcome revalidated from serialized event data carries the plain
    ``JsonObject`` form instead of the original model type.
    """

    type: Literal["completed"] = "completed"
    value: JsonObject | SerializeAsAny[BaseModel] | None = Field(
        default=None, union_mode="left_to_right"
    )
    output_text: str = ""


class Failed(BaseModel):
    """Terminal outcome for a run that reported it cannot deliver."""

    type: Literal["failed"] = "failed"
    reason: str
    output_text: str = ""


RunOutcome: TypeAlias = Completed | Failed
