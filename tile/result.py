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

    ``value`` carries assistant text for plain prompts and the validated result
    instance for result prompts. A serialized result outcome revalidates into
    its plain ``JsonObject`` form instead of the original model type.
    """

    type: Literal["completed"] = "completed"
    value: str | JsonObject | SerializeAsAny[BaseModel] = Field(
        union_mode="left_to_right"
    )


ExecutionFailureOrigin: TypeAlias = Literal["submission", "turn", "execution"]


class AgentFailure(BaseModel):
    """The agent declared it could not deliver the requested result.

    Execution finished normally, so the run's status stays ``completed``;
    this cause records the model's own verdict that the task failed.
    """

    type: Literal["agent_failure"] = "agent_failure"
    reason: str


class ExecutionFailure(BaseModel):
    """Serializable diagnostics for a run whose execution failed.

    ``origin`` names the runtime boundary that failed. The original
    in-process exception stays available on the run handle for local
    debugging; it is not part of the serialized contract.
    """

    type: Literal["execution_failure"] = "execution_failure"
    origin: ExecutionFailureOrigin
    exception_type: str
    message: str


FailureCause: TypeAlias = AgentFailure | ExecutionFailure


class Failed(BaseModel):
    """Terminal outcome for a run that could not deliver its result.

    The structured ``cause`` distinguishes an agent-declared failure from
    an execution failure instead of overloading optional fields.
    """

    type: Literal["failed"] = "failed"
    cause: FailureCause = Field(discriminator="type")


class Aborted(BaseModel):
    """Terminal outcome for a run cancelled before it reached a verdict."""

    type: Literal["aborted"] = "aborted"


RunOutcome: TypeAlias = Completed | Failed | Aborted
