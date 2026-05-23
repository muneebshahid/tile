from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

from ai.types.stream import AssistantBlock, StopReason


class UserMessage(BaseModel):
    """A user-authored conversation turn."""

    role: Literal["user"] = "user"
    content: str


class AssistantTurn(BaseModel):
    """A finalized assistant turn that can be replayed to a provider."""

    role: Literal["assistant"] = "assistant"
    blocks: list[AssistantBlock] = Field(default_factory=list)
    response_id: str | None = None
    stop_reason: StopReason = "stop"
    status: Literal["completed", "aborted", "error"] = "completed"
    error_message: str | None = None


class ToolResultTurn(BaseModel):
    """A replayable tool result tied to a prior assistant tool call."""

    role: Literal["tool_result"] = "tool_result"
    call_id: str
    tool_name: str
    content: str
    is_error: bool = False


ConversationItem: TypeAlias = UserMessage | AssistantTurn | ToolResultTurn
