from typing import Literal, TypeAlias

from pydantic import BaseModel, Field


class UserMessage(BaseModel):
    """A user-authored conversation turn."""

    role: Literal["user"] = "user"
    content: str


class AssistantTextBlock(BaseModel):
    """A replayable assistant text block."""

    type: Literal["text"] = "text"
    text: str
    message_id: str | None = None
    phase: Literal["commentary", "final_answer"] | None = None


class AssistantReasoningBlock(BaseModel):
    """A replayable assistant reasoning summary block."""

    type: Literal["reasoning"] = "reasoning"
    summary_text: str
    reasoning_id: str | None = None
    encrypted_content: str | None = None


AssistantBlock: TypeAlias = AssistantTextBlock | AssistantReasoningBlock


class AssistantTurn(BaseModel):
    """A finalized assistant turn that can be replayed to a provider."""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantBlock] = Field(default_factory=list)
    response_id: str | None = None
    status: Literal["completed", "aborted", "error"] = "completed"
    error_message: str | None = None


ConversationItem: TypeAlias = UserMessage | AssistantTurn
