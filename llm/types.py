from typing import Literal

from pydantic import BaseModel


class ResponseCreatedEvent(BaseModel):
    """Emitted when a response is first created and assigned an ID."""

    type: Literal["response.created"]
    sequence_number: int
    response: "ResponseInfo"


class ResponseInProgressEvent(BaseModel):
    """Emitted while a response is actively being processed by the model."""

    type: Literal["response.in_progress"]
    sequence_number: int
    response: "ResponseInfo"


class ResponseOutputItemAddedEvent(BaseModel):
    """Emitted when a new output item, like text, reasoning, or a tool call, starts streaming."""

    type: Literal["response.output_item.added"]
    sequence_number: int
    output_index: int
    item: "ResponseOutputMessageItem | ResponseReasoningItem | ResponseFunctionToolCallItem"


class ResponseInfo(BaseModel):
    id: str
    model: str
    status: str


class ResponseOutputMessageItem(BaseModel):
    id: str
    type: Literal["message"]
    status: str | None = None


class ResponseReasoningItem(BaseModel):
    id: str
    type: Literal["reasoning"]
    status: str | None = None


class ResponseFunctionToolCallItem(BaseModel):
    id: str
    type: Literal["function_call"]
    status: str | None = None
    call_id: str
    name: str


ResponseCreatedEvent.model_rebuild()
ResponseInProgressEvent.model_rebuild()
ResponseOutputItemAddedEvent.model_rebuild()
