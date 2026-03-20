from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_message import (
    Content as ResponseMessageContent,
    ResponseOutputMessage,
)
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
    ResponseReasoningItem,
)

from ai.openai_client import create_openai_client
from ai.contracts import AsyncEventStream, Reasoning as AppReasoning
from ai.types import (
    AssistantMessage,
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)

if TYPE_CHECKING:
    from openai.types.shared_params.reasoning import Reasoning as OpenAIReasoning


@dataclass
class StreamAssemblyState:
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    current_reasoning_block: ReasoningBlock | None = None
    current_text_block: TextBlock | None = None
    current_text_content_part: Literal["output_text", "refusal"] | None = None


async def stream(
    prompt: str,
    model: str,
    reasoning: AppReasoning | None = None,
    *,
    client: AsyncOpenAI | None = None,
) -> AsyncEventStream:
    """Stream internal assistant events from the OpenAI Responses API."""

    active_client = client or create_openai_client()
    raw_stream = await active_client.responses.create(
        model=model,
        input=prompt,
        stream=True,
        reasoning=cast("OpenAIReasoning | None", reasoning),
    )
    return _adapt_stream(raw_stream)


async def _adapt_stream(
    raw_stream: AsyncIterator[object],
) -> AsyncIterator[StreamEvent]:
    state = StreamAssemblyState()
    yield StreamStartEvent(type="start", partial=state.partial)

    async for event in raw_stream:
        for adapted_event in _adapt_raw_event(state, event):
            yield adapted_event

        if isinstance(event, ResponseCompletedEvent | ResponseFailedEvent):
            return


def _adapt_raw_event(
    state: StreamAssemblyState,
    event: object,
) -> Iterator[StreamEvent]:
    match event:
        # Start of a new response
        case ResponseCreatedEvent():
            state.partial.response_id = event.response.id

        # Reasoning block started
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseReasoningItem
        ):
            yield _start_reasoning_block(state, event.item)

        # Text block started
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseOutputMessage
        ):
            yield _start_text_block(state)

        # Reasoning summary text delta
        case ResponseReasoningSummaryTextDeltaEvent() if (
            state.current_reasoning_block is not None
        ):
            yield _append_reasoning_delta(state, event.delta)

        # Separate summary parts with a blank line while streaming
        case ResponseReasoningSummaryPartDoneEvent() if (
            state.current_reasoning_block is not None
        ):
            yield _append_reasoning_delta(state, "\n\n")

        # Track the active visible content part for message output
        case ResponseContentPartAddedEvent() if state.current_text_block is not None:
            _update_text_content_part(state, event)

        # Text and refusal deltas are surfaced through the same text block stream
        case ResponseTextDeltaEvent() | ResponseRefusalDeltaEvent() if (
            state.current_text_block is not None
            and state.current_text_content_part in {"output_text", "refusal"}
        ):
            yield _append_text_delta(state, event.delta)

        # Reasoning block finalized with the canonical item shape
        case ResponseOutputItemDoneEvent() if (
            isinstance(event.item, ResponseReasoningItem)
            and state.current_reasoning_block is not None
        ):
            yield _finalize_reasoning_block(state, event.item)

        # Message block finalized with the canonical item shape
        case ResponseOutputItemDoneEvent() if (
            isinstance(event.item, ResponseOutputMessage)
            and state.current_text_block is not None
        ):
            yield _finalize_text_block(state, event.item)

        # Response completed successfully
        case ResponseCompletedEvent():
            yield StreamDoneEvent(
                type="done", message=state.partial.model_copy(deep=True)
            )

        # Response failed
        case ResponseFailedEvent():
            yield StreamErrorEvent(
                type="error",
                message=_extract_error_message(event),
                partial=state.partial.model_copy(deep=True),
            )


def _start_reasoning_block(
    state: StreamAssemblyState,
    item: ResponseReasoningItem,
) -> ReasoningStartEvent:
    state.current_reasoning_block = ReasoningBlock(
        type="reasoning",
        reasoning="",
        reasoning_id=item.id,
    )
    state.current_text_block = None
    state.current_text_content_part = None
    state.partial.content.append(state.current_reasoning_block)
    return ReasoningStartEvent(type="reasoning_start", partial=state.partial)


def _start_text_block(state: StreamAssemblyState) -> TextStartEvent:
    state.current_text_block = TextBlock(type="text", text="")
    state.current_reasoning_block = None
    state.current_text_content_part = None
    state.partial.content.append(state.current_text_block)
    return TextStartEvent(type="text_start", partial=state.partial)


def _append_reasoning_delta(
    state: StreamAssemblyState,
    delta: str,
) -> ReasoningDeltaEvent:
    assert state.current_reasoning_block is not None
    state.current_reasoning_block.reasoning += delta
    return ReasoningDeltaEvent(
        type="reasoning_delta", delta=delta, partial=state.partial
    )


def _update_text_content_part(
    state: StreamAssemblyState,
    event: ResponseContentPartAddedEvent,
) -> None:
    assert state.current_text_block is not None
    if event.part.type == "output_text":
        state.current_text_content_part = "output_text"
    elif event.part.type == "refusal":
        state.current_text_content_part = "refusal"
    else:
        state.current_text_content_part = None


def _append_text_delta(
    state: StreamAssemblyState,
    delta: str,
) -> TextDeltaEvent:
    assert state.current_text_block is not None
    state.current_text_block.text += delta
    return TextDeltaEvent(type="text_delta", delta=delta, partial=state.partial)


def _finalize_reasoning_block(
    state: StreamAssemblyState,
    item: ResponseReasoningItem,
) -> ReasoningEndEvent:
    assert state.current_reasoning_block is not None
    if summary_text := _join_reasoning_summary_text(item.summary):
        state.current_reasoning_block.reasoning = summary_text
    state.current_reasoning_block = None
    return ReasoningEndEvent(type="reasoning_end", partial=state.partial)


def _finalize_text_block(
    state: StreamAssemblyState,
    item: ResponseOutputMessage,
) -> TextEndEvent:
    assert state.current_text_block is not None
    state.current_text_block.text = _join_message_text(item.content)
    state.current_text_block = None
    state.current_text_content_part = None
    return TextEndEvent(type="text_end", partial=state.partial)


def _extract_error_message(event: ResponseFailedEvent) -> str:
    error = getattr(event.response, "error", None)
    if error is None:
        return "OpenAI response failed."

    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message

    return "OpenAI response failed."


def _join_reasoning_summary_text(
    summary: Sequence[ResponseReasoningSummary],
) -> str:
    return "\n\n".join(item.text for item in summary if item.text)


def _join_message_text(content: Sequence[ResponseMessageContent]) -> str:
    parts = []
    for item in content:
        if isinstance(item, ResponseOutputText):
            parts.append(item.text)
        elif isinstance(item, ResponseOutputRefusal):
            parts.append(item.refusal)
    return "".join(parts)
