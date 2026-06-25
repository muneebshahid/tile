"""Assemble normalized OpenAI events into provider stream events."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import cast

from ori.openai.normalized_events import (
    CompletedNormalizedEvent,
    CreatedNormalizedEvent,
    FailedNormalizedEvent,
    IncompleteNormalizedEvent,
    MessageAddedNormalizedEvent,
    MessageDoneNormalizedEvent,
    MessageTextDeltaNormalizedEvent,
    MessageTextPartNormalizedEvent,
    NormalizedEvent,
    NormalizedEventType,
    ReasoningDeltaNormalizedEvent,
    ReasoningDoneNormalizedEvent,
    TERMINAL_NORMALIZED_EVENT_TYPES,
    TextPartType,
    ToolCallAddedNormalizedEvent,
    ToolCallArgumentsDeltaNormalizedEvent,
    ToolCallArgumentsDoneNormalizedEvent,
    ToolCallDoneNormalizedEvent,
)
from ori.types.stream_events import (
    AssistantBlock,
    ProviderMetadata,
    ProviderSource,
    ProviderStreamEvent,
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StopReason,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)


@dataclass
class StreamAssemblyState:
    """Mutable state used while assembling one assistant stream."""

    source: ProviderSource
    response_id: str | None = None
    blocks: list[AssistantBlock] = field(default_factory=list)
    active_block: AssistantBlock | None = None
    active_block_index: int | None = None
    active_text_part_type: TextPartType | None = None


async def assemble_stream(
    normalized_stream: AsyncIterator[NormalizedEvent],
    *,
    source: ProviderSource,
) -> AsyncIterator[ProviderStreamEvent]:
    """Assemble normalized provider events into provider stream events."""

    state = StreamAssemblyState(source=source)
    async for event in normalized_stream:
        if adapted_event := _yield_stream_event(state, event):
            yield adapted_event

        if event["type"] in TERMINAL_NORMALIZED_EVENT_TYPES:
            return


def _yield_stream_event(
    state: StreamAssemblyState,
    event: NormalizedEvent,
) -> ProviderStreamEvent | None:
    """Route one normalized event to its stream-level event."""

    match event["type"]:
        case NormalizedEventType.CREATED:
            created_event = cast(CreatedNormalizedEvent, event)
            return _record_created_event(state, created_event)
        case NormalizedEventType.REASONING_ADDED:
            return _start_reasoning_block(state)
        case NormalizedEventType.REASONING_DELTA:
            reasoning_delta_event = cast(ReasoningDeltaNormalizedEvent, event)
            return _append_reasoning_delta(state, reasoning_delta_event["delta"])
        case NormalizedEventType.REASONING_DONE:
            reasoning_done_event = cast(ReasoningDoneNormalizedEvent, event)
            return _finalize_reasoning_block(state, reasoning_done_event)
        case NormalizedEventType.MESSAGE_ADDED:
            message_added_event = cast(MessageAddedNormalizedEvent, event)
            return _start_text_block(state, message_added_event)
        case NormalizedEventType.MESSAGE_TEXT_PART:
            text_part_event = cast(MessageTextPartNormalizedEvent, event)
            _activate_text_part(state, text_part_event["part_type"])
        case NormalizedEventType.MESSAGE_TEXT_DELTA:
            text_delta_event = cast(MessageTextDeltaNormalizedEvent, event)
            return _append_text_delta(state, text_delta_event)
        case NormalizedEventType.MESSAGE_DONE:
            message_done_event = cast(MessageDoneNormalizedEvent, event)
            return _finalize_text_block(state, message_done_event)
        case NormalizedEventType.TOOL_CALL_ADDED:
            tool_call_added_event = cast(ToolCallAddedNormalizedEvent, event)
            return _start_tool_call_block(state, tool_call_added_event)
        case NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA:
            arguments_delta_event = cast(ToolCallArgumentsDeltaNormalizedEvent, event)
            return _append_tool_call_arguments_delta(
                state, arguments_delta_event["delta"]
            )
        case NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE:
            arguments_done_event = cast(ToolCallArgumentsDoneNormalizedEvent, event)
            _replace_tool_call_arguments(state, arguments_done_event)
        case NormalizedEventType.TOOL_CALL_DONE:
            tool_call_done_event = cast(ToolCallDoneNormalizedEvent, event)
            return _finalize_tool_call_block(state, tool_call_done_event)
        case NormalizedEventType.COMPLETED:
            completed_event = cast(CompletedNormalizedEvent, event)
            return _build_stream_done_event(state, completed_event["stop_reason"])
        case NormalizedEventType.INCOMPLETE:
            incomplete_event = cast(IncompleteNormalizedEvent, event)
            return _build_incomplete_event(state, incomplete_event)
        case NormalizedEventType.FAILED:
            failed_event = cast(FailedNormalizedEvent, event)
            return _build_stream_error_event(state, failed_event["message"])

    return None


def _record_created_event(
    state: StreamAssemblyState,
    event: CreatedNormalizedEvent,
) -> StreamStartEvent:
    """Record provider response identity."""

    state.response_id = event["response_id"]
    return StreamStartEvent(source=state.source, response_id=state.response_id)


def _start_reasoning_block(state: StreamAssemblyState) -> ReasoningStartEvent:
    """Start a reasoning block and return its stream event."""

    block = ReasoningBlock(summary_text="")
    content_index = _append_active_block(state, block)
    return ReasoningStartEvent(content_index=content_index)


def _append_reasoning_delta(
    state: StreamAssemblyState,
    delta: str,
) -> ReasoningDeltaEvent | None:
    """Append reasoning text to the active reasoning block."""

    block = state.active_block
    if not isinstance(block, ReasoningBlock) or state.active_block_index is None:
        return None

    block.summary_text += delta
    return ReasoningDeltaEvent(content_index=state.active_block_index, delta=delta)


def _finalize_reasoning_block(
    state: StreamAssemblyState,
    event: ReasoningDoneNormalizedEvent,
) -> ReasoningEndEvent | None:
    """Finalize the active reasoning block."""

    block = state.active_block
    if not isinstance(block, ReasoningBlock) or state.active_block_index is None:
        return None

    if event["summary_text"]:
        block.summary_text = event["summary_text"]
    block.provider_metadata = ProviderMetadata.from_values(
        reasoning_signature=event["reasoning_signature"],
    )
    event_block = block.model_copy(deep=True)
    content_index = state.active_block_index
    _clear_active_block(state)
    return ReasoningEndEvent(content_index=content_index, block=event_block)


def _start_text_block(
    state: StreamAssemblyState,
    event: MessageAddedNormalizedEvent,
) -> TextStartEvent:
    """Start a text block and return its stream event."""

    block = TextBlock(text="")
    state.active_text_part_type = None
    content_index = _append_active_block(state, block)
    return TextStartEvent(content_index=content_index)


def _activate_text_part(
    state: StreamAssemblyState,
    part_type: TextPartType | None,
) -> None:
    """Track the active supported text part for subsequent deltas."""

    if isinstance(state.active_block, TextBlock):
        state.active_text_part_type = part_type


def _append_text_delta(
    state: StreamAssemblyState,
    event: MessageTextDeltaNormalizedEvent,
) -> TextDeltaEvent | None:
    """Append text to the active text block."""

    block = state.active_block
    if (
        not isinstance(block, TextBlock)
        or state.active_block_index is None
        or state.active_text_part_type != event["part_type"]
    ):
        return None

    delta = event["delta"]
    block.text += delta
    return TextDeltaEvent(content_index=state.active_block_index, delta=delta)


def _finalize_text_block(
    state: StreamAssemblyState,
    event: MessageDoneNormalizedEvent,
) -> TextEndEvent | None:
    """Finalize the active text block."""

    block = state.active_block
    if not isinstance(block, TextBlock) or state.active_block_index is None:
        return None

    block.text = event["text"]
    block.provider_metadata = ProviderMetadata.from_values(
        message_id=event["item_id"],
        phase=event["phase"],
    )
    event_block = block.model_copy(deep=True)
    content_index = state.active_block_index
    _clear_active_block(state)
    state.active_text_part_type = None
    return TextEndEvent(content_index=content_index, block=event_block)


def _start_tool_call_block(
    state: StreamAssemblyState,
    event: ToolCallAddedNormalizedEvent,
) -> ToolCallStartEvent:
    """Start tracking a tool-call block."""

    block = ToolCallBlock(
        call_id=event["call_id"],
        name=event["name"],
        arguments=event["arguments"],
    )
    state.active_text_part_type = None
    content_index = _append_active_block(state, block)
    return ToolCallStartEvent(
        content_index=content_index,
        call_id=event["call_id"],
        name=event["name"],
    )


def _append_tool_call_arguments_delta(
    state: StreamAssemblyState,
    delta: str,
) -> ToolCallDeltaEvent | None:
    """Emit a tool-call argument delta for the active tool call."""

    if not isinstance(state.active_block, ToolCallBlock):
        return None
    if state.active_block_index is None:
        return None
    return ToolCallDeltaEvent(content_index=state.active_block_index, delta=delta)


def _replace_tool_call_arguments(
    state: StreamAssemblyState,
    event: ToolCallArgumentsDoneNormalizedEvent,
) -> None:
    """Replace active tool-call arguments with the parsed final JSON."""

    if isinstance(state.active_block, ToolCallBlock):
        state.active_block.arguments = event["arguments"]


def _finalize_tool_call_block(
    state: StreamAssemblyState,
    event: ToolCallDoneNormalizedEvent,
) -> ToolCallEndEvent | None:
    """Finalize the active tool-call block."""

    block = state.active_block
    if not isinstance(block, ToolCallBlock) or state.active_block_index is None:
        return None

    block.call_id = event["call_id"]
    block.name = event["name"]
    block.arguments = event["arguments"]
    block.provider_metadata = ProviderMetadata.from_values(
        provider_item_id=event["provider_item_id"],
    )
    event_block = block.model_copy(deep=True)
    content_index = state.active_block_index
    _clear_active_block(state)
    return ToolCallEndEvent(content_index=content_index, block=event_block)


def _build_incomplete_event(
    state: StreamAssemblyState,
    event: IncompleteNormalizedEvent,
) -> StreamDoneEvent | StreamErrorEvent:
    """Map incomplete provider responses to terminal stream events."""

    if event["stop_reason"] == "error":
        return _build_stream_error_event(
            state,
            event["error_message"] or "OpenAI response incomplete.",
        )
    return _build_stream_done_event(state, event["stop_reason"])


def _build_stream_done_event(
    state: StreamAssemblyState,
    stop_reason: StopReason,
) -> StreamDoneEvent:
    """Build the successful terminal stream event."""

    return StreamDoneEvent(
        source=state.source,
        response_id=state.response_id,
        stop_reason=stop_reason,
        blocks=_copy_blocks(state.blocks),
    )


def _build_stream_error_event(
    state: StreamAssemblyState,
    error_message: str,
) -> StreamErrorEvent:
    """Build the failed terminal stream event."""

    return StreamErrorEvent(
        source=state.source,
        response_id=state.response_id,
        error_message=error_message,
        blocks=_copy_blocks(state.blocks),
    )


def _append_active_block(
    state: StreamAssemblyState,
    block: AssistantBlock,
) -> int:
    """Append a new block and make it active."""

    content_index = len(state.blocks)
    state.blocks.append(block)
    state.active_block = block
    state.active_block_index = content_index
    return content_index


def _clear_active_block(state: StreamAssemblyState) -> None:
    """Clear the active block pointer without altering accumulated blocks."""

    state.active_block = None
    state.active_block_index = None


def _copy_blocks(blocks: list[AssistantBlock]) -> list[AssistantBlock]:
    """Return an isolated deep copy of accumulated blocks."""

    return [block.model_copy(deep=True) for block in blocks]
