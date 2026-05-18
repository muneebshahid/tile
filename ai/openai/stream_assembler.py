from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import cast

from ai.openai.normalized_events import (
    CompletedNormalizedEvent,
    CreatedNormalizedEvent,
    FailedNormalizedEvent,
    IncompleteNormalizedEvent,
    MessageAddedNormalizedEvent,
    MessageDoneNormalizedEvent,
    MessageTextDeltaNormalizedEvent,
    MessageTextPartNormalizedEvent,
    ReasoningDeltaNormalizedEvent,
    ReasoningDoneNormalizedEvent,
    NormalizedEvent,
    NormalizedEventType,
    TextPartType,
    ToolCallAddedNormalizedEvent,
    ToolCallArgumentsDeltaNormalizedEvent,
    ToolCallArgumentsDoneNormalizedEvent,
    ToolCallDoneNormalizedEvent,
)
from ai.types.stream import (
    AssistantBlock,
    AssistantMessage,
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    StopReason,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

TERMINAL_NORMALIZED_EVENT_TYPES: frozenset[NormalizedEventType] = frozenset(
    {
        NormalizedEventType.COMPLETED,
        NormalizedEventType.INCOMPLETE,
        NormalizedEventType.FAILED,
    }
)


@dataclass
class StreamAssemblyState:
    """Mutable state used while assembling one assistant stream."""

    message: AssistantMessage = field(default_factory=AssistantMessage)
    active_block: AssistantBlock | None = None
    active_text_part_type: TextPartType | None = None


async def assemble_stream(
    normalized_stream: AsyncIterator[NormalizedEvent],
) -> AsyncIterator[StreamEvent]:
    """Assemble normalized provider events into app-level stream events."""

    state = StreamAssemblyState()
    yield StreamStartEvent(type="start", message=state.message)

    async for event in normalized_stream:
        if adapted_event := _yield_stream_event(state, event):
            yield adapted_event

        if event["type"] in TERMINAL_NORMALIZED_EVENT_TYPES:
            return


def _yield_stream_event(
    state: StreamAssemblyState,
    event: NormalizedEvent,
) -> StreamEvent | None:
    match event["type"]:
        case NormalizedEventType.CREATED:
            created_event = cast(CreatedNormalizedEvent, event)
            state.message.response_id = created_event["response_id"]

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
                state,
                arguments_delta_event["delta"],
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
            if incomplete_event["stop_reason"] == "error":
                return _build_stream_error_event(
                    state,
                    incomplete_event["error_message"] or "OpenAI response incomplete.",
                )
            return _build_stream_done_event(state, incomplete_event["stop_reason"])

        case NormalizedEventType.FAILED:
            failed_event = cast(FailedNormalizedEvent, event)
            return _build_stream_error_event(state, failed_event["message"])

    return None


def _start_reasoning_block(state: StreamAssemblyState) -> ReasoningStartEvent:
    reasoning_block = ReasoningBlock(summary_text="")
    state.active_block = reasoning_block
    state.active_text_part_type = None
    state.message.blocks.append(reasoning_block)
    return ReasoningStartEvent(type="reasoning_start", message=state.message)


def _append_reasoning_delta(
    state: StreamAssemblyState,
    delta: str,
) -> ReasoningDeltaEvent | None:
    if not isinstance(state.active_block, ReasoningBlock):
        return None

    state.active_block.summary_text += delta
    return ReasoningDeltaEvent(
        type="reasoning_delta",
        delta=delta,
        message=state.message,
    )


def _finalize_reasoning_block(
    state: StreamAssemblyState,
    event: ReasoningDoneNormalizedEvent,
) -> ReasoningEndEvent | None:
    if not isinstance(state.active_block, ReasoningBlock):
        return None

    if event["summary_text"]:
        state.active_block.summary_text = event["summary_text"]
    state.active_block.reasoning_signature = event["reasoning_signature"]
    state.active_block = None
    return ReasoningEndEvent(type="reasoning_end", message=state.message)


def _start_text_block(
    state: StreamAssemblyState,
    event: MessageAddedNormalizedEvent,
) -> TextStartEvent:
    text_block = TextBlock(
        text="",
        message_id=event["item_id"],
        phase=event["phase"],
    )
    state.active_block = text_block
    state.active_text_part_type = None
    state.message.blocks.append(text_block)
    return TextStartEvent(type="text_start", message=state.message)


def _activate_text_part(
    state: StreamAssemblyState,
    part_type: TextPartType | None,
) -> None:
    if isinstance(state.active_block, TextBlock):
        state.active_text_part_type = part_type


def _append_text_delta(
    state: StreamAssemblyState,
    event: MessageTextDeltaNormalizedEvent,
) -> TextDeltaEvent | None:
    if (
        not isinstance(state.active_block, TextBlock)
        or state.active_text_part_type != event["part_type"]
    ):
        return None

    delta = event["delta"]
    state.active_block.text += delta
    return TextDeltaEvent(type="text_delta", delta=delta, message=state.message)


def _finalize_text_block(
    state: StreamAssemblyState,
    event: MessageDoneNormalizedEvent,
) -> TextEndEvent | None:
    if not isinstance(state.active_block, TextBlock):
        return None

    state.active_block.text = event["text"]
    state.active_block.message_id = event["item_id"]
    state.active_block.phase = event["phase"]
    state.active_block = None
    state.active_text_part_type = None
    return TextEndEvent(type="text_end", message=state.message)


def _start_tool_call_block(
    state: StreamAssemblyState,
    event: ToolCallAddedNormalizedEvent,
) -> ToolCallStartEvent:
    state.active_text_part_type = None
    tool_call_block = ToolCallBlock(
        call_id=event["call_id"],
        name=event["name"],
        arguments=event["arguments"],
        provider_item_id=event["provider_item_id"],
    )
    state.active_block = tool_call_block
    state.message.blocks.append(tool_call_block)
    return ToolCallStartEvent(type="tool_call_start", message=state.message)


def _append_tool_call_arguments_delta(
    state: StreamAssemblyState,
    delta: str,
) -> ToolCallDeltaEvent | None:
    if not isinstance(state.active_block, ToolCallBlock):
        return None

    return ToolCallDeltaEvent(
        type="tool_call_delta",
        delta=delta,
        message=state.message,
    )


def _replace_tool_call_arguments(
    state: StreamAssemblyState,
    event: ToolCallArgumentsDoneNormalizedEvent,
) -> None:
    if isinstance(state.active_block, ToolCallBlock):
        state.active_block.arguments = event["arguments"]


def _finalize_tool_call_block(
    state: StreamAssemblyState,
    event: ToolCallDoneNormalizedEvent,
) -> ToolCallEndEvent | None:
    if not isinstance(state.active_block, ToolCallBlock):
        return None

    state.active_block.call_id = event["call_id"]
    state.active_block.name = event["name"]
    state.active_block.arguments = event["arguments"]
    state.active_block.provider_item_id = event["provider_item_id"]
    state.active_block = None
    return ToolCallEndEvent(type="tool_call_end", message=state.message)


def _build_stream_done_event(
    state: StreamAssemblyState,
    stop_reason: StopReason,
) -> StreamDoneEvent:
    state.message.stop_reason = stop_reason
    return StreamDoneEvent(type="done", message=state.message)


def _build_stream_error_event(
    state: StreamAssemblyState,
    error_message: str,
) -> StreamErrorEvent:
    state.message.stop_reason = "error"
    state.message.error_message = error_message
    return StreamErrorEvent(type="error", error=state.message)
