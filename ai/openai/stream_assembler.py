from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import cast

from ai.openai.wire_events import (
    ResponseCompletedWireEvent,
    ResponseCreatedWireEvent,
    ResponseFailedWireEvent,
    ResponseIncompleteWireEvent,
    ResponseMessageAddedWireEvent,
    ResponseMessageDoneWireEvent,
    ResponseMessageTextDeltaWireEvent,
    ResponseMessageTextPartWireEvent,
    ResponseReasoningDeltaWireEvent,
    ResponseReasoningDoneWireEvent,
    ResponseToolCallAddedWireEvent,
    ResponseToolCallArgumentsDeltaWireEvent,
    ResponseToolCallArgumentsDoneWireEvent,
    ResponseToolCallDoneWireEvent,
    TextPartType,
    WireEvent,
    WireEventType,
)
from ai.types.stream import (
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

TERMINAL_WIRE_EVENT_TYPES: frozenset[WireEventType] = frozenset(
    {
        WireEventType.RESPONSE_COMPLETED,
        WireEventType.RESPONSE_INCOMPLETE,
        WireEventType.RESPONSE_FAILED,
    }
)


@dataclass
class StreamAssemblyState:
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    current_block: ReasoningBlock | TextBlock | ToolCallBlock | None = None
    active_text_part_type: TextPartType | None = None

    @property
    def is_reasoning(self) -> bool:
        return isinstance(self.current_block, ReasoningBlock)

    @property
    def is_text(self) -> bool:
        return isinstance(self.current_block, TextBlock)

    @property
    def is_tool_call(self) -> bool:
        return isinstance(self.current_block, ToolCallBlock)

    @property
    def reasoning_block(self) -> ReasoningBlock:
        assert isinstance(self.current_block, ReasoningBlock)
        return self.current_block

    @property
    def text_block(self) -> TextBlock:
        assert isinstance(self.current_block, TextBlock)
        return self.current_block

    @property
    def tool_call_block(self) -> ToolCallBlock:
        assert isinstance(self.current_block, ToolCallBlock)
        return self.current_block


async def assemble_stream(
    raw_stream: AsyncIterator[WireEvent],
) -> AsyncIterator[StreamEvent]:
    state = StreamAssemblyState()
    yield StreamStartEvent(type="start", partial=state.partial)

    async for event in raw_stream:
        if adapted_event := _yield_stream_event(state, event):
            yield adapted_event

        if event["type"] in TERMINAL_WIRE_EVENT_TYPES:
            return


def _yield_stream_event(
    state: StreamAssemblyState,
    event: WireEvent,
) -> StreamEvent | None:
    match event["type"]:
        case WireEventType.RESPONSE_CREATED:
            created_event = cast(ResponseCreatedWireEvent, event)
            state.partial.response_id = created_event["response_id"]

        case WireEventType.RESPONSE_REASONING_ADDED:
            return _start_reasoning_block(state)

        case WireEventType.RESPONSE_REASONING_DELTA if state.is_reasoning:
            reasoning_delta_event = cast(ResponseReasoningDeltaWireEvent, event)
            return _append_reasoning_delta(state, reasoning_delta_event["delta"])

        case WireEventType.RESPONSE_REASONING_DONE if state.is_reasoning:
            reasoning_done_event = cast(ResponseReasoningDoneWireEvent, event)
            return _finalize_reasoning_block(state, reasoning_done_event)

        case WireEventType.RESPONSE_MESSAGE_ADDED:
            message_added_event = cast(ResponseMessageAddedWireEvent, event)
            return _start_text_block(state, message_added_event)

        case WireEventType.RESPONSE_MESSAGE_TEXT_PART if state.is_text:
            text_part_event = cast(ResponseMessageTextPartWireEvent, event)
            state.active_text_part_type = text_part_event["part_type"]

        case WireEventType.RESPONSE_MESSAGE_TEXT_DELTA:
            text_delta_event = cast(ResponseMessageTextDeltaWireEvent, event)
            if _can_append_text_delta(state, text_delta_event["part_type"]):
                return _append_text_delta(state, text_delta_event["delta"])

        case WireEventType.RESPONSE_MESSAGE_DONE if state.is_text:
            message_done_event = cast(ResponseMessageDoneWireEvent, event)
            return _finalize_text_block(state, message_done_event)

        case WireEventType.RESPONSE_TOOL_CALL_ADDED:
            tool_call_added_event = cast(ResponseToolCallAddedWireEvent, event)
            return _start_tool_call_block(state, tool_call_added_event)

        case WireEventType.RESPONSE_TOOL_CALL_ARGUMENTS_DELTA if state.is_tool_call:
            arguments_delta_event = cast(ResponseToolCallArgumentsDeltaWireEvent, event)
            return _append_tool_call_arguments_delta(
                state, arguments_delta_event["delta"]
            )

        case WireEventType.RESPONSE_TOOL_CALL_ARGUMENTS_DONE if state.is_tool_call:
            arguments_done_event = cast(ResponseToolCallArgumentsDoneWireEvent, event)
            state.tool_call_block.arguments = arguments_done_event["arguments"]

        case WireEventType.RESPONSE_TOOL_CALL_DONE if state.is_tool_call:
            tool_call_done_event = cast(ResponseToolCallDoneWireEvent, event)
            return _finalize_tool_call_block(state, tool_call_done_event)

        case WireEventType.RESPONSE_COMPLETED:
            completed_event = cast(ResponseCompletedWireEvent, event)
            return _build_stream_done_event(state, completed_event["stop_reason"])

        case WireEventType.RESPONSE_INCOMPLETE:
            incomplete_event = cast(ResponseIncompleteWireEvent, event)
            if incomplete_event["stop_reason"] == "error":
                return _build_stream_error_event(
                    state,
                    incomplete_event["error_message"] or "OpenAI response incomplete.",
                )
            return _build_stream_done_event(state, incomplete_event["stop_reason"])

        case WireEventType.RESPONSE_FAILED:
            failed_event = cast(ResponseFailedWireEvent, event)
            return _build_stream_error_event(state, failed_event["message"])

    return None


def _start_reasoning_block(state: StreamAssemblyState) -> ReasoningStartEvent:
    state.current_block = ReasoningBlock(summary_text="")
    state.active_text_part_type = None
    state.partial.content.append(state.current_block)
    return ReasoningStartEvent(type="reasoning_start", partial=state.partial)


def _append_reasoning_delta(
    state: StreamAssemblyState,
    delta: str,
) -> ReasoningDeltaEvent:
    state.reasoning_block.summary_text += delta
    return ReasoningDeltaEvent(
        type="reasoning_delta",
        delta=delta,
        partial=state.partial,
    )


def _finalize_reasoning_block(
    state: StreamAssemblyState,
    event: ResponseReasoningDoneWireEvent,
) -> ReasoningEndEvent:
    if event["summary_text"]:
        state.reasoning_block.summary_text = event["summary_text"]
    state.reasoning_block.reasoning_signature = event["reasoning_signature"]
    state.current_block = None
    return ReasoningEndEvent(type="reasoning_end", partial=state.partial)


def _start_text_block(
    state: StreamAssemblyState,
    event: ResponseMessageAddedWireEvent,
) -> TextStartEvent:
    state.current_block = TextBlock(
        text="",
        message_id=event["item_id"],
        phase=event["phase"],
    )
    state.active_text_part_type = None
    state.partial.content.append(state.current_block)
    return TextStartEvent(type="text_start", partial=state.partial)


def _can_append_text_delta(
    state: StreamAssemblyState,
    part_type: TextPartType,
) -> bool:
    return state.is_text and state.active_text_part_type == part_type


def _append_text_delta(
    state: StreamAssemblyState,
    delta: str,
) -> TextDeltaEvent:
    state.text_block.text += delta
    return TextDeltaEvent(type="text_delta", delta=delta, partial=state.partial)


def _finalize_text_block(
    state: StreamAssemblyState,
    event: ResponseMessageDoneWireEvent,
) -> TextEndEvent:
    state.text_block.text = event["text"]
    state.text_block.message_id = event["item_id"]
    state.text_block.phase = event["phase"]
    state.current_block = None
    state.active_text_part_type = None
    return TextEndEvent(type="text_end", partial=state.partial)


def _start_tool_call_block(
    state: StreamAssemblyState,
    event: ResponseToolCallAddedWireEvent,
) -> ToolCallStartEvent:
    state.active_text_part_type = None
    state.current_block = ToolCallBlock(
        call_id=event["call_id"],
        name=event["name"],
        arguments=event["arguments"],
        provider_item_id=event["provider_item_id"],
        namespace=event["namespace"],
    )
    state.partial.content.append(state.current_block)
    return ToolCallStartEvent(type="tool_call_start", partial=state.partial)


def _append_tool_call_arguments_delta(
    state: StreamAssemblyState,
    delta: str,
) -> ToolCallDeltaEvent:
    return ToolCallDeltaEvent(
        type="tool_call_delta",
        delta=delta,
        partial=state.partial,
    )


def _finalize_tool_call_block(
    state: StreamAssemblyState,
    event: ResponseToolCallDoneWireEvent,
) -> ToolCallEndEvent:
    state.tool_call_block.call_id = event["call_id"]
    state.tool_call_block.name = event["name"]
    state.tool_call_block.arguments = event["arguments"]
    state.tool_call_block.provider_item_id = event["provider_item_id"]
    state.tool_call_block.namespace = event["namespace"]
    state.current_block = None
    return ToolCallEndEvent(type="tool_call_end", partial=state.partial)


def _build_stream_done_event(
    state: StreamAssemblyState,
    stop_reason: StopReason,
) -> StreamDoneEvent:
    state.partial.stop_reason = stop_reason
    return StreamDoneEvent(type="done", message=state.partial)


def _build_stream_error_event(
    state: StreamAssemblyState,
    error_message: str,
) -> StreamErrorEvent:
    state.partial.stop_reason = "error"
    state.partial.error_message = error_message
    return StreamErrorEvent(type="error", error=state.partial)
