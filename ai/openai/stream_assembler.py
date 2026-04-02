from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ai.openai.wire_events import TextPartType, WireEvent
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
        if adapted_event := _apply_wire_event(state, event):
            yield adapted_event

        if event["type"] in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            return


def _apply_wire_event(
    state: StreamAssemblyState,
    event: WireEvent,
) -> StreamEvent | None:
    match event["type"]:
        case "response.created":
            state.partial.response_id = event["response_id"]
        case "response.reasoning.added":
            return _start_reasoning_block(state)
        case "response.reasoning.delta" if state.is_reasoning:
            return _append_reasoning_delta(state, event["delta"])
        case "response.reasoning.done" if state.is_reasoning:
            return _finalize_reasoning_block(state, event)
        case "response.message.added":
            return _start_text_block(state, event)
        case "response.message.text_part" if state.is_text:
            state.active_text_part_type = event["part_type"]
        case "response.message.text.delta" if _can_append_text_delta(
            state, event["part_type"]
        ):
            return _append_text_delta(state, event["delta"])
        case "response.message.done" if state.is_text:
            return _finalize_text_block(state, event)
        case "response.tool_call.added":
            return _start_tool_call_block(state, event)
        case "response.tool_call.arguments.delta" if state.is_tool_call:
            return _append_tool_call_arguments_delta(state, event["delta"])
        case "response.tool_call.arguments.done" if state.is_tool_call:
            state.tool_call_block.arguments = event["arguments"]
        case "response.tool_call.done" if state.is_tool_call:
            return _finalize_tool_call_block(state, event)
        case "response.completed":
            return _build_stream_done_event(state, event["stop_reason"])
        case "response.incomplete":
            if event["stop_reason"] == "error":
                return _build_stream_error_event(
                    state,
                    event["error_message"] or "OpenAI response incomplete.",
                )
            return _build_stream_done_event(state, event["stop_reason"])
        case "response.failed":
            return _build_stream_error_event(state, event["message"])

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
    event: WireEvent,
) -> ReasoningEndEvent:
    assert event["type"] == "response.reasoning.done"
    state.reasoning_block.summary_text = event["summary_text"]
    state.reasoning_block.reasoning_signature = event["reasoning_signature"]
    state.current_block = None
    return ReasoningEndEvent(type="reasoning_end", partial=state.partial)


def _start_text_block(
    state: StreamAssemblyState,
    event: WireEvent,
) -> TextStartEvent:
    assert event["type"] == "response.message.added"
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
    event: WireEvent,
) -> TextEndEvent:
    assert event["type"] == "response.message.done"
    state.text_block.text = event["text"]
    state.text_block.message_id = event["item_id"]
    state.text_block.phase = event["phase"]
    state.current_block = None
    state.active_text_part_type = None
    return TextEndEvent(type="text_end", partial=state.partial)


def _start_tool_call_block(
    state: StreamAssemblyState,
    event: WireEvent,
) -> ToolCallStartEvent:
    assert event["type"] == "response.tool_call.added"
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
    event: WireEvent,
) -> ToolCallEndEvent:
    assert event["type"] == "response.tool_call.done"
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
