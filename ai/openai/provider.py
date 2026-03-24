import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
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
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
    ResponseReasoningItem,
)

from ai.types.contracts import AsyncEventStream, Reasoning as AppReasoning
from ai.openai.client import create_client
from ai.openai.serialization import serialize_history_items, serialize_tools
from ai.types.conversation import ConversationItem
from ai.types.stream import (
    AssistantMessage,
    Phase,
    ReasoningDeltaEvent,
    ReasoningBlock,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    StopReason,
    TextDeltaEvent,
    TextBlock,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ai.types.tools import JsonObject, ToolDefinition

if TYPE_CHECKING:
    from openai.types.shared_params.reasoning import Reasoning as OpenAIReasoning


SupportedTextPartType: TypeAlias = Literal["output_text", "refusal"]


@dataclass
class StreamAssemblyState:
    partial: AssistantMessage = field(default_factory=AssistantMessage)
    current_block: ReasoningBlock | TextBlock | ToolCallBlock | None = None
    active_text_part_type: SupportedTextPartType | None = None

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


async def stream(
    history: Sequence[ConversationItem],
    model: str,
    *,
    instructions: str,
    reasoning: AppReasoning | None = None,
    tools: Sequence[ToolDefinition] | None = None,
    client: AsyncOpenAI | None = None,
) -> AsyncEventStream:
    """Stream internal assistant events from the OpenAI Responses API."""

    active_client = client or create_client()
    serialized_history = serialize_history_items(history)
    request_params: ResponseCreateParamsStreaming = {
        "model": model,
        "input": serialized_history,
        "reasoning": cast("OpenAIReasoning | None", reasoning),
        "instructions": instructions,
        "stream": True,
    }
    if tools:
        request_params["tools"] = serialize_tools(tools)

    raw_stream = await active_client.responses.create(**request_params)
    return _adapt_stream(raw_stream)


async def _adapt_stream(
    raw_stream: AsyncIterator[object],
) -> AsyncIterator[StreamEvent]:
    state = StreamAssemblyState()
    yield StreamStartEvent(type="start", partial=state.partial)

    async for event in raw_stream:
        if adapted_event := _adapt_raw_event(state, event):
            yield adapted_event

        if isinstance(event, ResponseCompletedEvent | ResponseFailedEvent):
            return


def _adapt_raw_event(
    state: StreamAssemblyState,
    event: object,
) -> StreamEvent | None:
    match event:
        case ResponseCreatedEvent():
            state.partial.response_id = event.response.id
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseReasoningItem
        ):
            return _start_reasoning_block(state, event.item)
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseOutputMessage
        ):
            return _start_text_block(state, event.item)
        case ResponseOutputItemAddedEvent() if isinstance(
            event.item, ResponseFunctionToolCall
        ):
            return _start_tool_call_block(state, event.item)
        case ResponseReasoningSummaryTextDeltaEvent() if state.is_reasoning:
            return _append_reasoning_delta(state, event.delta)
        case ResponseReasoningSummaryPartDoneEvent() if state.is_reasoning:
            return _append_reasoning_delta(state, "\n\n")
        case ResponseFunctionCallArgumentsDeltaEvent() if state.is_tool_call:
            return _append_tool_call_arguments_delta(state, event)
        case ResponseFunctionCallArgumentsDoneEvent() if state.is_tool_call:
            _handle_function_tool_call_arguments_done(state, event)
        case ResponseContentPartAddedEvent() if state.is_text:
            _track_active_text_part(state, event)
        case ResponseTextDeltaEvent() if _can_append_output_text_delta(state):
            return _append_text_delta(state, event.delta)
        case ResponseRefusalDeltaEvent() if _can_append_refusal_delta(state):
            return _append_text_delta(state, event.delta)
        case ResponseOutputItemDoneEvent() if (
            isinstance(event.item, ResponseReasoningItem) and state.is_reasoning
        ):
            return _finalize_reasoning_block(state, event.item)
        case ResponseOutputItemDoneEvent() if (
            isinstance(event.item, ResponseOutputMessage) and state.is_text
        ):
            return _finalize_text_block(state, event.item)
        case ResponseOutputItemDoneEvent() if (
            isinstance(event.item, ResponseFunctionToolCall) and state.is_tool_call
        ):
            return _finalize_tool_call_block(state, event.item)
        case ResponseCompletedEvent():
            state.partial.stop_reason = _extract_stop_reason(event)
            return StreamDoneEvent(type="done", message=state.partial)
        case ResponseFailedEvent():
            return StreamErrorEvent(
                type="error",
                message=_extract_error_message(event),
                partial=state.partial,
            )

    return None


def _start_reasoning_block(
    state: StreamAssemblyState,
    item: ResponseReasoningItem,
) -> ReasoningStartEvent:
    state.current_block = ReasoningBlock(
        summary_text="",
        reasoning_id=item.id,
    )
    state.active_text_part_type = None
    state.partial.content.append(state.current_block)
    return ReasoningStartEvent(type="reasoning_start", partial=state.partial)


def _start_text_block(
    state: StreamAssemblyState,
    item: ResponseOutputMessage,
) -> TextStartEvent:
    state.current_block = TextBlock(
        text="",
        message_id=item.id,
        phase=_extract_message_phase(item),
    )
    state.active_text_part_type = None
    state.partial.content.append(state.current_block)
    return TextStartEvent(type="text_start", partial=state.partial)


def _append_reasoning_delta(
    state: StreamAssemblyState,
    delta: str,
) -> ReasoningDeltaEvent:
    state.reasoning_block.summary_text += delta
    return ReasoningDeltaEvent(
        type="reasoning_delta", delta=delta, partial=state.partial
    )


def _start_tool_call_block(
    state: StreamAssemblyState,
    item: ResponseFunctionToolCall,
) -> ToolCallStartEvent:
    state.active_text_part_type = None
    state.current_block = ToolCallBlock(
        call_id=item.call_id,
        name=item.name,
        arguments=_parse_tool_call_arguments(item.arguments or ""),
        provider_item_id=item.id,
        namespace=item.namespace,
    )
    state.partial.content.append(state.current_block)
    return ToolCallStartEvent(type="tool_call_start", partial=state.partial)


def _append_tool_call_arguments_delta(
    state: StreamAssemblyState,
    event: ResponseFunctionCallArgumentsDeltaEvent,
) -> ToolCallDeltaEvent:
    return ToolCallDeltaEvent(
        type="tool_call_delta",
        delta=event.delta,
        partial=state.partial,
    )


def _handle_function_tool_call_arguments_done(
    state: StreamAssemblyState,
    event: ResponseFunctionCallArgumentsDoneEvent,
) -> None:
    state.tool_call_block.arguments = _parse_tool_call_arguments(event.arguments)


def _track_active_text_part(
    state: StreamAssemblyState,
    event: ResponseContentPartAddedEvent,
) -> None:
    state.active_text_part_type = _extract_supported_text_part_type(event)


def _can_append_output_text_delta(state: StreamAssemblyState) -> bool:
    return state.is_text and state.active_text_part_type == "output_text"


def _can_append_refusal_delta(state: StreamAssemblyState) -> bool:
    return state.is_text and state.active_text_part_type == "refusal"


def _append_text_delta(
    state: StreamAssemblyState,
    delta: str,
) -> TextDeltaEvent:
    state.text_block.text += delta
    return TextDeltaEvent(type="text_delta", delta=delta, partial=state.partial)


def _finalize_reasoning_block(
    state: StreamAssemblyState,
    item: ResponseReasoningItem,
) -> ReasoningEndEvent:
    if summary_text := _join_reasoning_summary_text(item.summary):
        state.reasoning_block.summary_text = summary_text
    state.current_block = None
    return ReasoningEndEvent(type="reasoning_end", partial=state.partial)


def _finalize_text_block(
    state: StreamAssemblyState,
    item: ResponseOutputMessage,
) -> TextEndEvent:
    state.text_block.text = _join_message_text(item.content)
    state.text_block.message_id = item.id
    state.text_block.phase = _extract_message_phase(item)
    state.current_block = None
    state.active_text_part_type = None
    return TextEndEvent(type="text_end", partial=state.partial)


def _finalize_tool_call_block(
    state: StreamAssemblyState,
    item: ResponseFunctionToolCall,
) -> ToolCallEndEvent:
    state.tool_call_block.call_id = item.call_id
    state.tool_call_block.name = item.name
    state.tool_call_block.arguments = _parse_tool_call_arguments(item.arguments or "")
    state.tool_call_block.provider_item_id = item.id
    state.tool_call_block.namespace = item.namespace
    state.current_block = None
    return ToolCallEndEvent(type="tool_call_end", partial=state.partial)


def _extract_supported_text_part_type(
    event: ResponseContentPartAddedEvent,
) -> SupportedTextPartType | None:
    if event.part.type == "output_text":
        return "output_text"
    if event.part.type == "refusal":
        return "refusal"
    return None


def _parse_tool_call_arguments(arguments: str) -> JsonObject:
    if not arguments.strip():
        return {}

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return cast("JsonObject", parsed)
    return {}


def _extract_error_message(event: ResponseFailedEvent) -> str:
    error = getattr(event.response, "error", None)
    if error is None:
        return "OpenAI response failed."

    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message

    return "OpenAI response failed."


def _extract_stop_reason(event: ResponseCompletedEvent) -> StopReason:
    if any(
        isinstance(item, ResponseFunctionToolCall) for item in event.response.output
    ):
        return "tool_use"
    return "stop"


def _join_reasoning_summary_text(
    summary: Sequence[ResponseReasoningSummary],
) -> str:
    return "\n\n".join(item.text for item in summary if item.text)


def _join_message_text(content: Sequence[ResponseMessageContent]) -> str:
    parts: list[str] = []
    for item in content:
        if isinstance(item, ResponseOutputText):
            parts.append(item.text)
        elif isinstance(item, ResponseOutputRefusal):
            parts.append(item.refusal)
    return "".join(parts)


def _extract_message_phase(
    item: ResponseOutputMessage,
) -> Phase | None:
    phase = getattr(item, "phase", None)
    if phase in {"commentary", "final_answer"}:
        return phase
    return None
