"""Provider stream helpers for stateless agent tests."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from agent.types import StreamFn
from ai.types.contracts import Reasoning
from ai.types.conversation import ConversationItem
from ai.types.stream_events import (
    AssistantBlock,
    ProviderMetadata,
    ProviderSource,
    ProviderStreamEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
    StopReason,
    TextBlock,
    ToolCallBlock,
)
from ai.types.tools import JsonObject, ToolDefinition
from tests.support.async_streams import async_stream


@dataclass
class StreamInvocation:
    """Captured arguments from one provider stream invocation."""

    history: tuple[ConversationItem, ...]
    model: str
    instructions: str
    reasoning: Reasoning | None
    tools: tuple[ToolDefinition, ...] | None


def build_stream_fn(
    streams: Sequence[Sequence[ProviderStreamEvent]],
    invocations: list[StreamInvocation],
) -> StreamFn:
    """Build a provider stream function that records each invocation."""

    pending_streams = list(streams)

    async def _stream_fn(
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
        tools: Sequence[ToolDefinition] | None,
    ) -> AsyncIterator[ProviderStreamEvent]:
        """Return the next queued provider event stream."""

        invocations.append(
            StreamInvocation(
                history=tuple(history),
                model=model,
                instructions=instructions,
                reasoning=reasoning,
                tools=tuple(tools) if tools is not None else None,
            )
        )
        assert pending_streams, (
            "Provider stream invoked more times than queued test streams."
        )
        return async_stream(pending_streams.pop(0))

    return _stream_fn


def final_text_stream(response_id: str, text: str) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that completes with final text."""

    return [
        stream_start(response_id),
        stream_done(response_id, blocks=[TextBlock(text=text)]),
    ]


def tool_call_stream(
    *,
    response_id: str,
    call_id: str,
    tool_name: str,
    arguments: JsonObject,
    provider_item_id: str | None = None,
) -> list[ProviderStreamEvent]:
    """Build a minimal provider stream that requests one tool call."""

    return [
        stream_start(response_id),
        stream_done(
            response_id,
            stop_reason="tool_use",
            blocks=[
                tool_call_block(
                    call_id=call_id,
                    name=tool_name,
                    arguments=arguments,
                    provider_item_id=provider_item_id,
                )
            ],
        ),
    ]


def stream_start(response_id: str) -> StreamStartEvent:
    """Build a deterministic provider stream start event."""

    return StreamStartEvent(source=provider_source(), response_id=response_id)


def stream_done(
    response_id: str,
    *,
    stop_reason: StopReason = "stop",
    blocks: Sequence[AssistantBlock] = (),
) -> StreamDoneEvent:
    """Build a deterministic provider stream completion event."""

    return StreamDoneEvent(
        source=provider_source(),
        response_id=response_id,
        stop_reason=stop_reason,
        blocks=list(blocks),
    )


def stream_error(response_id: str, error_message: str) -> StreamErrorEvent:
    """Build a deterministic provider stream error event."""

    return StreamErrorEvent(
        source=provider_source(),
        response_id=response_id,
        error_message=error_message,
    )


def tool_call_block(
    *,
    call_id: str,
    name: str,
    arguments: JsonObject,
    provider_item_id: str | None = None,
) -> ToolCallBlock:
    """Build a tool-call block with optional provider replay metadata."""

    return ToolCallBlock(
        call_id=call_id,
        name=name,
        arguments=arguments,
        provider_metadata=ProviderMetadata.from_values(
            provider_item_id=provider_item_id
        ),
    )


def provider_source() -> ProviderSource:
    """Build the deterministic provider source used by agent tests."""

    return ProviderSource(provider="test", model="gpt-5.4")
