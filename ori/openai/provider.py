"""OpenAI provider entrypoints split by transport."""

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from openai.types.responses.response_stream_event import ResponseStreamEvent

from ori.openai.client import create_client
from ori.openai.serialization import serialize_history_items, serialize_tools
from ori.openai.sdk_event_adapter import normalize_sdk_events
from ori.openai.stream_assembler import assemble_stream
from ori.openai.subscription_event_adapter import (
    SubscriptionEventPayload,
    normalize_subscription_events,
)
from ori.types.contracts import AsyncEventStream, Reasoning
from ori.types.conversation import ConversationItem
from ori.types.stream_events import ProviderSource
from ori.types.tools import ToolDefinition

if TYPE_CHECKING:
    from openai.types.shared_params.reasoning import Reasoning as OpenAIReasoning


async def stream_api(
    history: Sequence[ConversationItem],
    model: str,
    *,
    instructions: str,
    reasoning: Reasoning | None = None,
    tools: Sequence[ToolDefinition] | None = None,
) -> AsyncEventStream:
    """Stream assistant events through the OpenAI SDK transport."""

    request_params = _build_stream_request_params(
        history,
        model,
        instructions=instructions,
        reasoning=reasoning,
        tools=tools,
    )
    raw_stream = await _create_api_stream(create_client(), request_params)
    return assemble_stream(
        normalize_sdk_events(raw_stream),
        source=ProviderSource(provider="openai", model=model),
    )


async def stream_subscription(
    history: Sequence[ConversationItem],
    model: str,
    *,
    instructions: str,
    reasoning: Reasoning | None = None,
    tools: Sequence[ToolDefinition] | None = None,
    raw_stream: AsyncIterator[SubscriptionEventPayload] | None = None,
) -> AsyncEventStream:
    """Stream assistant events through the subscription SSE transport."""

    request_params = _build_stream_request_params(
        history,
        model,
        instructions=instructions,
        reasoning=reasoning,
        tools=tools,
    )
    subscription_stream = await _create_subscription_stream(
        request_params,
        raw_stream=raw_stream,
    )
    return assemble_stream(
        normalize_subscription_events(subscription_stream),
        source=ProviderSource(provider="openai", model=model),
    )


async def _create_api_stream(
    client: AsyncOpenAI,
    request_params: ResponseCreateParamsStreaming,
) -> AsyncIterator[ResponseStreamEvent]:
    """Create the raw OpenAI SDK event stream for API-based auth."""

    return await client.responses.create(**request_params)


async def _create_subscription_stream(
    request_params: ResponseCreateParamsStreaming,
    *,
    raw_stream: AsyncIterator[SubscriptionEventPayload] | None,
) -> AsyncIterator[SubscriptionEventPayload]:
    """Create the raw subscription event stream.

    The concrete HTTP/SSE transport is added in the next phase. The injected
    raw stream keeps the provider and normalized event handler testable now.
    """

    _ = request_params
    if raw_stream is None:
        raise NotImplementedError("Subscription transport is not implemented yet.")
    return raw_stream


def _build_stream_request_params(
    history: Sequence[ConversationItem],
    model: str,
    *,
    instructions: str,
    reasoning: Reasoning | None = None,
    tools: Sequence[ToolDefinition] | None = None,
) -> ResponseCreateParamsStreaming:
    """Build the shared Responses API request payload for stream transports."""

    request_params: ResponseCreateParamsStreaming = {
        "model": model,
        "input": serialize_history_items(history),
        "reasoning": cast("OpenAIReasoning | None", reasoning),
        "instructions": instructions,
        "stream": True,
    }
    if tools:
        request_params["tools"] = serialize_tools(tools)
    return request_params
