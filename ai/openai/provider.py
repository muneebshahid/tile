"""OpenAI provider entrypoints split by transport."""

from collections.abc import AsyncIterator, Sequence

from openai import AsyncOpenAI
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from ai.openai.client import create_client
from ai.openai.request_params import build_stream_request_params
from ai.openai.sdk_event_adapter import normalize_sdk_events
from ai.openai.stream_assembler import assemble_stream
from ai.openai.subscription_event_adapter import (
    SubscriptionEventPayload,
    normalize_subscription_events,
)
from ai.types.contracts import AsyncEventStream, Reasoning as AppReasoning
from ai.types.conversation import ConversationItem
from ai.types.tools import ToolDefinition


async def stream_api(
    history: Sequence[ConversationItem],
    model: str,
    *,
    instructions: str,
    reasoning: AppReasoning | None = None,
    tools: Sequence[ToolDefinition] | None = None,
) -> AsyncEventStream:
    """Stream assistant events through the OpenAI SDK transport."""

    request_params = build_stream_request_params(
        history,
        model,
        instructions=instructions,
        reasoning=reasoning,
        tools=tools,
    )
    raw_stream = await _create_api_stream(create_client(), request_params)
    return assemble_stream(normalize_sdk_events(raw_stream))


async def stream_subscription(
    history: Sequence[ConversationItem],
    model: str,
    *,
    instructions: str,
    reasoning: AppReasoning | None = None,
    tools: Sequence[ToolDefinition] | None = None,
    raw_stream: AsyncIterator[SubscriptionEventPayload] | None = None,
) -> AsyncEventStream:
    """Stream assistant events through the subscription SSE transport."""

    request_params = build_stream_request_params(
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
    return assemble_stream(normalize_subscription_events(subscription_stream))


async def _create_api_stream(
    client: AsyncOpenAI,
    request_params: ResponseCreateParamsStreaming,
) -> AsyncIterator[object]:
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
