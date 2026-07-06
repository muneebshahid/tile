"""OpenAI provider entrypoint for the API streaming transport."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from openai import AsyncOpenAI
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from ori.events import StreamFn
from ori.providers.openai.serialization import serialize_history_items, serialize_tools
from ori.providers.openai.sdk_event_adapter import normalize_sdk_events
from ori.providers.openai.stream_assembler import assemble_stream
from ori.types.contracts import AsyncEventStream
from ori.types.conversation import ConversationItem
from ori.types.stream_events import ProviderSource
from ori.types.tools import ToolDefinition

if TYPE_CHECKING:
    from openai.types.shared_params.reasoning import Reasoning as OpenAIReasoning


class Reasoning(TypedDict, total=False):
    """OpenAI reasoning options bound to a stream function at creation."""

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    summary: Literal["auto", "concise", "detailed"]


def create_stream_api(
    client: AsyncOpenAI,
    *,
    reasoning: Reasoning | None = None,
) -> StreamFn:
    """Bind a caller-constructed OpenAI client to an API-transport stream function."""

    async def stream_api(
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
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
        raw_stream = await client.responses.create(**request_params)
        return assemble_stream(
            normalize_sdk_events(raw_stream),
            source=ProviderSource(provider="openai", model=model),
        )

    return stream_api


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
