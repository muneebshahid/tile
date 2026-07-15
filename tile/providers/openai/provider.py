"""OpenAI provider entrypoint for the API streaming transport."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from openai import AsyncOpenAI
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from tile.events import StreamFn
from tile.providers.openai.serialization import serialize_history_items, serialize_tools
from tile.providers.openai.sdk_event_adapter import normalize_sdk_events
from tile.providers.openai.stream_assembler import assemble_stream
from tile.types.contracts import AsyncEventStream
from tile.types.conversation import ConversationItem
from tile.types.stream_events import ProviderSource
from tile.types.tools import ToolDefinition

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

    return _StreamApi(client, reasoning=reasoning)


class _StreamApi:
    """API-transport stream function bound to one OpenAI client."""

    provider = "openai"

    def __init__(self, client: AsyncOpenAI, *, reasoning: Reasoning | None) -> None:
        """Bind the client and reasoning options for every stream call."""

        self._client = client
        self._reasoning = reasoning

    async def __call__(
        self,
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
            reasoning=self._reasoning,
            tools=tools,
        )
        raw_stream = await self._client.responses.create(**request_params)
        return assemble_stream(
            normalize_sdk_events(raw_stream),
            source=ProviderSource(provider=self.provider, model=model),
        )


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
