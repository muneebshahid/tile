from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI
from openai.types.responses.response_create_params import (
    ResponseCreateParamsStreaming,
)

from ai.openai.client import create_client
from ai.openai.sdk_event_adapter import normalize_sdk_events
from ai.openai.serialization import serialize_history_items, serialize_tools
from ai.openai.stream_assembler import assemble_stream
from ai.types.contracts import AsyncEventStream, Reasoning as AppReasoning
from ai.types.conversation import ConversationItem
from ai.types.tools import ToolDefinition

if TYPE_CHECKING:
    from openai.types.shared_params.reasoning import Reasoning as OpenAIReasoning


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
    raw_stream = await _create_sdk_stream(
        active_client,
        history,
        model,
        instructions=instructions,
        reasoning=reasoning,
        tools=tools,
    )
    return assemble_stream(normalize_sdk_events(raw_stream))


async def _create_sdk_stream(
    client: AsyncOpenAI,
    history: Sequence[ConversationItem],
    model: str,
    *,
    instructions: str,
    reasoning: AppReasoning | None,
    tools: Sequence[ToolDefinition] | None,
) -> AsyncIterator[object]:
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

    return await client.responses.create(**request_params)
