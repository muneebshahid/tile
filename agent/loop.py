from collections.abc import AsyncIterator
from typing import Callable, Awaitable, Protocol

from openai.types.responses.response_created_event import ResponseCreatedEvent
from openai.types.responses.response_in_progress_event import ResponseInProgressEvent
from openai.types.responses.response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)


class ResponseEventStream(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...


StreamFn = Callable[[str, str], Awaitable[ResponseEventStream]]


async def run_agent_loop(
    stream_fn: StreamFn,
    prompt: str,
    model: str,
) -> None:
    stream = await stream_fn(prompt, model)

    async for event in stream:
        await _dispatch_event(event)


async def _dispatch_event(event: object) -> None:
    match event:
        case ResponseCreatedEvent():
            await handle_response_created_event(event)
        case ResponseInProgressEvent():
            await handle_response_in_progress_event(event)
        case ResponseOutputItemAddedEvent():
            await handle_response_output_item_added_event(event)
        case _:
            return None


async def handle_response_created_event(event: ResponseCreatedEvent) -> None:
    del event


async def handle_response_in_progress_event(event: ResponseInProgressEvent) -> None:
    del event


async def handle_response_output_item_added_event(
    event: ResponseOutputItemAddedEvent,
) -> None:
    del event
