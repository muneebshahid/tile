from collections.abc import Awaitable, Sequence
from typing import Callable

from ai.contracts import AsyncEventStream, Reasoning
from ai.types import (
    AssistantMessage,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    SystemMessage,
    UserMessage,
)
from agent.prompt import PROMPT


StreamFn = Callable[[str, str, Reasoning | None], Awaitable[AsyncEventStream]]


class Agent:
    def __init__(
        self,
        stream_fn: StreamFn,
        model: str,
        reasoning: Reasoning | None = None,
        messages: Sequence[AssistantMessage] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._stream_fn = stream_fn
        self._model = model
        self._reasoning = reasoning
        self._messages = list(messages or [])
        self._messages.append(SystemMessage(content=system_prompt or PROMPT))

    def update_model(self, model: str) -> None:
        self._model = model

    def update_reasoning(self, reasoning: Reasoning | None) -> None:
        self._reasoning = reasoning

    def replace_messages(self, messages: Sequence[AssistantMessage]) -> None:
        self._messages = list(messages)

    def add_message(self, message: AssistantMessage) -> None:
        self._messages.append(message)

    async def run(self, prompt: str) -> None:
        self._is_streaming = True
        self._stream_message = None
        self._messages.append(UserMessage(content=prompt))
        stream = await self._stream_fn(self._messages, self._model, self._reasoning)

        async for event in stream:
            await self._dispatch_event(event)

    async def _dispatch_event(self, event: StreamEvent) -> None:
        match event:
            case StreamStartEvent():
                await self._handle_stream_start_event(event)
            case ReasoningStartEvent() | ReasoningDeltaEvent() | ReasoningEndEvent():
                await self._handle_reasoning_event(event)
            case TextStartEvent() | TextDeltaEvent() | TextEndEvent():
                await self._handle_text_event(event)
            case StreamDoneEvent():
                await self._handle_stream_done_event(event)
            case StreamErrorEvent():
                await self._handle_stream_error_event(event)
            case _:
                return None

    async def _handle_stream_start_event(
        self,
        event: StreamStartEvent,
    ) -> None:
        self._stream_message = event.partial

    async def _handle_reasoning_event(
        self,
        event: ReasoningStartEvent | ReasoningDeltaEvent | ReasoningEndEvent,
    ) -> None:
        self._stream_message = event.partial

    async def _handle_text_event(
        self,
        event: TextStartEvent | TextDeltaEvent | TextEndEvent,
    ) -> None:
        self._stream_message = event.partial

    async def _handle_stream_done_event(
        self,
        event: StreamDoneEvent,
    ) -> None:
        self._messages.append(event.message)
        self._stream_message = None
        self._is_streaming = False

    async def _handle_stream_error_event(
        self,
        event: StreamErrorEvent,
    ) -> None:
        self._stream_message = event.partial
        self._is_streaming = False
