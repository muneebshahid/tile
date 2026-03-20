from dataclasses import dataclass, field
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
)


@dataclass
class AgentState:
    model: str
    reasoning: Reasoning | None = None
    messages: list[AssistantMessage] = field(default_factory=list)
    stream_message: AssistantMessage | None = None
    is_streaming: bool = False


StreamFn = Callable[[str, str, Reasoning | None], Awaitable[AsyncEventStream]]


class Agent:
    def __init__(
        self,
        stream_fn: StreamFn,
        model: str,
        reasoning: Reasoning | None = None,
        messages: Sequence[AssistantMessage] | None = None,
    ) -> None:
        self._stream_fn = stream_fn
        self._state = AgentState(
            model=model,
            reasoning=reasoning,
            messages=list(messages or []),
        )

    @property
    def state(self) -> AgentState:
        return self._state

    def update_model(self, model: str) -> None:
        self._state.model = model

    def update_reasoning(self, reasoning: Reasoning | None) -> None:
        self._state.reasoning = reasoning

    def replace_messages(self, messages: Sequence[AssistantMessage]) -> None:
        self._state.messages = list(messages)

    def add_message(self, message: AssistantMessage) -> None:
        self._state.messages.append(message)

    async def run(self, prompt: str) -> None:
        self._state.is_streaming = True
        self._state.stream_message = None
        stream = await self._stream_fn(prompt, self._state.model, self._state.reasoning)

        async for event in stream:
            await _dispatch_event(self._state, event)


async def _dispatch_event(state: AgentState, event: StreamEvent) -> None:
    match event:
        case StreamStartEvent():
            await handle_stream_start_event(state, event)
        case ReasoningStartEvent() | ReasoningDeltaEvent() | ReasoningEndEvent():
            await handle_reasoning_event(state, event)
        case TextStartEvent() | TextDeltaEvent() | TextEndEvent():
            await handle_text_event(state, event)
        case StreamDoneEvent():
            await handle_stream_done_event(state, event)
        case StreamErrorEvent():
            await handle_stream_error_event(state, event)
        case _:
            return None


async def handle_stream_start_event(
    state: AgentState,
    event: StreamStartEvent,
) -> None:
    state.stream_message = event.partial


async def handle_reasoning_event(
    state: AgentState,
    event: ReasoningStartEvent | ReasoningDeltaEvent | ReasoningEndEvent,
) -> None:
    state.stream_message = event.partial


async def handle_text_event(
    state: AgentState,
    event: TextStartEvent | TextDeltaEvent | TextEndEvent,
) -> None:
    state.stream_message = event.partial


async def handle_stream_done_event(
    state: AgentState,
    event: StreamDoneEvent,
) -> None:
    state.messages.append(event.message)
    state.stream_message = None
    state.is_streaming = False


async def handle_stream_error_event(
    state: AgentState,
    event: StreamErrorEvent,
) -> None:
    state.stream_message = event.partial
    state.is_streaming = False
