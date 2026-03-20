from collections.abc import AsyncIterator

from agent.agent import Agent, StreamFn
from ai.contracts import Reasoning
from ai.types import (
    AssistantMessage,
    ReasoningBlock,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamEvent,
    StreamStartEvent,
    TextBlock,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
)


class FakeResponseStream:
    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        for event in self._events:
            yield event


class FakeResponsesClient:
    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def create(
        self,
        prompt: str,
        model: str,
        reasoning: Reasoning | None,
    ) -> FakeResponseStream:
        assert model == "gpt-5.4"
        assert prompt == "hello"
        assert reasoning == {"effort": "medium"}
        return FakeResponseStream(self._events)


def test_agent_run_tracks_streaming_message_and_final_message() -> None:
    partial = AssistantMessage(response_id="resp_123")
    final_message = AssistantMessage(
        response_id="resp_123",
        content=[
            ReasoningBlock(
                type="reasoning",
                reasoning="first think",
                reasoning_id="rs_123",
            ),
            TextBlock(
                type="text",
                text="final answer",
            ),
        ],
    )
    client = FakeResponsesClient(
        [
            StreamStartEvent(type="start", partial=partial),
            ReasoningStartEvent(type="reasoning_start", partial=partial),
            ReasoningDeltaEvent(
                type="reasoning_delta",
                delta="first think",
                partial=partial,
            ),
            ReasoningEndEvent(type="reasoning_end", partial=partial),
            TextStartEvent(type="text_start", partial=partial),
            TextDeltaEvent(
                type="text_delta",
                delta="final answer",
                partial=partial,
            ),
            TextEndEvent(type="text_end", partial=partial),
            StreamDoneEvent(type="done", message=final_message),
        ]
    )
    stream_fn: StreamFn = client.create
    agent = Agent(
        stream_fn=stream_fn,
        model="gpt-5.4",
        reasoning={"effort": "medium"},
    )
    import asyncio

    asyncio.run(agent.run("hello"))

    assert agent.state.is_streaming is False
    assert agent.state.stream_message is None
    assert agent.state.messages == [final_message]
