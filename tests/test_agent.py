from collections.abc import AsyncIterator, Sequence

from agent.agent import Agent, AgentRunError, StreamFn
from ai.types.contracts import Reasoning
from ai.types.conversation import AssistantTurn, ConversationItem, UserMessage
from ai.types.stream import (
    AssistantMessage,
    ReasoningDeltaEvent,
    ReasoningBlock,
    ReasoningEndEvent,
    ReasoningStartEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDeltaEvent,
    TextBlock,
    TextEndEvent,
    TextStartEvent,
    ToolCallBlock,
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
        history: Sequence[ConversationItem],
        model: str,
        *,
        instructions: str,
        reasoning: Reasoning | None,
    ) -> FakeResponseStream:
        assert model == "gpt-5.4"
        assert history == [UserMessage(content="hello")]
        assert reasoning == {"effort": "medium"}
        assert instructions
        return FakeResponseStream(self._events)


def test_agent_run_tracks_streaming_message_and_final_message() -> None:
    partial = AssistantMessage(response_id="resp_123")
    final_message = AssistantMessage(
        response_id="resp_123",
        content=[
            ReasoningBlock(
                summary_text="first think",
                reasoning_id="rs_123",
            ),
            TextBlock(text="final answer"),
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

    assert agent.history == (
        UserMessage(content="hello"),
        AssistantTurn(
            response_id="resp_123",
            content=[
                ReasoningBlock(
                    summary_text="first think",
                    reasoning_id="rs_123",
                ),
                TextBlock(text="final answer"),
            ],
        ),
    )


def test_agent_run_raises_when_stream_emits_error() -> None:
    client = FakeResponsesClient(
        [
            StreamStartEvent(type="start", partial=AssistantMessage()),
            StreamErrorEvent(
                type="error",
                message="Model overloaded",
                partial=AssistantMessage(),
            ),
        ]
    )
    stream_fn: StreamFn = client.create
    agent = Agent(
        stream_fn=stream_fn,
        model="gpt-5.4",
        reasoning={"effort": "medium"},
    )
    import asyncio

    try:
        asyncio.run(agent.run("hello"))
    except AgentRunError as error:
        assert str(error) == "Model overloaded"
    else:
        raise AssertionError("Expected Agent.run() to raise on stream error.")

    assert agent.history == (UserMessage(content="hello"),)


def test_agent_run_preserves_tool_call_blocks_and_stop_reason() -> None:
    final_message = AssistantMessage(
        response_id="resp_123",
        stop_reason="tool_use",
        content=[
            ToolCallBlock(
                call_id="call_123",
                name="ls",
                arguments={"directory": "."},
                provider_item_id="fc_123",
            )
        ],
    )
    client = FakeResponsesClient(
        [
            StreamStartEvent(type="start", partial=AssistantMessage()),
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

    assert agent.history == (
        UserMessage(content="hello"),
        AssistantTurn(
            response_id="resp_123",
            stop_reason="tool_use",
            content=[
                ToolCallBlock(
                    call_id="call_123",
                    name="ls",
                    arguments={"directory": "."},
                    provider_item_id="fc_123",
                )
            ],
        ),
    )
