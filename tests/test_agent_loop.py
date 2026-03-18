from unittest.mock import AsyncMock, patch
from collections.abc import AsyncIterator

from agent.loop import run_agent_loop
from openai.types.responses.response_created_event import ResponseCreatedEvent
from openai.types.responses.response_in_progress_event import ResponseInProgressEvent
from openai.types.responses.response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)


class FakeResponseStream:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def __aiter__(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event


class FakeResponsesClient:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def create(self, prompt: str, model: str) -> FakeResponseStream:
        assert model == "gpt-5.4"
        assert prompt == "hello"
        return FakeResponseStream(self._events)


def test_run_agent_loop_dispatches_supported_events() -> None:
    created_event = ResponseCreatedEvent.model_validate(
        {
            "type": "response.created",
            "sequence_number": 1,
            "response": {
                "id": "resp_123",
                "created_at": 0.0,
                "model": "gpt-5.4",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "in_progress",
            },
        }
    )
    in_progress_event = ResponseInProgressEvent.model_validate(
        {
            "type": "response.in_progress",
            "sequence_number": 2,
            "response": {
                "id": "resp_123",
                "created_at": 0.0,
                "model": "gpt-5.4",
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "status": "in_progress",
            },
        }
    )
    output_item_added_event = ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": 3,
            "output_index": 0,
            "item": {
                "id": "msg_123",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }
    )
    stream_fn = FakeResponsesClient(
        [created_event, in_progress_event, output_item_added_event]
    ).create

    with (
        patch(
            "agent.loop.handle_response_created_event",
            new_callable=AsyncMock,
        ) as handle_created,
        patch(
            "agent.loop.handle_response_in_progress_event",
            new_callable=AsyncMock,
        ) as handle_in_progress,
        patch(
            "agent.loop.handle_response_output_item_added_event",
            new_callable=AsyncMock,
        ) as handle_output_item_added,
    ):
        import asyncio

        asyncio.run(run_agent_loop(stream_fn, "hello", "gpt-5.4"))

    handle_created.assert_awaited_once_with(created_event)
    handle_in_progress.assert_awaited_once_with(in_progress_event)
    handle_output_item_added.assert_awaited_once_with(output_item_added_event)
