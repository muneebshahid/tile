"""Tests for normalizing subscription SSE payloads into provider events."""

import asyncio
import json
from collections.abc import Sequence

import pytest

from ori.providers.openai.normalized_events import NormalizedEvent, NormalizedEventType
from ori.providers.openai.subscription_event_adapter import (
    SubscriptionEventPayload,
    normalize_subscription_events,
)
from tests.support.async_streams import async_stream


@pytest.mark.parametrize(
    ("raw_events", "expected_events"),
    [
        pytest.param(
            [{"type": "response.created", "response": {"id": "resp_123"}}],
            [{"type": NormalizedEventType.CREATED, "response_id": "resp_123"}],
            id="created",
        ),
        pytest.param(
            [{"type": "response.completed", "response": {"output": []}}],
            [{"type": NormalizedEventType.COMPLETED, "stop_reason": "stop"}],
            id="completed-stop",
        ),
        pytest.param(
            [{"type": "response.done", "response": {"status": "completed"}}],
            [{"type": NormalizedEventType.COMPLETED, "stop_reason": "stop"}],
            id="done-completed",
        ),
        pytest.param(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_123",
                                "name": "get_weather",
                            }
                        ]
                    },
                }
            ],
            [{"type": NormalizedEventType.COMPLETED, "stop_reason": "tool_use"}],
            id="completed-tool-use",
        ),
        pytest.param(
            [
                {
                    "type": "response.incomplete",
                    "response": {
                        "incomplete_details": {"reason": "content_filter"},
                    },
                }
            ],
            [
                {
                    "type": NormalizedEventType.INCOMPLETE,
                    "stop_reason": "error",
                    "error_message": (
                        "OpenAI response was truncated by the content filter."
                    ),
                }
            ],
            id="incomplete-content-filter",
        ),
        pytest.param(
            [
                {
                    "type": "response.done",
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                }
            ],
            [
                {
                    "type": NormalizedEventType.INCOMPLETE,
                    "stop_reason": "length",
                    "error_message": "OpenAI response incomplete.",
                }
            ],
            id="done-incomplete-length",
        ),
        pytest.param(
            [
                {
                    "type": "response.failed",
                    "response": {"error": {"message": "Model overloaded"}},
                }
            ],
            [{"type": NormalizedEventType.FAILED, "message": "Model overloaded"}],
            id="failed",
        ),
        pytest.param(
            [{"type": "error", "message": "Socket closed"}],
            [{"type": NormalizedEventType.FAILED, "message": "Socket closed"}],
            id="error",
        ),
    ],
)
def test_normalize_subscription_events_maps_response_payloads(
    raw_events: Sequence[SubscriptionEventPayload],
    expected_events: Sequence[NormalizedEvent],
) -> None:
    """Normalize response lifecycle payloads."""

    assert _collect_events(raw_events) == list(expected_events)


@pytest.mark.parametrize(
    ("raw_events", "expected_events"),
    [
        pytest.param(
            [
                {
                    "type": "response.output_item.added",
                    "item": {
                        "id": "rs_123",
                        "type": "reasoning",
                        "summary": [],
                        "status": "in_progress",
                    },
                }
            ],
            [{"type": NormalizedEventType.REASONING_ADDED, "item_id": "rs_123"}],
            id="reasoning-added",
        ),
        pytest.param(
            [
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "rs_123",
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "Exploring traces"},
                            {"type": "summary_text", "text": "Selecting an answer"},
                        ],
                        "status": "completed",
                    },
                }
            ],
            [
                {
                    "type": NormalizedEventType.REASONING_DONE,
                    "item_id": "rs_123",
                    "summary_text": "Exploring traces\n\nSelecting an answer",
                    "reasoning_signature": json.dumps(
                        {
                            "id": "rs_123",
                            "type": "reasoning",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "Exploring traces",
                                },
                                {
                                    "type": "summary_text",
                                    "text": "Selecting an answer",
                                },
                            ],
                            "status": "completed",
                        }
                    ),
                }
            ],
            id="reasoning-done",
        ),
        pytest.param(
            [
                {
                    "type": "response.output_item.added",
                    "item": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [],
                    },
                }
            ],
            [
                {
                    "type": NormalizedEventType.MESSAGE_ADDED,
                    "item_id": "msg_123",
                    "phase": "final_answer",
                }
            ],
            id="message-added",
        ),
        pytest.param(
            [
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Hello",
                                "annotations": [],
                            }
                        ],
                    },
                }
            ],
            [
                {
                    "type": NormalizedEventType.MESSAGE_DONE,
                    "item_id": "msg_123",
                    "text": "Hello",
                    "phase": "final_answer",
                }
            ],
            id="message-done",
        ),
        pytest.param(
            [
                {
                    "type": "response.output_item.added",
                    "item": {
                        "id": "fc_123",
                        "type": "function_call",
                        "call_id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"city":"Berlin"}',
                    },
                }
            ],
            [
                {
                    "type": NormalizedEventType.TOOL_CALL_ADDED,
                    "provider_item_id": "fc_123",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": {"city": "Berlin"},
                }
            ],
            id="tool-call-added",
        ),
        pytest.param(
            [
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "fc_123",
                        "type": "function_call",
                        "call_id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"city":"Berlin"}',
                    },
                }
            ],
            [
                {
                    "type": NormalizedEventType.TOOL_CALL_DONE,
                    "provider_item_id": "fc_123",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": {"city": "Berlin"},
                }
            ],
            id="tool-call-done",
        ),
    ],
)
def test_normalize_subscription_events_maps_output_item_payloads(
    raw_events: Sequence[SubscriptionEventPayload],
    expected_events: Sequence[NormalizedEvent],
) -> None:
    """Normalize output item lifecycle payloads."""

    assert _collect_events(raw_events) == list(expected_events)


@pytest.mark.parametrize(
    ("raw_events", "expected_events"),
    [
        pytest.param(
            [
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs_123",
                    "delta": "Exploring traces",
                }
            ],
            [
                {
                    "type": NormalizedEventType.REASONING_DELTA,
                    "delta": "Exploring traces",
                }
            ],
            id="reasoning-summary-delta",
        ),
        pytest.param(
            [
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": "rs_123",
                    "delta": " with text",
                }
            ],
            [{"type": NormalizedEventType.REASONING_DELTA, "delta": " with text"}],
            id="reasoning-text-delta",
        ),
        pytest.param(
            [{"type": "response.reasoning_summary_part.done"}],
            [{"type": NormalizedEventType.REASONING_DELTA, "delta": "\n\n"}],
            id="reasoning-part-done",
        ),
        pytest.param(
            [
                {
                    "type": "response.content_part.added",
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }
            ],
            [
                {
                    "type": NormalizedEventType.MESSAGE_TEXT_PART,
                    "part_type": "output_text",
                }
            ],
            id="output-text-part",
        ),
        pytest.param(
            [
                {
                    "type": "response.content_part.added",
                    "part": {"type": "refusal", "refusal": ""},
                }
            ],
            [
                {
                    "type": NormalizedEventType.MESSAGE_TEXT_PART,
                    "part_type": "refusal",
                }
            ],
            id="refusal-part",
        ),
        pytest.param(
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_123",
                    "delta": "Hello",
                }
            ],
            [
                {
                    "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
                    "part_type": "output_text",
                    "delta": "Hello",
                }
            ],
            id="text-delta",
        ),
        pytest.param(
            [
                {
                    "type": "response.refusal.delta",
                    "item_id": "msg_123",
                    "delta": "No",
                }
            ],
            [
                {
                    "type": NormalizedEventType.MESSAGE_TEXT_DELTA,
                    "part_type": "refusal",
                    "delta": "No",
                }
            ],
            id="refusal-delta",
        ),
        pytest.param(
            [
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_123",
                    "delta": '{"city":',
                }
            ],
            [
                {
                    "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DELTA,
                    "delta": '{"city":',
                }
            ],
            id="tool-call-arguments-delta",
        ),
        pytest.param(
            [
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_123",
                    "arguments": '{"city":"Berlin"}',
                }
            ],
            [
                {
                    "type": NormalizedEventType.TOOL_CALL_ARGUMENTS_DONE,
                    "arguments": {"city": "Berlin"},
                }
            ],
            id="tool-call-arguments-done",
        ),
    ],
)
def test_normalize_subscription_events_maps_delta_payloads(
    raw_events: Sequence[SubscriptionEventPayload],
    expected_events: Sequence[NormalizedEvent],
) -> None:
    """Normalize incremental text, reasoning, and tool-call payloads."""

    assert _collect_events(raw_events) == list(expected_events)


def _collect_events(
    raw_events: Sequence[SubscriptionEventPayload],
) -> list[NormalizedEvent]:
    """Collect normalized events from raw subscription payloads."""

    async def _collect() -> list[NormalizedEvent]:
        return [
            event
            async for event in normalize_subscription_events(async_stream(raw_events))
        ]

    return asyncio.run(_collect())
