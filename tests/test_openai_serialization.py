from pydantic import TypeAdapter

from ai.conversation import (
    AssistantReasoningBlock,
    AssistantTextBlock,
    AssistantTurn,
    UserMessage,
)
from ai.openai.serialization import (
    serialize_history_items,
    serialize_response_input,
)
from openai.types.responses.response_input_param import ResponseInputParam


def test_serialize_response_input_flattens_sample_thread() -> None:
    history = [
        UserMessage(content="Write a haiku about rain."),
        AssistantTurn(
            response_id="resp_123",
            content=[
                AssistantReasoningBlock(
                    summary_text="Draft a short seasonal poem.",
                    reasoning_id="rs_123",
                    encrypted_content="enc_123",
                ),
                AssistantTextBlock(
                    text="Soft rain on pine leaves\nSilver threads stitch dusk to earth\nNight drinks every sound",
                    message_id="msg_123",
                    phase="final_answer",
                ),
            ],
        ),
        AssistantTurn(
            status="aborted",
            content=[
                AssistantReasoningBlock(
                    summary_text="This partial turn should be skipped.",
                    reasoning_id="rs_skip",
                )
            ],
        ),
        UserMessage(content="Revise the second line."),
    ]

    serialized = serialize_response_input(
        history,
        system_prompt="You are a careful poet.",
    )

    expected: ResponseInputParam = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": "You are a careful poet."}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Write a haiku about rain."}],
        },
        {
            "type": "reasoning",
            "id": "rs_123",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "Draft a short seasonal poem.",
                }
            ],
            "encrypted_content": "enc_123",
        },
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_123",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "Soft rain on pine leaves\nSilver threads stitch dusk to earth\nNight drinks every sound",
                    "annotations": [],
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Revise the second line."}],
        },
    ]

    assert serialized == expected
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_history_items_skips_reasoning_without_replay_metadata() -> None:
    history = [
        AssistantTurn(
            content=[
                AssistantReasoningBlock(summary_text="Think first."),
                AssistantTextBlock(text="Answer next."),
            ]
        )
    ]

    serialized = serialize_history_items(history)

    assert serialized == [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_0_1",
            "content": [
                {
                    "type": "output_text",
                    "text": "Answer next.",
                    "annotations": [],
                }
            ],
        },
    ]
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_history_items_generates_fallback_message_ids() -> None:
    history = [
        AssistantTurn(
            content=[
                AssistantTextBlock(text="Answer next."),
            ]
        )
    ]

    serialized = serialize_history_items(history)

    assert serialized == [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_0_0",
            "content": [
                {
                    "type": "output_text",
                    "text": "Answer next.",
                    "annotations": [],
                }
            ],
        },
    ]
    TypeAdapter(ResponseInputParam).validate_python(serialized)
