from pydantic import TypeAdapter

from ori.providers.openai.serialization import (
    serialize_history_items,
    serialize_response_input,
    serialize_tools,
)
from ori.types.conversation import AssistantTurn, ToolResultTurn, UserMessage
from ori.types.stream_events import (
    ProviderMetadata,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)
from ori.types.tools import ToolDefinition, ToolImageContent, ToolResult
from openai.types.responses.response_input_param import ResponseInputParam


async def _sample_tool_fn(city: str) -> ToolResult:
    """Return a deterministic payload for serialization-only tool definitions."""

    return ToolResult.text(f"city={city}")


def test_serialize_response_input_flattens_sample_thread() -> None:
    history = [
        UserMessage(content="Write a haiku about rain."),
        AssistantTurn(
            response_id="resp_123",
            blocks=[
                ReasoningBlock(
                    summary_text="Draft a short seasonal poem.",
                    provider_metadata=ProviderMetadata.from_values(
                        reasoning_signature='{"id":"rs_123","type":"reasoning","summary":[{"type":"summary_text","text":"Draft a short seasonal poem."}],"encrypted_content":"enc_123","status":"completed"}',
                    ),
                ),
                TextBlock(
                    text="Soft rain on pine leaves\nSilver threads stitch dusk to earth\nNight drinks every sound",
                    provider_metadata=ProviderMetadata.from_values(
                        message_id="msg_123",
                        phase="final_answer",
                    ),
                ),
            ],
        ),
        AssistantTurn(
            status="aborted",
            blocks=[
                ReasoningBlock(
                    summary_text="This partial turn should be skipped.",
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
            "status": "completed",
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
            blocks=[
                ReasoningBlock(summary_text="Think first."),
                TextBlock(text="Answer next."),
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


def test_serialize_history_items_skips_reasoning_without_signature() -> None:
    history = [
        AssistantTurn(
            blocks=[
                ReasoningBlock(
                    summary_text="Think first.",
                )
            ]
        )
    ]

    serialized = serialize_history_items(history)

    assert serialized == []
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_history_items_generates_fallback_message_ids() -> None:
    history = [
        AssistantTurn(
            blocks=[
                TextBlock(text="Answer next."),
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


def test_serialize_history_items_replays_tool_calls_and_tool_results() -> None:
    history = [
        AssistantTurn(
            blocks=[
                TextBlock(
                    text="Checking the weather.",
                    provider_metadata=ProviderMetadata.from_values(message_id="msg_0"),
                ),
                ToolCallBlock(
                    call_id="call_123",
                    name="get_weather",
                    arguments={"city": "Berlin"},
                    provider_metadata=ProviderMetadata.from_values(
                        provider_item_id="fc_123"
                    ),
                ),
            ]
        ),
        ToolResultTurn(
            call_id="call_123",
            tool_name="get_weather",
            content=ToolResult.text("Temperature: 14 C").content,
            is_error=False,
        ),
    ]

    serialized = serialize_history_items(history)

    assert serialized == [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": "msg_0",
            "content": [
                {
                    "type": "output_text",
                    "text": "Checking the weather.",
                    "annotations": [],
                }
            ],
        },
        {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"city": "Berlin"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "Temperature: 14 C",
        },
    ]
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_history_items_replays_tool_result_images() -> None:
    """Serialize image tool results as OpenAI Responses content parts."""

    history = [
        ToolResultTurn(
            call_id="call_123",
            tool_name="read",
            content=ToolResult.image(
                "Read image file [image/png]",
                ToolImageContent(data="ZmFrZQ==", mime_type="image/png"),
            ).content,
            is_error=False,
        )
    ]

    serialized = serialize_history_items(history)

    assert serialized == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": [
                {"type": "input_text", "text": "Read image file [image/png]"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,ZmFrZQ==",
                    "detail": "auto",
                },
            ],
        }
    ]
    TypeAdapter(ResponseInputParam).validate_python(serialized)


def test_serialize_tools_maps_tool_definitions_to_function_tools() -> None:
    tools = [
        ToolDefinition(
            name="get_weather",
            description="Return the current weather for a city.",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to look up.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            fn=_sample_tool_fn,
        )
    ]

    assert serialize_tools(tools) == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to look up.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
            "defer_loading": False,
        }
    ]
