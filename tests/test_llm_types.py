from llm.types import (
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
)


def test_response_created_event_model() -> None:
    event = ResponseCreatedEvent.model_validate(
        {
            "type": "response.created",
            "sequence_number": 1,
            "response": {
                "id": "resp_123",
                "model": "gpt-5.4",
                "status": "in_progress",
            },
        }
    )

    assert event.response.id == "resp_123"
    assert event.response.model == "gpt-5.4"
    assert event.response.status == "in_progress"


def test_response_in_progress_event_model() -> None:
    event = ResponseInProgressEvent.model_validate(
        {
            "type": "response.in_progress",
            "sequence_number": 2,
            "response": {
                "id": "resp_123",
                "model": "gpt-5.4",
                "status": "in_progress",
            },
        }
    )

    assert event.type == "response.in_progress"
    assert event.sequence_number == 2


def test_response_output_item_added_event_message_item() -> None:
    event = ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": 3,
            "output_index": 0,
            "item": {
                "id": "msg_123",
                "type": "message",
                "status": "in_progress",
            },
        }
    )

    assert event.item.type == "message"
    assert event.output_index == 0


def test_response_output_item_added_event_reasoning_item() -> None:
    event = ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": 4,
            "output_index": 1,
            "item": {
                "id": "rs_123",
                "type": "reasoning",
                "status": "in_progress",
            },
        }
    )

    assert event.item.type == "reasoning"


def test_response_output_item_added_event_function_call_item() -> None:
    event = ResponseOutputItemAddedEvent.model_validate(
        {
            "type": "response.output_item.added",
            "sequence_number": 5,
            "output_index": 2,
            "item": {
                "id": "fc_123",
                "type": "function_call",
                "status": "in_progress",
                "call_id": "call_123",
                "name": "lookup_customer",
            },
        }
    )

    assert event.item.type == "function_call"
    assert event.item.call_id == "call_123"
    assert event.item.name == "lookup_customer"
