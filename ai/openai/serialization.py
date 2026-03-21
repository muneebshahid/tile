from collections.abc import Sequence
from typing import Literal

from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_message_param import (
    ResponseOutputMessageParam,
)
from openai.types.responses.response_output_text_param import ResponseOutputTextParam
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
    Summary as ResponseReasoningSummaryParam,
)

from ai.conversation import (
    AssistantReasoningBlock,
    AssistantTextBlock,
    AssistantTurn,
    ConversationItem,
    UserMessage,
)


def serialize_response_input(
    history: Sequence[ConversationItem],
    *,
    system_prompt: str | None = None,
    system_role: Literal["system", "developer"] = "system",
) -> ResponseInputParam:
    """Serialize a conversation thread into OpenAI Responses input items."""

    items: ResponseInputParam = []
    if system_prompt:
        items.append(_serialize_system_prompt(system_prompt, system_role))
    items.extend(serialize_history_items(history))
    return items


def serialize_history_items(
    history: Sequence[ConversationItem],
) -> ResponseInputParam:
    """Serialize replayable conversation history into OpenAI Responses items."""

    items: ResponseInputParam = []
    assistant_turn_index = 0

    for item in history:
        match item:
            case UserMessage():
                items.append(_serialize_user_message(item))
            case AssistantTurn(status="completed"):
                items.extend(
                    _serialize_assistant_turn(
                        item,
                        assistant_turn_index=assistant_turn_index,
                    )
                )
                assistant_turn_index += 1
            case AssistantTurn():
                continue

    return items


def _serialize_system_prompt(
    system_prompt: str,
    system_role: Literal["system", "developer"],
) -> EasyInputMessageParam:
    return {
        "role": system_role,
        "content": [_build_input_text(system_prompt)],
    }


def _serialize_user_message(
    message: UserMessage,
) -> EasyInputMessageParam:
    return {
        "role": "user",
        "content": [_build_input_text(message.content)],
    }


def _serialize_assistant_turn(
    turn: AssistantTurn,
    *,
    assistant_turn_index: int,
) -> ResponseInputParam:
    items: ResponseInputParam = []

    for block_index, block in enumerate(turn.content):
        match block:
            case AssistantReasoningBlock(reasoning_id=reasoning_id) if (
                reasoning_id is not None
            ):
                items.append(
                    _serialize_assistant_reasoning_block(
                        block,
                        reasoning_id=reasoning_id,
                        assistant_turn_index=assistant_turn_index,
                        block_index=block_index,
                    )
                )
            case AssistantTextBlock():
                items.append(
                    _serialize_assistant_text_block(
                        block,
                        assistant_turn_index=assistant_turn_index,
                        block_index=block_index,
                    )
                )

    return items


def _serialize_assistant_reasoning_block(
    block: AssistantReasoningBlock,
    *,
    reasoning_id: str,
    assistant_turn_index: int,
    block_index: int,
) -> ResponseReasoningItemParam:
    summary: ResponseReasoningSummaryParam = {
        "type": "summary_text",
        "text": block.summary_text,
    }
    reasoning_item: ResponseReasoningItemParam = {
        "type": "reasoning",
        "id": reasoning_id,
        "summary": [summary],
    }
    if block.encrypted_content is not None:
        reasoning_item["encrypted_content"] = block.encrypted_content
    return reasoning_item


def _serialize_assistant_text_block(
    block: AssistantTextBlock,
    *,
    assistant_turn_index: int,
    block_index: int,
) -> ResponseOutputMessageParam:
    output_text: ResponseOutputTextParam = {
        "type": "output_text",
        "text": block.text,
        "annotations": [],
    }
    message: ResponseOutputMessageParam = {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "id": block.message_id
        or _build_fallback_message_id(assistant_turn_index, block_index),
        "content": [output_text],
    }
    if block.phase is not None:
        message["phase"] = block.phase
    return message


def _build_input_text(
    text: str,
) -> ResponseInputTextParam:
    return {
        "type": "input_text",
        "text": text,
    }


def _build_fallback_message_id(
    assistant_turn_index: int,
    block_index: int,
) -> str:
    return f"msg_{assistant_turn_index}_{block_index}"
