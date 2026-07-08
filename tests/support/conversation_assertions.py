"""Shared assertions for provider-neutral conversation history tests."""

from tile.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)


def expect_user_message(item: ConversationItem) -> UserMessage:
    """Assert and return a user conversation item."""

    assert isinstance(item, UserMessage)
    return item


def expect_assistant_turn(item: ConversationItem) -> AssistantTurn:
    """Assert and return an assistant conversation item."""

    assert isinstance(item, AssistantTurn)
    return item


def expect_tool_result_turn(item: ConversationItem) -> ToolResultTurn:
    """Assert and return a tool-result conversation item."""

    assert isinstance(item, ToolResultTurn)
    return item
