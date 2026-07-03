"""Serialization helpers for provider-neutral conversation history."""

from pydantic import TypeAdapter

from ori.types.conversation import ConversationItem

_CONVERSATION_ITEM_ADAPTER: TypeAdapter[ConversationItem] = TypeAdapter(
    ConversationItem
)


def dump_conversation_item(item: ConversationItem) -> str:
    """Serialize one model-visible conversation item as JSON."""

    return item.model_dump_json()


def load_conversation_item(payload_json: str) -> ConversationItem:
    """Deserialize one model-visible conversation item from JSON."""

    return _CONVERSATION_ITEM_ADAPTER.validate_json(payload_json)
