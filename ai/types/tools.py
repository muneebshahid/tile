"""Tool contracts shared by agents and AI providers."""

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from pydantic import BaseModel, JsonValue

JsonObject: TypeAlias = dict[str, JsonValue]
ToolFunction: TypeAlias = Callable[..., Awaitable[JsonValue]]


class ToolDefinition(BaseModel):
    """A provider-agnostic function tool definition."""

    name: str
    description: str
    input_schema: JsonObject
    defer_loading: bool = False
    fn: ToolFunction
