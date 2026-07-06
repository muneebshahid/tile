"""Tool contracts shared by agents and AI providers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, SerializeAsAny, field_validator

JsonObject: TypeAlias = dict[str, JsonValue]
ImageMimeType: TypeAlias = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ToolDetails(BaseModel):
    """Open base for tool-specific result metadata beyond model-visible content."""

    model_config = ConfigDict(extra="allow")


class ToolTextContent(BaseModel):
    """Text content returned by a tool."""

    type: Literal["text"] = "text"
    text: str


class ToolImageContent(BaseModel):
    """Base64-encoded image content returned by a tool."""

    type: Literal["image"] = "image"
    data: str
    mime_type: ImageMimeType


ToolResultContent: TypeAlias = ToolTextContent | ToolImageContent


class ToolResult(BaseModel):
    """Provider-neutral tool execution result."""

    content: list[ToolResultContent]
    details: SerializeAsAny[ToolDetails] | None = None

    @classmethod
    def text(
        cls,
        text: str,
        *,
        details: ToolDetails | None = None,
    ) -> ToolResult:
        """Create a text-only tool result."""

        return cls(content=[ToolTextContent(text=text)], details=details)

    @classmethod
    def image(
        cls,
        text: str,
        image: ToolImageContent,
        *,
        details: ToolDetails | None = None,
    ) -> ToolResult:
        """Create an image tool result with an explanatory text block."""

        return cls(content=[ToolTextContent(text=text), image], details=details)


ToolFunctionResult: TypeAlias = ToolResult
ToolFunction: TypeAlias = Callable[..., Awaitable[ToolFunctionResult]]


class ToolDefinition(BaseModel):
    """A provider-agnostic function tool definition."""

    name: str
    description: str
    input_schema: JsonObject
    defer_loading: bool = False
    fn: ToolFunction

    @field_validator("name")
    @classmethod
    def _require_clean_name(cls, name: str) -> str:
        """Reject empty or whitespace-padded tool names at registration."""

        if not name or name != name.strip():
            raise ValueError(
                "Tool name must be non-empty without surrounding whitespace."
            )
        return name
