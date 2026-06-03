"""Tool contracts shared by agents and AI providers."""

from collections.abc import Awaitable, Callable
from typing import Literal, TypeAlias

from pydantic import BaseModel, JsonValue

JsonObject: TypeAlias = dict[str, JsonValue]
ImageMimeType: TypeAlias = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
ToolTruncationReason: TypeAlias = Literal["lines", "bytes"]
ToolTruncationKeep: TypeAlias = Literal["head", "tail"]


class ToolTruncationDetails(BaseModel):
    """Structured metadata describing tool output truncation."""

    reason: ToolTruncationReason
    keep: ToolTruncationKeep
    line_limit: int
    byte_limit: int
    lines_returned: int
    bytes_returned: int
    total_lines: int
    total_bytes: int
    edge_line_exceeds_limit: bool = False


class LsDetails(BaseModel):
    """Directory listing metadata for UI and persistence."""

    type: Literal["ls"] = "ls"
    path: str
    truncation: ToolTruncationDetails | None = None


ToolResultDetails: TypeAlias = LsDetails


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
    details: ToolResultDetails | None = None

    @classmethod
    def text(
        cls,
        text: str,
        *,
        details: ToolResultDetails | None = None,
    ) -> "ToolResult":
        """Create a text-only tool result."""

        return cls(content=[ToolTextContent(text=text)], details=details)

    @classmethod
    def image(
        cls,
        text: str,
        image: ToolImageContent,
        *,
        details: ToolResultDetails | None = None,
    ) -> "ToolResult":
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
