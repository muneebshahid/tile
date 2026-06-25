"""Tool contracts shared by agents and AI providers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, TypeAlias

from pydantic import BaseModel, JsonValue

from ori.tool_truncation import Truncation, TruncationKeep, TruncationReason

JsonObject: TypeAlias = dict[str, JsonValue]
ImageMimeType: TypeAlias = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
ToolTruncationReason: TypeAlias = TruncationReason
ToolTruncationKeep: TypeAlias = TruncationKeep


class ToolOutputDetails(BaseModel):
    """Structured metadata describing bounded tool output."""

    truncated: bool
    truncated_by: ToolTruncationReason | None
    keep: ToolTruncationKeep
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    edge_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    @classmethod
    def from_truncation(
        cls,
        truncation: Truncation,
    ) -> ToolOutputDetails:
        """Create output details from matching truncation metadata."""

        return cls(
            truncated=truncation.truncated,
            truncated_by=truncation.truncated_by,
            keep=truncation.keep,
            total_lines=truncation.total_lines,
            total_bytes=truncation.total_bytes,
            output_lines=truncation.output_lines,
            output_bytes=truncation.output_bytes,
            edge_line_exceeds_limit=truncation.edge_line_exceeds_limit,
            max_lines=truncation.max_lines,
            max_bytes=truncation.max_bytes,
        )


class LsDetails(BaseModel):
    """Directory listing metadata for UI and persistence."""

    type: Literal["ls"] = "ls"
    output: ToolOutputDetails


class GrepDetails(BaseModel):
    """Search metadata for UI and persistence."""

    type: Literal["grep"] = "grep"
    output: ToolOutputDetails
    match_limit_reached: int | None = None
    lines_truncated: bool = False


class FindDetails(BaseModel):
    """File path search metadata for UI and persistence."""

    type: Literal["find"] = "find"
    output: ToolOutputDetails


class ReadDetails(BaseModel):
    """File read metadata for UI and persistence."""

    type: Literal["read"] = "read"
    output: ToolOutputDetails


class BashDetails(BaseModel):
    """Shell command metadata for UI and persistence."""

    type: Literal["bash"] = "bash"
    output: ToolOutputDetails


class EditDetails(BaseModel):
    """File edit metadata for UI and persistence."""

    type: Literal["edit"] = "edit"
    diff: str


ToolResultDetails: TypeAlias = (
    LsDetails | GrepDetails | FindDetails | ReadDetails | BashDetails | EditDetails
)


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
    ) -> ToolResult:
        """Create a text-only tool result."""

        return cls(content=[ToolTextContent(text=text)], details=details)

    @classmethod
    def image(
        cls,
        text: str,
        image: ToolImageContent,
        *,
        details: ToolResultDetails | None = None,
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
