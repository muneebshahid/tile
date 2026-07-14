"""Tool contracts shared by agents and AI providers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, Self, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    PrivateAttr,
    SerializeAsAny,
    field_validator,
    model_validator,
)

JsonObject: TypeAlias = dict[str, JsonValue]
ImageMimeType: TypeAlias = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ToolDetails(BaseModel):
    """Open base for tool-specific result metadata beyond model-visible content."""

    model_config = ConfigDict(extra="allow")

    type: str


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


class ToolInput(BaseModel):
    """Strict base for model-controlled tool arguments."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ToolResult(BaseModel):
    """Provider-neutral tool execution result."""

    content: list[ToolResultContent]
    details: SerializeAsAny[ToolDetails] | None = None
    is_error: bool = False
    terminate: bool = False

    @model_validator(mode="after")
    def _reject_terminating_error(self) -> Self:
        """Prevent an error result from terminating the agent run."""

        if self.is_error and self.terminate:
            raise ValueError("A tool error cannot terminate the run.")
        return self

    @classmethod
    def text(
        cls,
        text: str,
        *,
        details: ToolDetails | None = None,
        terminate: bool = False,
    ) -> ToolResult:
        """Create a text-only tool result."""

        return cls(
            content=[ToolTextContent(text=text)],
            details=details,
            terminate=terminate,
        )

    @classmethod
    def image(
        cls,
        text: str,
        image: ToolImageContent,
        *,
        details: ToolDetails | None = None,
        terminate: bool = False,
    ) -> ToolResult:
        """Create an image tool result with an explanatory text block."""

        return cls(
            content=[ToolTextContent(text=text), image],
            details=details,
            terminate=terminate,
        )

    @classmethod
    def error(
        cls,
        text: str,
        *,
        details: ToolDetails | None = None,
    ) -> ToolResult:
        """Create a text error that the model may inspect and correct."""

        return cls(
            content=[ToolTextContent(text=text)],
            details=details,
            is_error=True,
        )


ToolFunction: TypeAlias = Callable[..., Awaitable[ToolResult]]


class ToolDefinition(BaseModel):
    """A provider-agnostic function tool definition."""

    name: str
    description: str
    input_model: type[BaseModel]
    defer_loading: bool = False
    fn: ToolFunction
    _input_schema: JsonObject = PrivateAttr()

    @property
    def input_schema(self) -> JsonObject:
        """Return the cached provider schema generated from the input model."""

        return self._input_schema

    @field_validator("name")
    @classmethod
    def _require_clean_name(cls, name: str) -> str:
        """Reject empty or whitespace-padded tool names at registration."""

        if not name or name != name.strip():
            raise ValueError(
                "Tool name must be non-empty without surrounding whitespace."
            )
        return name

    @model_validator(mode="after")
    def _cache_json_schema(self) -> Self:
        """Generate and cache the provider schema during tool construction."""

        schema = cast(JsonObject, self.input_model.model_json_schema())
        if schema.get("type") != "object":
            raise ValueError("Tool input models must generate an object schema.")
        self._input_schema = schema
        return self
