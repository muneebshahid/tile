"""Shared tool definition builders for tests."""

from pydantic import Field

from tile.types.tools import ToolDefinition, ToolFunction, ToolInput


class CityInput(ToolInput):
    """Strict city input shared by deterministic test tools."""

    city: str = Field(description="The city to look up.")


def city_tool(
    name: str,
    description: str,
    fn: ToolFunction,
) -> ToolDefinition:
    """Build a deterministic tool definition with a single city parameter."""

    return ToolDefinition(
        name=name,
        description=description,
        input_model=CityInput,
        fn=fn,
    )
