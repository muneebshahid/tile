"""Shared tool definition builders for tests."""

from pydantic import Field, create_model

from tile.types.tools import ToolDefinition, ToolFunction, ToolInput


def city_tool(
    name: str,
    description: str,
    fn: ToolFunction,
    *,
    city_description: str | None = None,
) -> ToolDefinition:
    """Build a deterministic tool definition with a single city parameter."""

    return ToolDefinition(
        name=name,
        description=description,
        input_model=_city_input_model(city_description),
        fn=fn,
    )


def _city_input_model(city_description: str | None) -> type[ToolInput]:
    """Build an input model for a single required city string parameter."""

    return create_model(
        "CityInput",
        __base__=ToolInput,
        city=(str, Field(description=city_description)),
    )
