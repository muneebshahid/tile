"""Shared tool definition builders for tests."""

from pydantic import Field, create_model

from tile.types.tools import ToolDefinition, ToolFunction, ToolInput


class CityInput(ToolInput):
    """Strict city input shared by deterministic test tools."""

    city: str


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

    if city_description is None:
        return CityInput
    return create_model(
        "CityInput",
        __base__=CityInput,
        city=(str, Field(description=city_description)),
    )
