"""Shared tool definition builders for tests."""

from ori.types.tools import JsonObject, ToolDefinition, ToolFunction


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
        input_schema=_city_input_schema(city_description),
        fn=fn,
    )


def _city_input_schema(city_description: str | None) -> JsonObject:
    """Build the object schema for a single required city string parameter."""

    city_schema: JsonObject = {"type": "string"}
    if city_description is not None:
        city_schema["description"] = city_description
    return {
        "type": "object",
        "properties": {"city": city_schema},
        "required": ["city"],
        "additionalProperties": False,
    }
