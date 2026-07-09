"""Result tool that ends an output-contract run with a validated result."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, JsonValue

from tile.result import COMPLETE_TOOL_NAME
from tile.types.tools import JsonObject, ToolDefinition, ToolResult


def tool(result: type[BaseModel]) -> ToolDefinition:
    """Build a complete tool that validates results against one schema."""

    input_schema = strict_object_schema(result.model_json_schema())
    _require_all_fields_required(input_schema)

    async def complete(**arguments: JsonValue) -> ToolResult:
        """Validate the run's final result against the required schema."""

        result.model_validate(arguments)
        return ToolResult.text("Result recorded.")

    return ToolDefinition(
        name=COMPLETE_TOOL_NAME,
        description=(
            "Report the final result and end the run. The arguments are "
            "validated against the required result schema; validation errors "
            "are returned for correction. Call this exactly once, when the "
            "task is done."
        ),
        input_schema=input_schema,
        fn=complete,
    )


def strict_object_schema(schema: JsonObject) -> JsonObject:
    """Close every object node in a JSON schema to undeclared properties."""

    return cast(JsonObject, _close_objects(schema))


def _close_objects(node: JsonValue) -> JsonValue:
    """Recursively add additionalProperties: false to object schema nodes."""

    if isinstance(node, dict):
        closed = {key: _close_objects(value) for key, value in node.items()}
        if closed.get("type") == "object" and "additionalProperties" not in closed:
            closed["additionalProperties"] = False
        return closed
    if isinstance(node, list):
        return [_close_objects(item) for item in node]
    return node


def _require_all_fields_required(node: JsonValue, path: str = "") -> None:
    """Reject schemas whose object nodes leave declared properties optional.

    Provider strict modes demand that every property appears in ``required``.
    Pydantic omits fields that carry defaults, so a result model with a
    default would be rejected by the provider on every request. Optional
    result fields must be expressed as nullable (`| None`) without a default.
    """

    if isinstance(node, list):
        for index, item in enumerate(node):
            _require_all_fields_required(item, f"{path}[{index}]")
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        required = node.get("required")
        declared = set(required) if isinstance(required, list) else set()
        missing = sorted(set(properties) - declared)
        if missing:
            location = path or "the result schema"
            raise ValueError(
                f"Optional fields in {location}: {', '.join(missing)}. "
                "Result models must not use field defaults; make optional "
                "fields nullable (`| None`) instead."
            )
    for key, value in node.items():
        _require_all_fields_required(value, f"{path}.{key}" if path else key)
