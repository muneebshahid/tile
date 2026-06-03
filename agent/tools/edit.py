"""File edit tool scaffold for the default agent."""

import re
from pathlib import Path

from ai.types.tools import ToolDefinition, ToolResult
from agent.tools.paths import resolve_to_cwd

UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")


async def fn(path: str, edits: list[dict[str, str]], *, cwd: Path) -> ToolResult:
    """Edit a file with one or more targeted text replacements."""

    resolved_path = _resolve_path(path, cwd)
    _ = resolved_path
    _ = edits
    raise NotImplementedError("edit execution is not implemented yet.")


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a path with forgiving user-input path variants."""

    normalized_path = _normalize_unicode_spaces(_normalize_at_prefix(path))
    return resolve_to_cwd(normalized_path, cwd)


def _normalize_at_prefix(path: str) -> str:
    """Strip a leading at sign used when users paste referenced paths."""

    if path.startswith("@"):
        return path[1:]
    return path


def _normalize_unicode_spaces(path: str) -> str:
    """Normalize uncommon Unicode spaces to ordinary spaces."""

    return UNICODE_SPACES.sub(" ", path)


tool = ToolDefinition(
    name="edit",
    description=(
        "Edit a single file using exact text replacement. Every edits[].oldText "
        "must match a unique, non-overlapping region of the original file. If "
        "two changes affect the same block or nearby lines, merge them into one "
        "edit instead of emitting overlapping edits. Do not include large "
        "unchanged regions just to connect distant changes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative or absolute).",
            },
            "edits": {
                "type": "array",
                "description": (
                    "One or more targeted replacements. Each edit is matched "
                    "against the original file, not incrementally. Do not include "
                    "overlapping or nested edits. If two changes touch the same "
                    "block or nearby lines, merge them into one edit instead."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": (
                                "Exact text for one targeted replacement. It "
                                "must be unique in the original file and must "
                                "not overlap with any other edits[].oldText in "
                                "the same call."
                            ),
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text for this targeted edit.",
                        },
                    },
                    "required": ["oldText", "newText"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["path", "edits"],
        "additionalProperties": False,
    },
    fn=fn,
)
