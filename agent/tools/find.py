"""File path search tool scaffold for the default agent."""

from pathlib import Path

from pydantic import BaseModel

from ai.types.tools import ToolDefinition, ToolResult
from agent.tools.executables import execute, require_executable
from agent.tools.truncation import OUTPUT_BYTE_LIMIT_LABEL, truncate_head


class Results(BaseModel):
    """Structured file path search results returned by the find tool."""

    paths: list[str]


async def fn(
    pattern: str,
    path: str = ".",
    limit: int = 1000,
    *,
    cwd: Path,
) -> ToolResult:
    """Find file paths matching a glob pattern."""

    limit = max(1, limit)
    executable = require_executable("fd", "fd")
    args = _build_args(pattern, path, limit)
    output = await execute(executable, args, cwd=cwd)
    results = _parse_output(output)
    return ToolResult.text(_format_results(results, limit))


def _build_args(
    pattern: str,
    path: str,
    limit: int,
) -> list[str]:
    """Build command arguments for a file path search."""

    args = [
        "--glob",
        "--color=never",
        "--hidden",
        "--no-require-git",
        "--max-results",
        str(limit + 1),
    ]

    effective_pattern = _build_effective_pattern(pattern)
    if _matches_full_path(pattern):
        args.append("--full-path")

    args.extend(["--", effective_pattern, path])
    return args


def _parse_output(output: str) -> Results:
    """Parse fd stdout into structured file path search results."""

    paths = [_normalize_path(line) for line in output.splitlines() if line]
    return Results(paths=paths)


def _format_results(results: Results, limit: int) -> str:
    """Format file path search results as compact plain text."""

    if not results.paths:
        return "No files found matching pattern"

    truncation = truncate_head("\n".join(results.paths), max_lines=limit)
    output = truncation.content

    notices: list[str] = []
    if truncation.truncated_by == "lines":
        notices.append(
            f"{limit} results limit reached. "
            f"Use limit={limit * 2} for more, or refine pattern"
        )
    if truncation.truncated_by == "bytes":
        notices.append(f"{OUTPUT_BYTE_LIMIT_LABEL} limit reached")
    if notices:
        output += f"\n\n[{'. '.join(notices)}]"
    return output


def _build_effective_pattern(pattern: str) -> str:
    """Return the pattern adjusted for fd full-path matching."""

    if not _matches_full_path(pattern) or pattern.startswith("**/"):
        return pattern

    if pattern.startswith("/"):
        return f"**{pattern}"

    return f"**/{pattern}"


def _normalize_path(path: str) -> str:
    """Normalize one fd output path for compact display."""

    normalized = path.rstrip("\r").replace("\\", "/")
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def _matches_full_path(pattern: str) -> bool:
    """Return whether a glob pattern should match candidate paths."""

    return "/" in pattern


tool = ToolDefinition(
    name="find",
    description="Search for files by glob pattern.",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match file paths, for example '*.py' or 'src/**/*.py'.",
            },
            "path": {
                "type": "string",
                "description": "The directory path to search. Defaults to the current directory.",
            },
            "limit": {
                "type": "integer",
                "description": "The maximum number of file paths to return. Defaults to 1000.",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    fn=fn,
)
