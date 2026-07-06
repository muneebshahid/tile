"""Built-in tools for the default agent."""

from functools import partial
from pathlib import Path
from typing import cast

from ori.types.tools import ToolDefinition, ToolFunction
from ori.tools.bash import tool as bash_tool
from ori.tools.details import (
    BashDetails,
    EditDetails,
    FindDetails,
    GrepDetails,
    LsDetails,
    ReadDetails,
    ToolOutputDetails,
)
from ori.tools.edit import tool as edit_tool
from ori.tools.find import tool as find_tool
from ori.tools.grep import tool as grep_tool
from ori.tools.ls import tool as ls_tool
from ori.tools.support.paths import normalize_cwd
from ori.tools.read import tool as read_tool
from ori.tools.write import tool as write_tool

__all__ = [
    "BashDetails",
    "EditDetails",
    "FindDetails",
    "GrepDetails",
    "LsDetails",
    "ReadDetails",
    "ToolOutputDetails",
    "build_tools",
    "tools",
]

_TOOLS = [read_tool, bash_tool, edit_tool, grep_tool, find_tool, ls_tool, write_tool]


def build_tools(cwd: Path | str) -> list[ToolDefinition]:
    """Build the default tool list with a shared working directory bound in."""

    normalized_cwd = normalize_cwd(cwd)
    return [_bind_cwd(tool, normalized_cwd) for tool in _TOOLS]


def _bind_cwd(tool: ToolDefinition, cwd: Path) -> ToolDefinition:
    """Return a copy of a tool whose function receives the supplied cwd."""

    fn = cast(ToolFunction, partial(tool.fn, cwd=cwd))
    return tool.model_copy(update={"fn": fn})


tools = build_tools(Path.cwd())
