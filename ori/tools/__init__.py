"""Built-in tools for the default agent."""

from functools import partial
from pathlib import Path
from typing import cast

from ori.types.tools import ToolDefinition, ToolFunction
from ori.tool_truncation import ToolOutputDetails
from ori.tools.bash import BashDetails, tool as bash_tool
from ori.tools.edit import EditDetails, tool as edit_tool
from ori.tools.find import FindDetails, tool as find_tool
from ori.tools.grep import GrepDetails, tool as grep_tool
from ori.tools.ls import LsDetails, tool as ls_tool
from ori.tools.support.paths import normalize_cwd
from ori.tools.read import ReadDetails, tool as read_tool
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
