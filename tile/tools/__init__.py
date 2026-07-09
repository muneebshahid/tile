"""Built-in tools for the default agent."""

from functools import partial
from pathlib import Path
from typing import cast

from tile.types.tools import ToolDefinition, ToolFunction
from tile.tool_truncation import ToolOutputDetails
from tile.tools.bash import BashDetails, tool as bash_tool
from tile.tools.complete import CompleteDetails, tool as complete_tool
from tile.tools.edit import EditDetails, tool as edit_tool
from tile.tools.fail import FailDetails, tool as fail_tool
from tile.tools.find import FindDetails, tool as find_tool
from tile.tools.grep import GrepDetails, tool as grep_tool
from tile.tools.ls import LsDetails, tool as ls_tool
from tile.tools.support.paths import normalize_cwd
from tile.tools.read import ReadDetails, tool as read_tool
from tile.tools.write import tool as write_tool

__all__ = [
    "BashDetails",
    "CompleteDetails",
    "EditDetails",
    "FailDetails",
    "FindDetails",
    "GrepDetails",
    "LsDetails",
    "ReadDetails",
    "ToolOutputDetails",
    "build_tools",
    "complete_tool",
    "fail_tool",
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
