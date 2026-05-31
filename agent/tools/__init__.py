"""Built-in tools for the default agent."""

from functools import partial
from pathlib import Path
from typing import cast

from ai.types.tools import ToolDefinition, ToolFunction
from agent.tools.bash import tool as bash_tool
from agent.tools.find import tool as find_tool
from agent.tools.grep import tool as grep_tool
from agent.tools.ls import tool as ls_tool
from agent.tools.paths import normalize_cwd
from agent.tools.read import tool as read_tool
from agent.tools.write import tool as write_tool

_TOOLS = [read_tool, bash_tool, grep_tool, find_tool, ls_tool, write_tool]


def build_tools(cwd: Path | str) -> list[ToolDefinition]:
    """Build the default tool list with a shared working directory bound in."""

    normalized_cwd = normalize_cwd(cwd)
    return [_bind_cwd(tool, normalized_cwd) for tool in _TOOLS]


def _bind_cwd(tool: ToolDefinition, cwd: Path) -> ToolDefinition:
    """Return a copy of a tool whose function receives the supplied cwd."""

    fn = cast(ToolFunction, partial(tool.fn, cwd=cwd))
    return tool.model_copy(update={"fn": fn})


tools = build_tools(Path.cwd())
