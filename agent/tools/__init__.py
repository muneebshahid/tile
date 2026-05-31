"""Built-in tools for the default agent."""

from agent.tools.find import tool as find_tool
from agent.tools.grep import tool as grep_tool
from agent.tools.ls import tool as ls_tool
from agent.tools.read import tool as read_tool
from agent.tools.write import tool as write_tool

tools = [read_tool, grep_tool, find_tool, ls_tool, write_tool]
