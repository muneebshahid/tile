"""Example local runner for one headless piy prompt."""

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from agent.agent import run_agent
from agent.tools import build_tools
from agent.types import AgentEvent, StreamFn
from ai.openai.provider import stream_api
from ai.types.conversation import UserMessage
from ai.types.tools import ToolDefinition
from settings import settings


def main() -> None:
    """Run the example local runner."""

    raise SystemExit(asyncio.run(run_cli(sys.argv[1:])))


async def run_cli(argv: Sequence[str]) -> int:
    """Run a prompt from command arguments or standard input."""

    prompt = _read_prompt(argv, sys.stdin)
    if not prompt:
        print("Provide a prompt as arguments or stdin.", file=sys.stderr)
        return 2

    await run_prompt(prompt)
    return 0


async def run_prompt(
    prompt: str,
    *,
    stream_fn: StreamFn = stream_api,
    model: str | None = None,
    tools: Sequence[ToolDefinition] | None = None,
    cwd: Path | str | None = None,
    output: TextIO | None = None,
) -> None:
    """Run one prompt through the stateless agent and write JSON event lines."""

    working_directory = _resolve_cwd(cwd)
    active_tools = (
        tuple(tools) if tools is not None else tuple(build_tools(working_directory))
    )
    history = [UserMessage(content=prompt)]
    event_output = output or sys.stdout

    async for event in run_agent(
        history,
        stream_fn=stream_fn,
        model=model or settings.openai_model,
        tools=active_tools,
        cwd=working_directory,
    ):
        event_output.write(_serialize_event(event))
        event_output.write("\n")


def _read_prompt(argv: Sequence[str], stdin: TextIO) -> str:
    """Read a prompt from positional arguments or standard input."""

    if argv:
        return " ".join(argv).strip()
    return stdin.read().strip()


def _resolve_cwd(cwd: Path | str | None) -> Path:
    """Resolve the local working directory for tools and instructions."""

    if cwd is None:
        return Path.cwd().resolve()
    return Path(cwd).expanduser().resolve()


def _serialize_event(event: AgentEvent) -> str:
    """Serialize one agent event for line-oriented local output."""

    return event.model_dump_json()


if __name__ == "__main__":
    main()
