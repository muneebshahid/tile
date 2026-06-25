"""Example local runner for one headless Ori prompt."""

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from ori import AgentRuntime, HistoryStore
from ori.events import AgentEvent, StreamFn
from ori.openai import stream_api
from ori.tools import build_tools
from ori.types import ToolDefinition
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
    history_store: HistoryStore | None = None,
    cwd: Path | str | None = None,
    output: TextIO | None = None,
) -> None:
    """Run one prompt through a runtime session and write JSON event lines."""

    working_directory = _resolve_cwd(cwd)
    active_tools = (
        tuple(tools) if tools is not None else tuple(build_tools(working_directory))
    )
    runtime = AgentRuntime(
        stream_fn=stream_fn,
        model=model or settings.openai_model,
        history_store=history_store,
        tools=active_tools,
        cwd=working_directory,
    )
    session = runtime.session(name="local-runner")
    event_output = output or sys.stdout

    async for event in session.prompt(prompt):
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
