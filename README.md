# Tile

[![CI](https://github.com/muneebshahid/tile/actions/workflows/ci.yml/badge.svg)](https://github.com/muneebshahid/tile/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A compact, Python-native runtime for headless, tool-using agent sessions.

Tile is a **runtime, not a framework**. You construct the pieces — a provider
client, a tool list, a working directory, optionally a history store — and hand
them to `AgentRuntime`. It runs prompt-driven agent sessions on top of them:
provider streaming, a tool-execution loop, typed run outcomes, and session
history. There are no plugins or global configuration. The
provider stream, model, tools, and working directory are explicit runtime
inputs; conversation history defaults to an in-memory store, and applications
can supply a `HistoryStore` when persistence is required. Embed it in an
application, or build a service on top.

**Status: 0.x.** APIs change without deprecation cycles. OpenAI (Responses API)
is the only provider today; more are planned. Requires Python 3.13+.

## Why a runtime?

Tile owns the lifecycle around an agent loop:

- a prompt becomes a task-owned `Run`;
- execution continues independently of event subscribers;
- a `Session` owns model-visible conversation history;
- providers normalize into one event and history contract;
- prompts may require explicit, typed success or failure outcomes.

Tile does not provide graphs, teams, workflows, memory/RAG, a UI, or a
deployment platform. Applications compose those concerns around the runtime.

## Install

```bash
pip install tile-runtime
```

The distribution is `tile-runtime`; the import name is `tile`.

## Quickstart

With `OPENAI_API_KEY` set:

```python
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from tile import AgentRuntime
from tile.providers.openai import create_stream_api
from tile.tools import BUILTIN_TOOLS


async def main() -> None:
    runtime = AgentRuntime(
        stream_fn=create_stream_api(AsyncOpenAI()),
        model="gpt-5.4",
        tools=BUILTIN_TOOLS,
        cwd=Path.cwd(),
    )
    session = runtime.session(name="quickstart")
    run = await session.prompt("List the files in the current directory.")
    print(await run.wait())  # "completed"
    print(run.output_text)


asyncio.run(main())
```

`cwd` is required and is the runtime's single working directory: it is
announced to the model in the system prompt and injected into every tool whose
function declares a `cwd` parameter. `BUILTIN_TOOLS` (`read`, `bash`, `edit`,
`grep`, `find`, `ls`, `write`) are plain, unbound definitions — the runtime
binds them. Tool inputs are Pydantic models: Tile generates the provider schema
from the model and validates every model-supplied call before invocation.
A custom tool opts into the working directory the same way:

```python
from pathlib import Path

from pydantic import Field

from tile.types import ToolDefinition, ToolInput, ToolResult


class SearchInput(ToolInput):
    query: str = Field(description="Text to search for.")


async def search(query: str, cwd: Path) -> ToolResult:
    ...  # cwd is injected by the runtime, not exposed to the model


search_tool = ToolDefinition(
    name="search",
    description="Search the current workspace.",
    input_model=SearchInput,
    fn=search,
)
```

`ToolInput` rejects wrong types and extra fields. Validation errors are returned
to the model for correction. A tool may return `ToolResult.error(...)` for an
expected failure; an exception that escapes the tool is normalized by the
runtime as an invocation failure.

Prompt execution is task-owned: `session.prompt(...)` submits a run and returns
a handle immediately, the runtime drives it to completion, and any number of
subscribers can observe the event stream.

```python
run = await session.prompt("Inspect the current repository")
async for event in run.events():
    ...
status = await run.wait()  # "completed" | "failed" | "aborted"
```

Run events are currently replayable in process while the `Run` handle exists.
Conversation history can be persisted with SQLite. Durable run records,
cross-process event replay, approval resumption, and service mode are planned,
not current capabilities.

## Typed results

Pass a pydantic model to get a validated result object back instead of prose to
parse:

```python
from pydantic import BaseModel

from tile import Completed, Failed


class WeatherReport(BaseModel):
    city: str
    temp_c: float
    summary: str


run = await session.prompt("What's the weather in Munich?", result=WeatherReport)
await run.wait()
match run.outcome:
    case Completed(value=report):
        print(report.city, report.temp_c)   # a WeatherReport instance
    case Failed(reason=reason):
        print("model declared failure:", reason)
```

For that prompt only, the runtime registers a `complete` tool (whose schema is
your model) and a `fail(reason)` tool, and instructs the model to end the run
through one of them. Validation errors route back to the model as ordinary tool
errors for correction; a run that ends in plain text is reminded to deliver,
a bounded number of times. The names `complete` and `fail` are reserved —
caller tools may not use them.

**Designing result schemas:** demand judgment, not transcripts. The result
should be the model's *verdict* — small, typed fields it decides — not a
container for data your tools already produced (bulk data belongs on
`ToolResult.details`). Add a `summary: str` field when you want guaranteed
prose alongside the structure.

**Prompt caching:** reuse one `result=` schema per session. The result tools
and contract text sit at the front of every provider request, so alternating
typed and plain prompts — or switching schemas — within a session re-reads the
whole session history at full price on each flip.

## Status and outcome

`run.status` says whether the run *executed*; `run.outcome` says what the task
*concluded*. `outcome` is non-`None` exactly when `status == "completed"`.

| Run ending | `status` | `outcome` |
|---|---|---|
| Plain prompt, text answer | `completed` | `Completed(value=text)` |
| `complete` validates | `completed` | `Completed(value=model instance)` |
| `fail(reason)` | `completed` | `Failed(reason)` |
| Reminder cap exhausted | `completed` | `Failed(reason=...)` |
| Provider dies (stream error or raise) | `failed` | `None` — see `run.error_message` |
| Aborted | `aborted` | `None` |

A provider death never corrupts the session: partial turns are dropped, history
ends at the last stable item, unanswered tool calls are healed, and the session
accepts the next prompt immediately. Tile does not retry; request-level retries
belong to the `AsyncOpenAI` client you construct (`max_retries`), and the
recovery unit above that is re-prompting the session.

## Public API

Use the package facades for application code. Deep module paths are internal
and may move as Tile grows.

```python
from tile import AgentRuntime, Completed, Failed, InMemoryHistoryStore, Run
from tile.events import AgentEvent, MessageEndEvent, StreamFn
from tile.providers.openai import create_stream_api
from tile.tools import BUILTIN_TOOLS
from tile.types import ToolDefinition, ToolInput, ToolResult
from tile.types import ToolInputValidationFailure, ToolInvocationFailure
```

`tile` exposes the runtime, session, run-handle, outcome, history-store, and
runtime-error contracts. `tile.events` exposes the structured events yielded by
`Run.events()`. `tile.types` exposes provider-neutral conversation, stream, and
tool contracts, including structured validation and invocation failures on
tool-execution event details. `tile.providers.openai` exposes
`create_stream_api`, which
binds a caller-constructed `AsyncOpenAI` client and optional provider reasoning
options to the runtime's stream-function contract:
`create_stream_api(AsyncOpenAI(...), reasoning={"effort": "medium"})`.

## Architecture

```
tile/
├── history/         # Session metadata and conversation history stores
├── providers/       # Provider integrations (OpenAI today)
├── tools/           # Built-in local tool implementations
├── types/           # Provider-neutral contracts for conversations and tools
├── agent.py         # Stateless agent loop: provider turns and tool batches
├── events.py        # Runtime event contracts
├── prompt.py        # System prompt composition
├── result.py        # Typed run outcomes and the output-contract protocol
└── runtime.py       # Session runtime facade: policy, persistence, runs
tests/               # Test suite
```

## Security posture

Tile's built-in tools are deliberately unconfined. `bash` executes arbitrary
shell commands with the permissions of the process running the agent, and the
file tools accept absolute paths — the session working directory is a default,
not a sandbox. Run Tile only where you would run the model's commands yourself,
and use OS-level isolation such as a container or VM when you need a boundary.
Resource exhaustion from trusted local input is out of scope for now. Tool
authorization and first-class approval are planned, not current capabilities.

## Development

```bash
uv sync         # install dependencies
make test       # pytest
make format     # ruff
make type_check # ty
```

Run the example CLI against the current directory:

```bash
uv run python -m examples.local_runner "Inspect the current repository"
```
