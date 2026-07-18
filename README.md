# Tile

[![CI](https://github.com/muneebshahid/tile/actions/workflows/ci.yml/badge.svg)](https://github.com/muneebshahid/tile/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A compact, Python-native runtime for headless, tool-using agent sessions.

Tile is a **runtime, not a framework**. You construct the pieces — a provider
client, a tool list, a working directory, and the history and run stores —
and hand them to `AgentRuntime`. It runs prompt-driven agent sessions on top
of them: provider streaming, a tool-execution loop, typed run outcomes, session
history, and durable run summaries. There are no plugins or global
configuration. The provider stream, model, working directory, and both stores
are explicit runtime inputs with no defaults; pass the in-memory stores for
process-lifetime state. Embed it in an application, or build a service on top.

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

from tile import AgentRuntime, InMemoryHistoryStore, InMemoryRunStore
from tile.providers.openai import create_stream_api
from tile.tools import BUILTIN_TOOLS


async def main() -> None:
    runtime = AgentRuntime(
        stream_fn=create_stream_api(AsyncOpenAI()),
        model="gpt-5.4",
        tools=BUILTIN_TOOLS,
        cwd=Path.cwd(),
        history_store=InMemoryHistoryStore(),
        run_store=InMemoryRunStore(),
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

from tile.types import ToolDefinition, ToolError, ToolInput, ToolResult


class SearchInput(ToolInput):
    query: str = Field(description="Text to search for.")


async def search(params: SearchInput, *, cwd: Path) -> ToolResult:
    if not params.query:
        raise ToolError("A search query is required.")
    ...  # cwd is injected and never exposed to the model


search_tool = ToolDefinition(
    name="search",
    description="Search the current workspace.",
    input_model=SearchInput,
    fn=search,
)
```

`ToolInput` rejects wrong types and extra fields. Tile passes the validated model
instance directly to the tool, preserving nested models, aliases, and defaults.
Validation errors are returned to the model for correction. Tool functions
return `ToolResult` only for success and raise `ToolError` for intentional,
model-visible failures. Any other exception is normalized as an unexpected
invocation failure, while cancellation continues to propagate.

Prompt execution is task-owned: `session.prompt(...)` submits a run and returns
a handle immediately, the runtime drives it to completion, and any number of
subscribers can observe the event stream.

```python
run = await session.prompt("Inspect the current repository")
async for event in run.events():
    ...
status = await run.wait()  # "completed" | "failed" | "aborted"
```

Every run's log begins with `RunStartEvent` and ends with exactly one
`RunEndEvent` carrying the run's terminal outcome, on every in-process
termination path. Inner events carry no such guarantee: a failure or
abort can tear the run down with inner scopes still open, and the run
end sweeps them — its outcome names why, exactly once. `run.wait()`
returns only after that closure, so waiters always observe a closed
log.

Run events are currently replayable in process while the `Run` handle exists.
Conversation history and run summaries can be persisted with SQLite.
Cross-process event replay, approval resumption, and service mode are planned,
not current capabilities.

## Durable run records

`HistoryStore` owns only model-visible conversation items. `RunStore` separately
owns one summary per submitted prompt: stable run and session IDs, execution
status, UTC start/end timestamps, configured model, provider identity, and one
typed terminal outcome carrying any structured failure cause. Provider identity
comes from the
stream function's declared `provider` attribute at submission; when a message
finalizes, the identity observed on the provider stream replaces it.

```python
from pathlib import Path

from openai import AsyncOpenAI

from tile import AgentRuntime, SQLiteHistoryStore, SQLiteRunStore
from tile.providers.openai import create_stream_api


database_path = Path("tile.db")
history_store = SQLiteHistoryStore(database_path)
run_store = SQLiteRunStore(database_path)
runtime = AgentRuntime(
    stream_fn=create_stream_api(AsyncOpenAI()),
    model="gpt-5.4",
    cwd=Path.cwd(),
    history_store=history_store,
    run_store=run_store,
)

session = runtime.session()
run = await session.prompt("Inspect this repository")
await run.wait()

record = runtime.get_run(run.id)
session_records = runtime.runs_for(session.id)
```

The SQLite stores are separate contracts and use separate tables and schema
version markers, even when they share one database file. A running record is
written before the prompt enters history, so a rejected submission never
leaves a user message without a run record; if submission fails after the
record exists, the record is finished with a `submission`-origin execution
failure. A run's terminal status and outcome are derived only from agent
execution; the terminal store write is best-effort bookkeeping. When that
write fails, the live `Run` handle keeps the true state and exposes the error
as `run.persistence_error`, while the store retains its last written state
and may report the run as `running`. A hard process death leaves the same
stale `running` record; automatic interruption classification and recovery
are outside this contract.

## Typed results

Pass a pydantic model to get a validated result object back instead of prose to
parse:

```python
from pydantic import BaseModel

from tile import AgentFailure, Completed, Failed


class WeatherReport(BaseModel):
    city: str
    temp_c: float
    summary: str


run = await session.prompt("What's the weather in Munich?", result=WeatherReport)
await run.wait()
match run.outcome:
    case Completed(value=report):
        print(report.city, report.temp_c)   # a WeatherReport instance
    case Failed(cause=AgentFailure(reason=reason)):
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
*concluded*. `outcome` is `None` only while the run is still running: every
terminal run carries exactly one outcome — `Completed`, `Failed`, or
`Aborted` — and a `Failed` outcome names its structured cause. An
`AgentFailure` cause is the model's own verdict that it could not deliver
(execution finished normally), while an `ExecutionFailure` cause means a
runtime boundary broke, with an explicit `origin` (`submission`, `turn`, or
`execution`), the exception type, and the message. The terminal status is
derived from the outcome variant, so the two can never contradict each other.
`run.failure` is shorthand for the `ExecutionFailure` cause when there is one,
`run.error_message` for its message, and `run.exception` retains the original
in-process exception for local debugging — it is not part of the serialized
contract.

| Run ending | `status` | `outcome` |
|---|---|---|
| Plain prompt, text answer | `completed` | `Completed(value=text)` |
| `complete` validates | `completed` | `Completed(value=model instance)` |
| `fail(reason)` | `completed` | `Failed(cause=AgentFailure(...))` |
| Reminder cap exhausted | `completed` | `Failed(cause=AgentFailure(...))` |
| Provider dies (stream error or raise) | `failed` | `Failed(cause=ExecutionFailure(...))` |
| Aborted | `aborted` | `Aborted()` |

A provider death never corrupts the session: partial turns are dropped, history
ends at the last stable item, unanswered tool calls are healed, and the session
accepts the next prompt immediately. Tile does not retry; request-level retries
belong to the `AsyncOpenAI` client you construct (`max_retries`), and the
recovery unit above that is re-prompting the session.

Run events are replayable facts, and the run-level closure survives every
in-process termination: an exception or abort still lands exactly one
`RunEndEvent` as the log's final event before the terminal status lands.
`run.status` and `run.outcome` remain the authoritative terminal state, and
`RunEndEvent.outcome` always matches them.

## Public API

Use the package facades for application code. Deep module paths are internal
and may move as Tile grows.

```python
from tile import (
    Aborted,
    AgentFailure,
    AgentRuntime,
    Completed,
    ExecutionFailure,
    Failed,
    InMemoryHistoryStore,
    InMemoryRunStore,
    Run,
    RunRecord,
    SQLiteHistoryStore,
    SQLiteRunStore,
)
from tile.events import AgentEvent, MessageEndEvent, RunEndEvent, StreamFn
from tile.providers.openai import create_stream_api
from tile.tools import BUILTIN_TOOLS
from tile.types import ToolDefinition, ToolError, ToolInput, ToolResult
from tile.types import ToolInputValidationFailure, ToolInvocationFailure
```

`tile` exposes the runtime, session, run handle, outcome, history-store,
run-store, and runtime-error contracts. `tile.events` exposes the structured
events yielded by `Run.events()`. `tile.types` exposes provider-neutral
conversation, stream, and tool contracts, including structured validation and
invocation failures on tool-execution event details. `tile.providers.openai`
exposes
`create_stream_api`, which
binds a caller-constructed `AsyncOpenAI` client and optional provider reasoning
options to the runtime's stream-function contract:
`create_stream_api(AsyncOpenAI(...), reasoning={"effort": "medium"})`. A stream
function declares its provider identity via a `provider` attribute on the
callable, stated once where the callable is constructed.

## Architecture

```
tile/
├── history/         # Session metadata and conversation history stores
├── providers/       # Provider integrations (OpenAI today)
├── runs/            # Durable run-summary contracts and stores
├── tools/           # Built-in local tool implementations
├── types/           # Provider-neutral contracts for conversations and tools
├── agent.py         # Stateless agent loop: provider turns and tool batches
├── events.py        # Runtime event contracts and run lifecycle rules
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
