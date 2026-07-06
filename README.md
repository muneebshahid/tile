# Tile

[![CI](https://github.com/muneebshahid/tile/actions/workflows/ci.yml/badge.svg)](https://github.com/muneebshahid/tile/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> ⚠️ **Work in Progress** — This project is under active development and APIs may change.

A small Python-native runtime for tool-using agent sessions.

## Overview

**Tile** is a headless agent session runtime for Python. Providers, tools, events, and serialization are explicit runtime contracts. Use it as an embedded library or as the core of a service without adopting a broad application framework.

## Features

- **Headless Runtime**: Run agents from Python code or a small local command without a UI dependency
- **Explicit Contracts**: Swap providers, tools, event handlers, and serializers without modifying core agent logic
- **Minimal Core**: Only what you need—everything else is optional
- **Async First**: Built on Python async/await for non-blocking I/O
- **Type Safe**: Full Pydantic integration with ty support
- **Streaming Support**: Real-time structured runtime events
- **Tool Execution**: Pluggable tool definitions and execution strategies
- **Reasoning**: Extensible support for extended thinking workflows

## Architecture

```
tile/
├── history/         # Session metadata and conversation history stores
├── providers/       # Provider integrations
│   └── openai/      # OpenAI provider implementation
├── tools/           # Built-in local tool implementations
├── types/           # Provider-neutral contracts for conversations and tools
├── events.py        # Runtime event contracts
└── runtime.py       # Session runtime facade
tests/               # Test suite
```

## Quick Start

Run a local prompt:

```bash
uv run python -m examples.local_runner "Inspect the current repository"
```

Or pipe a prompt through stdin:

```bash
printf "Inspect the current repository" | uv run python -m examples.local_runner
```

## Public API

Use the package facades for application code. Deep module paths are internal and
may move as Tile grows.

```python
from tile import AgentRuntime, HistoryStore, InMemoryHistoryStore, Run
from tile.events import AgentEvent, MessageEndEvent, StreamFn
from tile.providers.openai import create_stream_api
from tile.types import ToolDefinition, ToolResult
```

`tile` exposes the runtime, session, run-handle, history-store, and
runtime-error contracts. `tile.events` exposes structured runtime events
yielded by `Run.events()`. `tile.types` exposes provider-neutral conversation,
stream, and tool contracts. `tile.providers.openai` exposes
`create_stream_api`, which binds a caller-constructed `AsyncOpenAI` client and
optional provider reasoning options to the runtime's stream-function contract:
`create_stream_api(AsyncOpenAI(...), reasoning={"effort": "medium"})`.

Prompt execution is task-owned: `Session.prompt(...)` submits a run and
returns immediately, the runtime drives it to completion, and any number of
subscribers can observe it.

```python
run = await session.prompt("Inspect the current repository")
async for event in run.events():
    ...
status = await run.wait()  # "completed" | "failed" | "aborted"
```

## Security Posture

Tile's built-in tools are deliberately unconfined. `bash` executes arbitrary
shell commands with the permissions of the process running the agent, and the
file tools accept absolute paths — the session working directory is a default,
not a sandbox. Run Tile only where you would run the model's commands yourself,
and use OS-level isolation such as a container or VM when you need a boundary.
Resource exhaustion from trusted local input is out of scope for now. Tool
authorization hooks arrive with the runtime hooks release.

## Development

Install dependencies:
```bash
uv sync
```

Run tests:
```bash
make test
```

Format and type-check:
```bash
make format
make type_check
```
