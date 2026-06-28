# Ori

> ⚠️ **Work in Progress** — This project is under active development and APIs may change.

A small Python-native runtime for tool-using agent sessions.

## Overview

**Ori** is a headless agent session runtime for Python. Providers, tools, events, and serialization are explicit runtime contracts. Use it as an embedded library or as the core of a service without adopting a broad application framework.

## Features

- **Headless Runtime**: Run agents from Python code or a small local command without a UI dependency
- **Explicit Contracts**: Swap providers, tools, event handlers, and serializers without modifying core agent logic
- **Minimal Core**: Only what you need—everything else is optional
- **Async First**: Built on Python async/await for non-blocking I/O
- **Type Safe**: Full Pydantic integration with mypy support
- **Streaming Support**: Real-time structured runtime events
- **Tool Execution**: Pluggable tool definitions and execution strategies
- **Reasoning**: Extensible support for extended thinking workflows

## Architecture

```
ori/
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
may move as Ori grows.

```python
from ori import AgentRuntime, HistoryStore, InMemoryHistoryStore
from ori.events import AgentEvent, MessageEndEvent, StreamFn
from ori.providers.openai import stream_api
from ori.types import ToolDefinition, ToolResult
```

`ori` exposes the runtime, session, history-store, and runtime-error contracts.
`ori.events` exposes structured runtime events yielded by `Session.prompt(...)`.
`ori.types` exposes provider-neutral conversation, stream, and tool contracts.
`ori.providers.openai` exposes the implemented OpenAI stream entrypoint.

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
