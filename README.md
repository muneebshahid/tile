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
├── agent/           # Core agent orchestration and event dispatch
├── ai/
│   ├── openai/      # OpenAI provider implementation
│   └── types/       # Shared type definitions for contracts, tools, and streams
└── tests/           # Test suite
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
