# Ori Plan: Python-Native Agent Session Runtime

This plan replaces the older pre-Ori roadmap. It reflects the current direction:
embrace the similarity to adjacent agent frameworks, keep the core small, and
use real downstream applications to force useful runtime and extension seams.

## Direction

Ori is a small Python-native runtime for tool-using agent sessions.

It should be usable in two shapes:

- embedded directly inside Python applications, scripts, tests, and workers
- exposed as a small service for non-Python clients and long-running sessions

The project is in the same broad category as Flue and PyFlue. The distinction is
not that Ori invents a different kind of framework. The distinction is Python
nativeness, compact scope, library/service duality, and eventually runtime hooks
that make it easy to customize the agent loop from Python.

Ori should avoid becoming:

- a feature-for-feature Flue clone
- a PyFlue parity project
- a full personal-agent product
- a terminal UI product
- a workflow/channel/deployment framework before the runtime is solid
- a broad provider matrix before one serious downstream app needs it

Ori should become:

> A compact Python-native agent session runtime that can be embedded or served.

Shorter:

> Ori gives Python applications a reusable agent session runtime.

## Product Thesis

Ori provides the runtime substrate for building concrete Python-native agents.
The product value comes from making real agents easy to build, customize, and
operate, not from pretending the framework category is unique.

The runtime should make it easy to:

- create and list sessions
- stream structured runtime events
- persist model-visible conversation history
- fork sessions
- execute typed Python tools
- reject or control overlapping prompts clearly
- expose the same runtime over a service boundary
- connect Python and non-Python clients
- add runtime hooks when real downstream apps need customization

The first serious downstream app, such as Hermes or an email agent, should be
used to validate the extension API and avoid speculative hook design.

## Current Baseline

The current codebase already has a useful local kernel:

- stateless `run_agent(...)` provider/tool loop
- provider stream events separated from agent runtime events
- `AgentRuntime`, `Session`, and `HistoryStore`
- generated and explicit session ids
- session history snapshots
- session forking by copying history
- fail-fast `SessionBusyError` for overlapping same-session prompts
- in-memory history store
- tool execution boundary
- built-in local coding tools
- OpenAI provider path
- local JSONL runner
- focused tests for agent, runtime, provider, serialization, and tools

This baseline should not be re-planned as future work. Future tickets should
build on it.

## Non-Goals For The Next Release

Defer these until the server/client/runtime core is proven:

- full append-only event replay
- durable mid-run recovery
- checkpoint-based resume
- auto-compaction
- provider switching mid-session
- provider breadth beyond the first practical need
- workflows, channels, schedules, and hosted deployment abstractions
- a plugin marketplace or package ecosystem
- terminal UI work

## Core Concepts

### Runtime

`AgentRuntime` is the application service for sessions. It owns provider
configuration, tool execution, history writes, and active prompt coordination.

The runtime must remain usable from normal Python code. Server mode should adapt
the same runtime instead of creating a second implementation.

### Session

A session is one conversation or task thread. A `Session` is a scoped handle over
runtime services, not the owner of infrastructure.

Useful session operations:

- prompt
- inspect history
- fork
- later: abort active run
- later: subscribe through a service endpoint

### Conversation History

Conversation history is model-visible, provider-neutral state:

- `UserMessage`
- `AssistantTurn`
- `ToolResultTurn`

It is not an operational log, not a debug logger, and not active runtime state.

### Run Record

A run record is product state for one prompt execution:

- `run_id`
- `session_id`
- status
- started/completed timestamps
- error details when failed

Run records give the server a concrete object to report, cancel, and mark failed
after process errors. They are the next persistence concept after conversation
history.

### Active Runtime State

Active runtime state is in-process execution state:

- active run task
- session busy marker
- cancellation flag
- subscribers
- current tool call metadata

This state should be explicit but not model-visible. The current
`SessionBusyError` set is the simplest version of this concept.

### Service Boundary

Server mode should expose the runtime without changing who executes work:

- clients submit prompts
- Ori executes provider calls and tools server-side
- clients stream events
- clients can inspect sessions and completed history

For the first server, SSE is enough. WebSockets and bidirectional steering can
come later if an app actually needs them.

### Extensions

Extensions are runtime-facing Python code, not just model-facing skills.

Do not overdesign the extension API before a downstream app needs it. The first
useful hook set is likely:

- `before_model_call`
- `build_context`
- `authorize_tool_call`
- `after_tool_result`
- `after_history_append`

Extensions should be introduced after SQLite/server/client basics, then refined
against a real app such as Hermes.

## Release Strategy

### Release 1: Stable Local Runtime

Goal: make the existing local runtime coherent, documented, and stable enough to
serve as the foundation for persistence and server mode.

Includes:

- current runtime/session/tool/provider baseline
- stable public imports for the current API surface
- clear error vocabulary for session busy and session lookup failures
- updated README and examples using Ori naming
- stale roadmap/todo cleanup

Does not include:

- module/package import reorganization under `ori`
- SQLite
- server mode
- extension hooks

### Release 2: Persistent Sessions And Run Records

Goal: make sessions survive process restart at completed-turn boundaries.

Includes:

- SQLite-backed session metadata and conversation history
- `RunRecord` model and store
- run ids attached to prompt execution
- failed/completed run status persistence
- continuation from completed history after restart

Does not include:

- durable mid-run recovery
- full event journal
- exact stream replay after reconnect

### Release 3: Service Mode And Python Client

Goal: expose the same runtime to clients without making clients execution hosts.

Includes:

- `ori serve`
- create/get/list/fork session endpoints
- submit prompt endpoint
- busy-session rejection
- SSE event stream for a running prompt
- completed history retrieval
- simple Python client
- minimal auth suitable for local/private deployment

Does not include:

- hosted deployment matrix
- WebSocket steering
- durable queue semantics
- multi-process workers

### Release 4: Runtime Hooks

Goal: add the smallest extension surface needed by a concrete downstream app.

Includes:

- deterministic hook ordering
- hook registration through Python code
- hook failures with explicit policy
- initial hooks for context building, model calls, tool authorization, tool
  results, and history append
- one or two proof extensions

The acceptance bar is not "can arbitrary plugins exist." The bar is "Hermes or a
similar app can customize Ori without forking the runtime."

### Release 5: Downstream App Validation

Goal: use a real agent product to decide what Ori actually needs next.

Candidate app:

- Hermes-style email triage
- spam filtering
- unsubscribe automation
- follow-up drafting
- mailbox search and summarization

Ori should only promote new abstractions from this work when the app proves they
are reusable runtime concerns.

## High-Level Tickets

Each ticket should be testable end to end.

### 1. Finish The Ori Rename

Complete project identity migration without breaking local Codex session
continuity.

Acceptance criteria:

- README, package metadata, prompt text, and user-facing examples use Ori.
- GitHub repository is renamed to `muneebshahid/ori`.
- Local path can remain `/Users/muneeb/work/piy` until old sessions no longer
  matter, or it can be renamed with a `/Users/muneeb/work/piy` symlink.
- Roadmap uses Ori terminology.

End-to-end test:

- Run the full validation suite and verify the local remote points at the renamed
  GitHub repository.

### 2. Stabilize Public Runtime Imports

Define the first stable import surface without doing a large module move.

Acceptance criteria:

- Public imports are documented from existing packages.
- Internal modules are clearly not guaranteed stable.
- Existing examples use the documented imports.
- No top-level `ori/` module move is required in this ticket.

End-to-end test:

- Run an example script that imports the documented runtime/session/tool/provider
  APIs and completes a fake-provider prompt.

### 3. Add SQLite History Store

Add durable completed conversation history behind the existing `HistoryStore`
contract.

Acceptance criteria:

- SQLite store persists sessions and conversation items by session id.
- In-memory and SQLite stores satisfy the same behavior tests.
- Runtime can continue a session after process restart using completed history.
- Stored items remain provider-neutral.

End-to-end test:

- Run a session, recreate the runtime with the same SQLite database, prompt
  again, and assert previous completed history is sent to the provider.

### 4. Add Run Records

Persist prompt execution metadata separately from model-visible history.

Acceptance criteria:

- Each prompt has a `run_id`.
- Run status transitions through running to completed or failed.
- Run records store timestamps and structured errors.
- Failed runs do not corrupt completed conversation history.

End-to-end test:

- Start a prompt that succeeds and one that fails; assert run records reflect both
  outcomes and history contains only stable conversation items.

### 5. Add Minimal Server Mode

Expose runtime sessions through a small HTTP service.

Acceptance criteria:

- `ori serve` starts a local server.
- Clients can create, get, list, and fork sessions.
- Clients can submit a prompt to a session.
- Same-session overlap returns a clear busy response.
- Tool execution remains server-side.

End-to-end test:

- Start the server with a fake provider, submit a prompt through HTTP, and assert
  session history is persisted.

### 6. Add SSE Prompt Streaming

Stream runtime events from server-side prompt execution.

Acceptance criteria:

- Prompt submission exposes or returns a stream endpoint.
- SSE events preserve the same typed runtime event payloads.
- Slow/disconnected clients do not block prompt completion.
- Completed history can be fetched separately.

End-to-end test:

- Submit a prompt, consume SSE events through completion, disconnect a second
  slow consumer, and assert the run still completes.

### 7. Add Python Client

Provide a small client for the server protocol.

Acceptance criteria:

- Client can create/list/get/fork sessions.
- Client can submit prompts and consume streamed events.
- Client raises structured errors for busy sessions and missing sessions.
- Client does not contain agent execution logic.

End-to-end test:

- Use the Python client against a local fake-provider server to run a prompt and
  inspect completed history.

### 8. Add First Runtime Hooks

Introduce hooks only after server/client basics are usable.

Acceptance criteria:

- Hooks are registered through Python code.
- Hook ordering is deterministic.
- Hook failures have explicit behavior.
- Initial hooks cover context building, model-call inspection, tool
  authorization, tool-result transformation, and history append observation.

End-to-end test:

- Register a tool-authorization hook that rejects a write command and assert the
  model receives a structured tool error while the run completes.

### 9. Validate With A Real Downstream App

Build or spike a Hermes-like email agent outside Ori.

Acceptance criteria:

- The app uses Ori through the public runtime or server/client surface.
- The app drives at least one real extension or hook requirement.
- Any new Ori abstractions are promoted only when they are reusable.

End-to-end test:

- Run the downstream app against a test mailbox fixture and assert it can classify
  messages, propose unsubscribe actions, and require approval before mutation.

## Explicit Deferrals

Do not schedule these until a downstream app proves the need:

- full event sourcing
- exact reconnect replay
- durable queues and multi-worker claiming
- checkpoint recovery during provider/tool execution
- automatic compaction
- provider switching mid-session
- subagents
- workflows
- channels and gateway integrations
- hosted deployment adapters
