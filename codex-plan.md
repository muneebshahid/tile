# TIL-12 Implementation Plan: Wide-Event Run Telemetry

## Current objective

Implement [TIL-12](https://linear.app/tileagent/issue/TIL-12/add-wide-event-run-telemetry):
accumulate high-cardinality context during a run and deliver exactly one
plain, serializable telemetry record to every caller-provided sink when the run
finishes.

The record is Tile's canonical wide event: callers may turn it into a JSON log,
an OpenTelemetry trace, a Honeycomb event, a Sentry error, or another
application-specific representation. Tile core owns the record and lifecycle
semantics, but does not depend on a logging, telemetry, or vendor SDK.

## Repository constraints

All implementation work must follow `AGENTS.md`:

- Preserve DDD, SOLID, modularity, and clean boundaries.
- Keep high-level functions and classes above lower-level details.
- Keep every function under 50 lines.
- Add module, class, function, and relevant variable documentation.
- Avoid `Any` and `object` type hints where a specific type is possible.
- Use `uv` for dependency changes. This ticket should require no new dependency.
- After each implementation unit, run the targeted tests and the full required
  gates:

  ```bash
  make format
  make type_check
  make test
  ```

- Check whether `docs/openai-stream-event-lifecycle.md` diagrams remain accurate.

## Agreed design

### One record, one sink protocol, zero or more configured sinks

Expose one sink protocol:

```python
class RunTelemetrySink(Protocol):
    """Receives one completed run telemetry emission."""

    def emit(
        self,
        record: RunTelemetryRecord,
        *,
        exceptions: Sequence[CapturedRunException],
    ) -> None: ...
```

`AgentRuntime` receives a sequence that defaults to empty:

```python
telemetry_sinks: Sequence[RunTelemetrySink] = ()
```

The runtime copies it to a tuple during construction. At run completion it
calls each sink once, sequentially, in caller-provided order. A normal
`Exception` from one sink is captured and does not prevent later sinks from
running or alter the run outcome. Process-control `BaseException` subclasses
must retain the repository's existing propagation semantics. An empty tuple
means delivery is disabled; it is not an ambient or no-op sink.

Do not add:

- `CompositeSink`; the runtime loop already supplies fan-out.
- A separate `RunErrorReporter`; sinks receive original exception sidecars.
- Ambient/default sinks.
- Core JSON, stdout, file, OTel, Honeycomb, Sentry, or sampling implementations.

Tests may define a private collecting sink. Application adapters can be
documented or implemented in later tickets without changing the core protocol.

### Event recording, final fold, and sinks are separate responsibilities

The existing run event list is the accumulated source of truth. During
execution, a small run-owned lifecycle tracker stamps lifecycle events with
stable scope identity, parent identity, and monotonic time as they are
published. It does not maintain duplicate token, turn, or tool aggregates.

After execution and all pre-delivery finalization steps, the pure
`build_run_telemetry` function folds the completed event list, finished
`RunRecord`, finalization errors, and optional context receipt into one frozen
`RunTelemetryRecord`. It performs no I/O, does not read a clock, does not
generate IDs, and returns the same output for the same inputs.

The sinks are only delivery boundaries. They do not own runtime accumulation
or record construction.

### Serializable record plus original-exception sidecars

`RunTelemetryRecord` must remain a frozen Pydantic model that round-trips
through JSON. Its structured error details contain no live exception objects.

`CapturedRunException` is a non-serializable sidecar containing:

- Whether the exception is the primary run failure or a secondary failure.
- The stage at which it occurred.
- The original `BaseException`, preserving traceback and exception chaining.

This lets JSON/OTel sinks rely on the record while a Sentry-style sink can
capture the original traceback through the same `emit` call.

### Canonical record shape

Define focused, provider-neutral models rather than one loose dictionary:

```text
RunTelemetryRecord
├── identity
│   ├── run_id
│   ├── session_id
│   ├── provider
│   └── model
├── terminal state
│   ├── status
│   └── outcome
├── time
│   ├── started_at
│   ├── started_monotonic_ns
│   ├── ended_monotonic_ns
│   └── duration_ns
├── aggregates
│   ├── turn_count
│   ├── token_usage
│   └── tools
├── lifecycle scopes
├── structured errors
└── context_receipt
```

Use integer nanoseconds as the authoritative monotonic representation. A UTC
start anchor plus monotonic offsets allows an external adapter to construct
valid epoch timestamps without trusting wall-clock duration.

The optional context field is an opaque future reference:

```python
context_receipt: str | None = None
```

TIL-12 does not produce a receipt because Tile has no inspectable context store
yet. The field is reserved so a future context-capture boundary can return
either:

- A primary key/reference to a record in a dedicated context store.
- A content-addressed digest of the exact assembled model context.

It is not the primary key of the current prompt in `HistoryStore`.
`HistoryStore` stores replayable conversation history, while an inspectable
context receipt would identify the complete model-visible input for a specific
provider call or run, including instructions, selected history, and tools.
Until that feature exists, the value remains `None`. The telemetry record must
never embed full prompts or conversation content.

### Token usage

Add a provider-neutral `TokenUsage` model with non-negative counters:

- `input_tokens`
- `output_tokens`
- `total_tokens`
- `cached_input_tokens`
- `reasoning_output_tokens`

Missing provider usage contributes zero and is distinguishable if needed by an
optional response count or presence flag; do not invent token estimates.

Carry usage through the existing OpenAI pipeline:

```text
OpenAI terminal SDK event
→ normalized terminal event
→ StreamDoneEvent / StreamErrorEvent
→ MessageEndEvent
→ run event list
→ build_run_telemetry
```

Map `response.usage` on completed, incomplete, and failed response objects when
present. Transport errors without a response legitimately have no usage.

Aggregate usage across every provider response in a run, including tool loops
and typed-result follow-up attempts.

Token usage is also attached to the lifecycle scopes that own provider work:

- A message scope carries the usage reported for that provider response.
- A turn scope aggregates its message usage.
- An agent-attempt scope aggregates its turn usage.
- The root run scope and `RunTelemetryRecord.token_usage` aggregate the entire
  run.
- Tool-execution scopes carry no token usage. A tool does not consume model
  tokens; the surrounding message/turn does. Copying one response's usage onto
  every tool call would double-count when a response requests multiple tools.

These repeated scope totals are descriptive attributes for trace export.
Consumers must not sum usage across different hierarchy levels.

### Turn semantics

`turn_count` counts published `TurnStartEvent` instances. A provider failure
before `StreamStartEvent` therefore contributes zero turns under the current
lifecycle contract. Do not restructure `agent.py` or redefine turn boundaries
inside TIL-12; TIL-44 owns that future control-flow work.

### Tool aggregates

Aggregate by stable tool name. Each `ToolAggregate` should contain:

- `tool_name`
- `call_count`
- `completed_count`
- `error_count`
- `total_duration_ns`

Count a call at `ToolExecutionStartEvent`. A normal end is classified from
`ToolExecutionOutcome.tool_result_turn.is_error`. If a tool start has no
matching end when `RunEndEvent` arrives, its scope is interrupted. If callers
need an interrupted count, derive it as `call_count - completed_count`.

Do not store raw tool arguments, model-visible result content, or
`ToolResult.details` in the telemetry record. Those values remain available in
the event stream for callers that explicitly choose to inspect them. The
telemetry record keeps only bounded tool identity, timing, and outcome
aggregates.

### Structured errors

Preserve errors in observation order and never replace the primary run failure.
Each serializable error contains:

- `role`: `primary` or `secondary`
- `stage`
- Stable error category
- Exception type when applicable
- Message

The outcome remains the authoritative task verdict. Agent-declared failure is
represented without fabricating an exception. Original in-process exceptions
are additionally passed as `CapturedRunException` sidecars.

Stages should cover at least:

- `submission`
- `turn`
- `execution`
- `run_persistence`
- `history_healing`
- `owner_release`

Sink delivery failures occur after the emitted record has been frozen and
cannot be inserted into that same record. Expose them separately on the live
run handle, for example as `run.telemetry_errors`, containing sink identity and
the original exception. They must never cause another emission attempt.

### Lifecycle scope records and TIL-44 compatibility

TIL-41 currently guarantees the outer run boundary. Inner scopes can remain
open and are swept by `RunEndEvent`. TIL-44 deliberately defers precise
producer-owned interruption events until the next forced `agent.py` rework,
likely TIL-24 or TIL-32.

TIL-12 must not reopen TIL-41, block on TIL-44, or restructure `agent.py`.
Instead, the lifecycle tracker and final fold support both shapes:

```text
Current:
MessageStart → RunEnd

After TIL-44:
MessageStart → MessageInterrupted → TurnInterrupted
             → AgentInterrupted → RunEnd
```

Each `LifecycleScopeRecord` contains:

- `scope_id`
- `parent_scope_id`
- `scope_type`: run, agent, turn, message, or tool execution
- Monotonic start and end timestamps
- Closure classification:
  - `completed`
  - `interrupted`
- Optional stable operation identity such as tool name or response ID, never
  prompt/tool payload content.
- Optional provider token usage for run, agent, turn, and message scopes; tool
  scopes carry none.

The run-side lifecycle tracker tracks open scopes. Normal end events close
matching scopes. Future TIL-44 interruption events close them precisely.
`RunEndEvent` closes anything still open at the run-end timestamp as
`interrupted`, innermost first, without emitting duplicate lifecycle events.

This produces a structurally valid span tree today. Interrupted child end
timestamps become more precise automatically after TIL-44; the telemetry schema
does not distinguish whether interruption closure came from the current
run-end rule or a future TIL-44 event.

Runtime-exposed lifecycle events must also gain stable scope identity,
parent-scope identity, and monotonic timestamps. Stamp this metadata centrally
at the run publication boundary so producers remain focused on execution.
Provider content fragments remain message content rather than lifecycle
scopes.

### Finalization order

Refactor `Run` finalization into named, narrow steps that read like a table of
contents:

```text
1. Derive the primary task outcome.
2. Commit RunEndEvent and close any remaining scopes as interrupted.
3. Finish the live durable RunRecord.
4. Attempt terminal run persistence; capture secondary failure.
5. Attempt unanswered-tool healing; capture secondary failure.
6. Attempt owner/session release; capture secondary failure.
7. Build and freeze RunTelemetryRecord and exception sidecars through the pure
   final fold.
8. Invoke every configured sink once; capture delivery failures per sink.
9. Set the finalized signal unconditionally.
10. Re-raise any process-control signal according to existing semantics.
```

The run duration ends at `RunEndEvent`; persistence, healing, release, and sink
latency are finalization work and must not inflate execution duration.

For a submission failure after the stable run record exists, finalize and emit
the failed telemetry record before re-raising to the caller. If creating the
initial run record itself fails, the prompt was never accepted as a run and no
telemetry record is required.

For every finalized run, Tile makes at most one in-process delivery attempt to
each configured sink. An empty sink sequence makes no attempts. A successful
`emit` call does not promise that a remote backend received the record: an
adapter may enqueue it and the process may die before flushing, or a network
retry may duplicate it. Vendor adapters own buffering, retries, flushing, and
remote delivery guarantees; `run_id` is the stable deduplication key.

If a synchronous `emit` call raises a normal `Exception`, record it in
`run.telemetry_errors`, report it through the existing internal bookkeeping
error channel, and continue to later sinks. Do not rebuild the record, retry
the sink, or change the run outcome. This is reporting of a telemetry-system
failure, not the per-run wide-event logging product prohibited by the ticket's
"no debug logger" boundary.

## Proposed modules and ownership

Prefer these boundaries, adjusting filenames only if the implementation reveals
a simpler equally modular shape:

```text
tile/telemetry/
├── __init__.py       public telemetry exports
├── models.py         frozen serializable models
└── sink.py           sink and exception-sidecar protocols/contracts

tile/runtime/
└── telemetry.py      private lifecycle tracker and pure final record builder
```

Expected existing files to change:

- `tile/types/stream_events.py`
- `tile/providers/openai/normalized_events.py`
- `tile/providers/openai/sdk_event_adapter.py`
- `tile/providers/openai/stream_assembler.py`
- `tile/events.py`
- `tile/agent.py` only to propagate usage on existing events, not to restructure
- `tile/runtime/runtime.py`
- `tile/runtime/run.py`
- `tile/__init__.py`
- `tile/types/__init__.py`
- `README.md`
- `docs/openai-stream-event-lifecycle.md`

Keep provider SDK types out of telemetry and runtime modules.

## Implementation units

Implement one unit at a time. For every unit: add the stub, add the focused
test, implement only enough to pass it, run validation, perform an `AGENTS.md`
compliance pass, and ask whether the unit should be committed before continuing.

### Unit 1: Public telemetry contracts

1. Stub the telemetry models, sink protocol, captured-exception sidecar, and
   public exports.
2. Add tests for:
   - Frozen models.
   - JSON round-trip.
   - Non-negative token validation.
   - Outcome/error union serialization.
   - No exception objects appearing in serialized output.
3. Implement validators and concise documentation.

### Unit 2: Provider token usage propagation

1. Add usage fields to normalized and provider-terminal event contracts.
2. Add raw OpenAI response fixtures containing input, output, cached, reasoning,
   and total token counts.
3. Test completed, incomplete, failed, and absent-usage paths at adapter and
   assembler boundaries.
4. Propagate usage to `MessageEndEvent` without persisting it into conversation
   history.

### Unit 3: Lifecycle metadata and scope accumulation

1. Stub lifecycle metadata on runtime events.
2. Implement the private scope tracker with an injected/test clock and stable-ID
   factory.
3. Test:
   - Successful nested scope tree.
   - Typed-result sequential agent attempts.
   - Multiple tool siblings.
   - Provider failure before and after message start.
   - Abort during message and tool execution.
   - Run-end interruption closure without duplication.
   - Future interruption-event compatibility through representative test
     events, without implementing TIL-44.

### Unit 4: Pure final record fold

1. Add `build_run_telemetry` as a pure fold over completed events, the finished
   run record, finalization errors, and the optional context receipt.
2. Test:
   - Token sums across multiple provider calls.
   - Message, turn, agent-attempt, and root token usage without tool-scope
     attribution.
   - Turn count semantics.
   - Tool success and handled-error aggregates, with interruption derivable
     from calls minus completions.
   - Monotonic duration despite wall-clock movement.
   - Primary and secondary error ordering.
   - Context receipt omission and serialization.

### Unit 5: Runtime injection and finalization

1. Add `telemetry_sinks: Sequence[RunTelemetrySink] = ()` to `AgentRuntime`.
2. Thread the tuple through `_RunDependencies`.
3. Refactor finalization into the explicit ordered stages above.
4. Add `run.telemetry_errors`.
5. Add end-to-end tests using collecting/failing sinks for:
   - Successful prompt.
   - Provider/runtime failure.
   - Abort before first tick.
   - Abort mid-stream and mid-tool.
   - Submission failure after run-record creation.
   - Typed-result retry and tool loop.
   - Terminal run-store failure.
   - History healing and owner-release failures.
   - Multiple sinks called once in order.
   - One failing sink not blocking later sinks.
   - Sink failure not changing status/outcome or triggering retry.

### Unit 6: Documentation and public surface

1. Update runtime construction examples to pass sinks.
2. Document that the record is the canonical structured log event.
3. Show short application-owned examples for:
   - A collecting test sink.
   - A one-line JSON sink.
   - The mapping an OTel adapter performs.
4. Explain that current run-end closure and future TIL-44 interruption events
   share the same `interrupted` telemetry status; TIL-44 only improves timing
   precision.
5. Update lifecycle diagrams if metadata or closure representation changes.
6. Verify public import tests.

## End-to-end acceptance matrix

| Path | Records per sink | Status | Primary error | Secondary errors | Open scopes |
| --- | ---: | --- | --- | --- | --- |
| Successful plain prompt | 1 | completed | none | preserved if any | none |
| Typed result / agent failure | 1 | completed | structured agent failure | preserved | none |
| Provider exception | 1 | failed | original execution failure | preserved | interrupted |
| Abort | 1 | aborted | none | preserved | interrupted |
| Submission failure after record creation | 1 | failed | submission failure | preserved | none |
| Terminal persistence failure | 1 | execution status unchanged | unchanged | persistence failure | none |
| Healing/release failure | 1 | execution status unchanged | unchanged | ordered failures | none |
| One sink fails | 1 attempted for every sink | unchanged | unchanged | unchanged in emitted record | none |

## Plan critique and safeguards

### Risks challenged

- **Too many extension points:** resolved by one sink protocol; no separate
  reporter or composite abstraction.
- **Vendor coupling:** original exceptions are sidecars, while the record stays
  provider/vendor neutral and serializable.
- **TIL-44 scope creep:** TIL-12 tolerates both current run-end closure and
  precise future interruption events but exposes both as interrupted and does
  not change agent control flow.
- **Misleading exactly-once claim:** explicitly limited to one in-process call
  per sink; remote guarantees remain adapter concerns.
- **Sink failure recursion:** sink failures are exposed on the run handle and
  never cause re-emission through the same sink set.
- **Telemetry changing execution:** accumulation performs no I/O, and sink or
  finalization failures never replace the primary outcome.
- **Sensitive payload leakage:** prompts, tool arguments, tool result content,
  tool details, and full context are excluded; only an opaque future context
  receipt is reserved.
- **Unbounded cardinality inside a record:** aggregate by tool name and store
  compact lifecycle scope summaries, not streaming deltas.

### Decisions that must not be silently changed

- The sink API is synchronous; production adapters should enqueue/batch.
- `AgentRuntime` accepts a sequence directly; do not add `CompositeSink`.
- The sink sequence may be empty; do not add a required or ambient sink.
- Do not add a separate Sentry/error-reporting hook.
- Do not add an ambient/no-op default sink.
- Do not put telemetry fields into durable conversation history.
- Do not merge `RunTelemetryRecord` into durable `RunRecord`.
- Do not implement TIL-44 as part of TIL-12.
- Do not add vendor SDK dependencies to Tile core.

If implementation evidence requires changing one of these decisions, stop,
document the evidence and trade-off, and discuss it before expanding scope.
