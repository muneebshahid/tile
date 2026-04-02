# ChatGPT Subscription Support Implementation Plan

## Goal

Add ChatGPT subscription-based auth to `piy` without introducing a new top-level provider concept yet.

The implementation should:

- keep `openai` as the current provider family
- support two auth modes: API key and ChatGPT subscription
- refactor the current OpenAI provider so stream assembly is shared across transports
- load ChatGPT credentials from `CODEX_HOME/auth.json` or `~/.codex/auth.json`
- avoid `.env` token overrides

## Architectural Decision

We will separate:

1. request transport
2. auth source
3. shared stream assembly

We will **not** keep the shared assembler coupled to OpenAI SDK runtime event classes.

We will introduce a normalized OpenAI wire-event layer using `TypedDict` discriminated unions.

Why:

- current code in `ai/openai/provider.py` is tightly coupled to `openai` SDK event classes
- ChatGPT subscription transport will likely come from raw SSE JSON, not SDK event objects
- reconstructing SDK runtime event classes from subscription payloads is fragile and unnecessary
- `TypedDict` gives a lightweight shared contract without adding another heavy class hierarchy

## End State

After implementation:

- API mode uses `AsyncOpenAI`
- subscription mode uses a custom HTTP/SSE client backed by `httpx.AsyncClient`
- both paths normalize their source events into shared wire events
- one shared assembler converts wire events into app `StreamEvent`s
- auth mode selection happens once in the OpenAI entrypoint
- ChatGPT credentials load from the Codex auth file and can refresh when expired

## Current State

### Existing coupling

- `ai/openai/client.py` only creates an `AsyncOpenAI` client
- `ai/openai/provider.py` both sends requests and assembles stream events
- the assembler matches directly on OpenAI SDK event classes
- `settings.py` only supports API-key style OpenAI settings

### Constraints

- no `.env`-based subscription token fallback
- tests should use mocks instead of env token overrides
- follow refactor-first approach before adding subscription transport

## Implementation Phases

## Phase 1: Refactor Shared Stream Assembly

### Objective

Extract stream assembly into shared logic that does not depend on SDK runtime classes.

### Tasks

1. Add a new module for normalized wire events.
   - Suggested file: `ai/openai/wire_events.py`
   - Use `TypedDict` with a discriminant `type`
   - Define only the event shapes the assembler actually needs

2. Add a new shared assembler module.
   - Suggested file: `ai/openai/stream_assembler.py`
   - Move `StreamAssemblyState`
   - Move all `_start_*`, `_append_*`, `_finalize_*`, and completion/error handling functions
   - Change event dispatch from `match` on SDK classes to `match` or `if` on normalized `type`

3. Add an SDK normalization module.
   - Suggested file: `ai/openai/sdk_event_adapter.py`
   - Convert `openai` SDK event classes into normalized wire events
   - Keep SDK-specific parsing isolated here

4. Update the current provider entrypoint to use:
   - SDK stream -> SDK adapter -> shared assembler

### Deliverables

- no behavior change for API mode
- provider code no longer directly assembles from SDK event classes
- tests updated to target shared assembler behavior through normalized events where appropriate

## Phase 2: Introduce Auth Mode Configuration

### Objective

Introduce mode selection without yet implementing the full subscription transport.

### Tasks

1. Extend `settings.py`.
   - Add `openai_auth_mode: Literal["api", "chatgpt"] = "api"`
   - Add `chatgpt_base_url: str = "https://chatgpt.com/backend-api/codex"`
   - Add `codex_home: str | None = None`

2. Add an auth-mode resolver.
   - Suggested file: `ai/openai/auth_mode.py`
   - Resolve auth mode and base URL defaults

3. Refactor `ai/openai/client.py`.
   - Keep a public `create_client()` dispatcher
   - Add `create_client_api()`
   - Add a subscription creation path placeholder or interface, depending on sequencing

### Deliverables

- settings cleanly express mode and path information
- credentials are still not stored in settings

## Phase 3: Add ChatGPT Auth File Loading

### Objective

Load subscription credentials from the Codex auth file.

### Tasks

1. Add auth store module.
   - Suggested file: `ai/openai/auth_store.py`
   - Resolve auth file path from:
     - `CODEX_HOME/auth.json` if `CODEX_HOME` is set
     - otherwise `~/.codex/auth.json`

2. Define auth file models.
   - Suggested file: `ai/openai/auth_models.py`
   - Parse only the fields needed:
     - `auth_mode`
     - `tokens.access_token`
     - `tokens.refresh_token`
     - `tokens.account_id`
     - `tokens.id_token` if needed

3. Add subscription token utilities.
   - Suggested file: `ai/openai/subscription_tokens.py`
   - check expiry from JWT
   - derive account id from JWT if missing
   - refresh token through `https://auth.openai.com/oauth/token`
   - persist refreshed tokens back to auth file

### Deliverables

- API to obtain a valid access token and account id for subscription mode
- clear errors when auth file is missing or invalid

## Phase 4: Add Subscription Transport

### Objective

Implement ChatGPT subscription streaming over HTTP/SSE.

### Tasks

1. Add a subscription transport client.
   - Suggested file: `ai/openai/subscription_client.py`
   - Back it with `httpx.AsyncClient`
   - Build headers:
     - `Authorization`
     - `ChatGPT-Account-ID`
     - `OpenAI-Beta: responses=experimental`
     - `Accept: text/event-stream`

2. Build request payload.
   - Reuse current OpenAI serialization logic
   - Reuse current request parameter shape where compatible

3. Parse SSE stream.
   - Suggested file: `ai/openai/sse.py`
   - Parse `data:` frames
   - decode JSON payloads
   - ignore keepalive/no-op frames

4. Add subscription event adapter.
   - Suggested file: `ai/openai/subscription_event_adapter.py`
   - Normalize backend SSE JSON events into shared wire events
   - Handle event-name differences such as `response.done` vs `response.completed` if needed

### Deliverables

- working subscription-mode stream path
- transport-specific code isolated from shared assembly code

## Phase 5: Unify Public OpenAI Entry Point

### Objective

Expose one public `stream(...)` entrypoint that dispatches by auth mode.

### Tasks

1. Keep `ai/openai/provider.py` as the public facade.
2. Dispatch:
   - API auth -> API transport path
   - ChatGPT auth -> subscription transport path
3. Ensure both paths return the same `AsyncEventStream`

### Deliverables

- no call-site changes in `ui/app.py` or `agent/agent.py`

## Phase 6: Testing

### Objective

Cover the refactor and the new transport without relying on real credentials.

### Tests to add or update

1. Shared assembler tests
   - feed normalized wire events directly
   - verify text, reasoning, tool calls, and completion/error handling

2. SDK adapter tests
   - verify OpenAI SDK events normalize correctly

3. Auth store tests
   - valid auth file load
   - missing auth file
   - missing tokens
   - account id derivation

4. Token refresh tests
   - expired token refresh success
   - refresh failure surfaces useful error

5. Subscription SSE adapter tests
   - parse SSE frames
   - normalize subscription event variants
   - handle `response.done` compatibility mapping if needed

6. Provider dispatch tests
   - API mode routes to API path
   - ChatGPT mode routes to subscription path

### Test strategy

- use mocks/fakes for all network calls
- do not use env token overrides
- do not require real Codex auth state

## Phase 7: Documentation

### Objective

Document how the new mode works.

### Tasks

1. Update `README.md`
   - describe API mode
   - describe ChatGPT subscription mode
   - explain that subscription mode reads `~/.codex/auth.json`
   - explain that users should run `codex login`

2. Add `.env` example values for mode and optional `CODEX_HOME`

## Proposed File Layout

- `ai/openai/provider.py`
- `ai/openai/api_provider.py`
- `ai/openai/subscription_provider.py`
- `ai/openai/stream_assembler.py`
- `ai/openai/wire_events.py`
- `ai/openai/sdk_event_adapter.py`
- `ai/openai/subscription_event_adapter.py`
- `ai/openai/subscription_client.py`
- `ai/openai/auth_store.py`
- `ai/openai/auth_models.py`
- `ai/openai/subscription_tokens.py`
- `ai/openai/sse.py`

## Sequencing Recommendation

Recommended order:

1. shared assembler extraction
2. SDK event adapter
3. auth mode settings
4. auth file loading
5. subscription transport
6. provider dispatch
7. README updates

This order minimizes risk because the first phase is behavior-preserving and gives us a clean seam for the new transport.

## Risks

### Risk 1: Subscription SSE payload shape differs from SDK expectations

Mitigation:

- normalize into our own wire-event contract
- keep subscription adapter isolated

### Risk 2: Token refresh behavior is more subtle than expected

Mitigation:

- keep refresh logic in one module
- test expiry and persistence paths directly

### Risk 3: Shared assembler refactor changes behavior

Mitigation:

- do Phase 1 as a pure refactor
- preserve current test coverage and add direct assembler tests

## Definition of Done

The work is complete when:

- API mode continues to work unchanged
- ChatGPT mode works using the Codex auth file
- no subscription secrets are configured in `settings.py`
- shared stream assembly is transport-agnostic
- tests cover both transport paths using mocks
- README explains how to use both modes
