# OpenAI Stream Event Lifecycle

This document maps raw OpenAI stream events from both supported transports to the final agent-facing events. The executable source of truth is still the test suite:

- `tests/test_openai_provider.py` covers raw SDK and subscription payloads through provider stream events.
- `tests/test_openai_stream_assembler.py` covers normalized events through app stream events.
- `tests/test_agent.py` covers app stream events through agent events.

The diagrams below use actor-style Mermaid sequence diagrams. Columns are stages in the pipeline, and time flows from top to bottom.

## Actors

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Sub as Subscription raw event
    participant SDKA as normalize_sdk_events
    participant SubA as normalize_subscription_events
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent
    participant Hist as Local run history

    SDK->>SDKA: OpenAI Python SDK event object
    Sub->>SubA: ChatGPT subscription SSE payload
    SDKA->>Norm: transport-independent event
    SubA->>Norm: transport-independent event
    Norm->>Asm: consumed in order
    Asm->>Stream: app-level stream event
    Stream->>Agent: consumed in order
    Agent->>Hist: finalized assistant/tool-result turns
```

## Stream Start And Created

`assemble_stream` emits a start event before it consumes provider events. The later `CREATED` event mutates the same shared assistant message with the provider response id.

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Sub as Subscription raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent

    Asm->>Stream: StreamStartEvent(message=empty AssistantMessage)
    Stream->>Agent: start
    Agent-->>Agent: emit TurnStartEvent
    Agent-->>Agent: emit MessageStartEvent(message)

    SDK->>Adapter: ResponseCreatedEvent
    Sub->>Adapter: response.created
    Adapter->>Norm: CREATED(response_id)
    Norm->>Asm: CREATED
    Asm-->>Asm: message.response_id = response_id
    Note over Asm,Agent: No new StreamEvent is emitted. The shared message is mutated.
```

## Reasoning Item

Reasoning summary deltas and reasoning text deltas both normalize to `REASONING_DELTA`. A reasoning summary part completion becomes a paragraph-separator reasoning delta.

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Sub as Subscription raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent

    SDK->>Adapter: ResponseOutputItemAddedEvent(item=reasoning)
    Sub->>Adapter: response.output_item.added(item.type=reasoning)
    Adapter->>Norm: REASONING_ADDED(item_id)
    Norm->>Asm: REASONING_ADDED
    Asm-->>Asm: create ReasoningBlock and set active_block=ReasoningBlock
    Asm->>Stream: ReasoningStartEvent(message)
    Stream->>Agent: reasoning_start
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseReasoningSummaryTextDeltaEvent
    SDK->>Adapter: ResponseReasoningTextDeltaEvent
    Sub->>Adapter: response.reasoning_summary_text.delta
    Sub->>Adapter: response.reasoning_text.delta
    Adapter->>Norm: REASONING_DELTA(delta)
    Norm->>Asm: REASONING_DELTA
    Asm-->>Asm: append delta to ReasoningBlock.summary_text
    Asm->>Stream: ReasoningDeltaEvent(delta, message)
    Stream->>Agent: reasoning_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseReasoningSummaryPartDoneEvent
    Sub->>Adapter: response.reasoning_summary_part.done
    Adapter->>Norm: REASONING_DELTA(delta="\\n\\n")
    Norm->>Asm: REASONING_DELTA
    Asm-->>Asm: append paragraph separator
    Asm->>Stream: ReasoningDeltaEvent(delta, message)
    Stream->>Agent: reasoning_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseOutputItemDoneEvent(item=reasoning)
    Sub->>Adapter: response.output_item.done(item.type=reasoning)
    Adapter->>Norm: REASONING_DONE(summary_text, reasoning_signature)
    Norm->>Asm: REASONING_DONE
    Asm-->>Asm: finalize ReasoningBlock and set active_block=None
    Asm->>Stream: ReasoningEndEvent(message)
    Stream->>Agent: reasoning_end
    Agent-->>Agent: emit MessageUpdateEvent
```

## Text And Refusal Item

`MESSAGE_TEXT_PART` selects which text channel is active. Output-text deltas are ignored while refusal is active, refusal deltas are ignored while output text is active, and unsupported parts set the active part to `None`.

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Sub as Subscription raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent

    SDK->>Adapter: ResponseOutputItemAddedEvent(item=message)
    Sub->>Adapter: response.output_item.added(item.type=message)
    Adapter->>Norm: MESSAGE_ADDED(item_id, phase)
    Norm->>Asm: MESSAGE_ADDED
    Asm-->>Asm: create TextBlock and set active_text_part_type=None
    Asm->>Stream: TextStartEvent(message)
    Stream->>Agent: text_start
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseContentPartAddedEvent
    Sub->>Adapter: response.content_part.added
    Adapter->>Norm: MESSAGE_TEXT_PART(output_text | refusal | None)
    Norm->>Asm: MESSAGE_TEXT_PART
    Asm-->>Asm: set active_text_part_type
    Note over Asm: No StreamEvent is emitted for content-part activation.

    SDK->>Adapter: ResponseTextDeltaEvent
    Sub->>Adapter: response.output_text.delta
    Adapter->>Norm: MESSAGE_TEXT_DELTA(part_type=output_text)
    Norm->>Asm: MESSAGE_TEXT_DELTA
    Asm-->>Asm: append only if active_text_part_type=output_text
    Asm->>Stream: TextDeltaEvent(delta, message)
    Stream->>Agent: text_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseRefusalDeltaEvent
    Sub->>Adapter: response.refusal.delta
    Adapter->>Norm: MESSAGE_TEXT_DELTA(part_type=refusal)
    Norm->>Asm: MESSAGE_TEXT_DELTA
    Asm-->>Asm: append only if active_text_part_type=refusal
    Asm->>Stream: TextDeltaEvent(delta, message)
    Stream->>Agent: text_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseOutputItemDoneEvent(item=message)
    Sub->>Adapter: response.output_item.done(item.type=message)
    Adapter->>Norm: MESSAGE_DONE(text, phase)
    Norm->>Asm: MESSAGE_DONE
    Asm-->>Asm: finalize TextBlock and clear active_block and active_text_part_type
    Asm->>Stream: TextEndEvent(message)
    Stream->>Agent: text_end
    Agent-->>Agent: emit MessageUpdateEvent
```

## Tool Call Item

Argument deltas are emitted for streaming UI updates. Parsed arguments are stored on the tool-call block when the arguments-done or item-done events arrive.

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Sub as Subscription raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent

    SDK->>Adapter: ResponseOutputItemAddedEvent(item=function_call)
    Sub->>Adapter: response.output_item.added(item.type=function_call)
    Adapter->>Norm: TOOL_CALL_ADDED(provider_item_id, call_id, name, arguments)
    Norm->>Asm: TOOL_CALL_ADDED
    Asm-->>Asm: create ToolCallBlock and set active_block=ToolCallBlock
    Asm->>Stream: ToolCallStartEvent(message)
    Stream->>Agent: tool_call_start
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseFunctionCallArgumentsDeltaEvent
    Sub->>Adapter: response.function_call_arguments.delta
    Adapter->>Norm: TOOL_CALL_ARGUMENTS_DELTA(delta)
    Norm->>Asm: TOOL_CALL_ARGUMENTS_DELTA
    Asm->>Stream: ToolCallDeltaEvent(delta, message)
    Stream->>Agent: tool_call_delta
    Agent-->>Agent: emit MessageUpdateEvent

    SDK->>Adapter: ResponseFunctionCallArgumentsDoneEvent
    Sub->>Adapter: response.function_call_arguments.done
    Adapter->>Norm: TOOL_CALL_ARGUMENTS_DONE(arguments)
    Norm->>Asm: TOOL_CALL_ARGUMENTS_DONE
    Asm-->>Asm: replace ToolCallBlock.arguments
    Note over Asm: No StreamEvent is emitted for parsed-arguments replacement.

    SDK->>Adapter: ResponseOutputItemDoneEvent(item=function_call)
    Sub->>Adapter: response.output_item.done(item.type=function_call)
    Adapter->>Norm: TOOL_CALL_DONE(final tool call data)
    Norm->>Asm: TOOL_CALL_DONE
    Asm-->>Asm: finalize ToolCallBlock and set active_block=None
    Asm->>Stream: ToolCallEndEvent(message)
    Stream->>Agent: tool_call_end
    Agent-->>Agent: emit MessageUpdateEvent
```

## Completed Turn Without Tools

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Sub as Subscription raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent
    participant Hist as Local run history

    SDK->>Adapter: ResponseCompletedEvent
    Sub->>Adapter: response.completed or response.done(status=completed)
    Adapter->>Norm: COMPLETED(stop_reason=stop)
    Norm->>Asm: COMPLETED
    Asm-->>Asm: message.stop_reason=stop
    Asm->>Stream: StreamDoneEvent(message)
    Stream->>Agent: done
    Agent-->>Agent: build AssistantTurn(status=completed)
    Agent->>Hist: append AssistantTurn to run-local history
    Agent-->>Agent: emit MessageEndEvent
    Agent-->>Agent: emit TurnEndEvent(tool_results=[])
    Agent-->>Agent: emit AgentEndEvent(new_items=[...])
```

## Completed Turn With Tools

```mermaid
sequenceDiagram
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent
    participant Tool as Tool execution
    participant Hist as Local run history

    Adapter->>Norm: COMPLETED(stop_reason=tool_use)
    Norm->>Asm: COMPLETED
    Asm->>Stream: StreamDoneEvent(message with ToolCallBlock)
    Stream->>Agent: done
    Agent->>Hist: append AssistantTurn to run-local history
    Agent-->>Agent: emit MessageEndEvent
    Agent->>Tool: execute tool call
    Agent-->>Agent: emit ToolExecutionStartEvent
    Tool-->>Agent: result
    Agent-->>Agent: emit ToolExecutionEndEvent
    Agent->>Hist: append ToolResultTurn to run-local history
    Agent-->>Agent: emit TurnEndEvent(tool_results=[...])
    Agent-->>Agent: request follow-up stream
    Note over Agent,Stream: The follow-up stream starts a new turn and repeats the same lifecycle.
```

## Incomplete And Failed Turns

```mermaid
sequenceDiagram
    participant SDK as SDK raw event
    participant Sub as Subscription raw event
    participant Adapter as Adapter
    participant Norm as NormalizedEvent
    participant Asm as assemble_stream
    participant Stream as StreamEvent
    participant Agent as run_agent
    participant Hist as Local run history

    SDK->>Adapter: ResponseIncompleteEvent(max_output_tokens)
    Sub->>Adapter: response.incomplete or response.done(status=incomplete)
    Adapter->>Norm: INCOMPLETE(stop_reason=length)
    Norm->>Asm: INCOMPLETE(length)
    Asm->>Stream: StreamDoneEvent(message)
    Stream->>Agent: done
    Agent->>Hist: append AssistantTurn to run-local history
    Agent-->>Agent: emit MessageEndEvent
    Agent-->>Agent: emit TurnEndEvent

    SDK->>Adapter: ResponseIncompleteEvent(content_filter)
    Sub->>Adapter: response.incomplete or response.done(status=incomplete)
    Adapter->>Norm: INCOMPLETE(stop_reason=error)
    Norm->>Asm: INCOMPLETE(error)
    Asm->>Stream: StreamErrorEvent(error=message)
    Stream->>Agent: error
    Agent->>Hist: append AssistantTurn to run-local history
    Agent-->>Agent: emit MessageEndEvent
    Agent-->>Agent: emit TurnEndEvent(tool_results=[])

    SDK->>Adapter: ResponseFailedEvent or ResponseErrorEvent
    Sub->>Adapter: response.failed or error
    Adapter->>Norm: FAILED(message)
    Norm->>Asm: FAILED
    Asm->>Stream: StreamErrorEvent(error=message)
    Stream->>Agent: error
    Agent->>Hist: append AssistantTurn to run-local history
    Agent-->>Agent: emit MessageEndEvent
    Agent-->>Agent: emit TurnEndEvent(tool_results=[])
```

## Raw Event Mapping

| Raw SDK event | Raw subscription event | Normalized event | Stream assembler effect | Agent effect |
| --- | --- | --- | --- | --- |
| `ResponseCreatedEvent` | `response.created` | `CREATED` | Mutates `message.response_id`; no new stream event | Shared message already referenced by `MessageStartEvent` |
| `ResponseOutputItemAddedEvent` with reasoning item | `response.output_item.added` with `item.type=reasoning` | `REASONING_ADDED` | `ReasoningStartEvent` | `MessageUpdateEvent` |
| `ResponseReasoningSummaryTextDeltaEvent` | `response.reasoning_summary_text.delta` | `REASONING_DELTA` | `ReasoningDeltaEvent` | `MessageUpdateEvent` |
| `ResponseReasoningTextDeltaEvent` | `response.reasoning_text.delta` | `REASONING_DELTA` | `ReasoningDeltaEvent` | `MessageUpdateEvent` |
| `ResponseReasoningSummaryPartDoneEvent` | `response.reasoning_summary_part.done` | `REASONING_DELTA` with paragraph separator | `ReasoningDeltaEvent` | `MessageUpdateEvent` |
| `ResponseOutputItemDoneEvent` with reasoning item | `response.output_item.done` with `item.type=reasoning` | `REASONING_DONE` | `ReasoningEndEvent` | `MessageUpdateEvent` |
| `ResponseOutputItemAddedEvent` with message item | `response.output_item.added` with `item.type=message` | `MESSAGE_ADDED` | `TextStartEvent` | `MessageUpdateEvent` |
| `ResponseContentPartAddedEvent` | `response.content_part.added` | `MESSAGE_TEXT_PART` | Sets active text part; no new stream event | No direct event |
| `ResponseTextDeltaEvent` | `response.output_text.delta` | `MESSAGE_TEXT_DELTA(output_text)` | `TextDeltaEvent` if output text is active | `MessageUpdateEvent` |
| `ResponseRefusalDeltaEvent` | `response.refusal.delta` | `MESSAGE_TEXT_DELTA(refusal)` | `TextDeltaEvent` if refusal is active | `MessageUpdateEvent` |
| `ResponseOutputItemDoneEvent` with message item | `response.output_item.done` with `item.type=message` | `MESSAGE_DONE` | `TextEndEvent` | `MessageUpdateEvent` |
| `ResponseOutputItemAddedEvent` with function-call item | `response.output_item.added` with `item.type=function_call` | `TOOL_CALL_ADDED` | `ToolCallStartEvent` | `MessageUpdateEvent` |
| `ResponseFunctionCallArgumentsDeltaEvent` | `response.function_call_arguments.delta` | `TOOL_CALL_ARGUMENTS_DELTA` | `ToolCallDeltaEvent` | `MessageUpdateEvent` |
| `ResponseFunctionCallArgumentsDoneEvent` | `response.function_call_arguments.done` | `TOOL_CALL_ARGUMENTS_DONE` | Replaces parsed arguments; no new stream event | No direct event |
| `ResponseOutputItemDoneEvent` with function-call item | `response.output_item.done` with `item.type=function_call` | `TOOL_CALL_DONE` | `ToolCallEndEvent` | `MessageUpdateEvent` |
| `ResponseCompletedEvent` | `response.completed` or completed `response.done` | `COMPLETED` | `StreamDoneEvent` | `MessageEndEvent`, `TurnEndEvent`, optional tool execution |
| `ResponseIncompleteEvent` with length stop | `response.incomplete` or incomplete `response.done` | `INCOMPLETE(length)` | `StreamDoneEvent` | `MessageEndEvent`, `TurnEndEvent` |
| `ResponseIncompleteEvent` with content-filter stop | `response.incomplete` or incomplete `response.done` | `INCOMPLETE(error)` | `StreamErrorEvent` | `MessageEndEvent`, `TurnEndEvent` with error assistant turn |
| `ResponseFailedEvent` or `ResponseErrorEvent` | `response.failed` or `error` | `FAILED` | `StreamErrorEvent` | `MessageEndEvent`, `TurnEndEvent` with error assistant turn |
