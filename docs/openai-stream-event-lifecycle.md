# OpenAI Stream Event Lifecycle

This document maps raw OpenAI stream events from both supported transports to the final agent-facing events. The executable source of truth is still the test suite:

- `tests/test_openai_provider.py` covers raw SDK and subscription payloads through provider stream events.
- `tests/test_openai_stream_assembler.py` covers normalized events through app stream events.
- `tests/test_agent.py` covers app stream events through agent events.

## Pipeline

```mermaid
flowchart TD
    sdk["OpenAI SDK raw event object"]
    subscription["ChatGPT subscription SSE payload"]
    sdk_adapter["normalize_sdk_events"]
    subscription_adapter["normalize_subscription_events"]
    normalized["NormalizedEvent"]
    assembler["assemble_stream"]
    stream["StreamEvent"]
    agent["Agent.run"]
    agent_event["AgentEvent"]
    history["conversation history"]

    sdk --> sdk_adapter
    subscription --> subscription_adapter
    sdk_adapter --> normalized
    subscription_adapter --> normalized
    normalized --> assembler
    assembler --> stream
    stream --> agent
    agent --> agent_event
    agent --> history
```

`assemble_stream` emits `StreamStartEvent` before consuming the first normalized event. Later `CREATED` events mutate that same shared assistant message with the provider response id.

## Response Start

```mermaid
flowchart TD
    start["assemble_stream starts"]
    stream_start["StreamStartEvent(type=start, message=empty AssistantMessage)"]
    agent_turn_start["TurnStartEvent"]
    agent_message_start["MessageStartEvent(message=AssistantMessage)"]

    raw_created["SDK: ResponseCreatedEvent<br/>Subscription: response.created"]
    normalized_created["NormalizedEventType.CREATED<br/>response_id"]
    mutate_response_id["Assembler mutates message.response_id<br/>no new StreamEvent"]

    start --> stream_start
    stream_start --> agent_turn_start
    agent_turn_start --> agent_message_start

    raw_created --> normalized_created
    normalized_created --> mutate_response_id
    mutate_response_id -. "same shared message object" .-> agent_message_start
```

## Reasoning Events

```mermaid
flowchart TD
    raw_reasoning_added["SDK: ResponseOutputItemAddedEvent(item=reasoning)<br/>Subscription: response.output_item.added item.type=reasoning"]
    normalized_reasoning_added["REASONING_ADDED"]
    start_reasoning["Assembler creates ReasoningBlock<br/>active_block=ReasoningBlock"]
    reasoning_start["ReasoningStartEvent"]
    agent_reasoning_start["MessageUpdateEvent(stream_event=reasoning_start)"]

    raw_reasoning_delta["SDK: ResponseReasoningSummaryTextDeltaEvent<br/>SDK: ResponseReasoningTextDeltaEvent<br/>Subscription: response.reasoning_summary_text.delta<br/>Subscription: response.reasoning_text.delta"]
    normalized_reasoning_delta["REASONING_DELTA"]
    append_reasoning["Append delta to ReasoningBlock.summary_text"]
    reasoning_delta["ReasoningDeltaEvent"]
    agent_reasoning_delta["MessageUpdateEvent(stream_event=reasoning_delta)"]

    raw_reasoning_part_done["SDK: ResponseReasoningSummaryPartDoneEvent<br/>Subscription: response.reasoning_summary_part.done"]
    normalized_reasoning_separator["REASONING_DELTA(delta='\\n\\n')"]

    raw_reasoning_done["SDK: ResponseOutputItemDoneEvent(item=reasoning)<br/>Subscription: response.output_item.done item.type=reasoning"]
    normalized_reasoning_done["REASONING_DONE<br/>summary_text + reasoning_signature"]
    finalize_reasoning["Finalize ReasoningBlock<br/>active_block=None"]
    reasoning_end["ReasoningEndEvent"]
    agent_reasoning_end["MessageUpdateEvent(stream_event=reasoning_end)"]

    raw_reasoning_added --> normalized_reasoning_added --> start_reasoning --> reasoning_start --> agent_reasoning_start
    raw_reasoning_delta --> normalized_reasoning_delta --> append_reasoning --> reasoning_delta --> agent_reasoning_delta
    raw_reasoning_part_done --> normalized_reasoning_separator --> append_reasoning
    raw_reasoning_done --> normalized_reasoning_done --> finalize_reasoning --> reasoning_end --> agent_reasoning_end
```

## Text And Refusal Events

```mermaid
flowchart TD
    raw_message_added["SDK: ResponseOutputItemAddedEvent(item=message)<br/>Subscription: response.output_item.added item.type=message"]
    normalized_message_added["MESSAGE_ADDED"]
    start_text["Assembler creates TextBlock<br/>active_block=TextBlock<br/>active_text_part_type=None"]
    text_start["TextStartEvent"]
    agent_text_start["MessageUpdateEvent(stream_event=text_start)"]

    raw_part_added["SDK: ResponseContentPartAddedEvent<br/>Subscription: response.content_part.added"]
    normalized_part["MESSAGE_TEXT_PART<br/>part_type=output_text | refusal | None"]
    activate_part["Set active_text_part_type<br/>no StreamEvent"]

    raw_output_delta["SDK: ResponseTextDeltaEvent<br/>Subscription: response.output_text.delta"]
    normalized_output_delta["MESSAGE_TEXT_DELTA(part_type=output_text)"]
    raw_refusal_delta["SDK: ResponseRefusalDeltaEvent<br/>Subscription: response.refusal.delta"]
    normalized_refusal_delta["MESSAGE_TEXT_DELTA(part_type=refusal)"]
    part_guard["Append only if active_text_part_type matches delta part_type"]
    text_delta["TextDeltaEvent"]
    agent_text_delta["MessageUpdateEvent(stream_event=text_delta)"]

    raw_message_done["SDK: ResponseOutputItemDoneEvent(item=message)<br/>Subscription: response.output_item.done item.type=message"]
    normalized_message_done["MESSAGE_DONE<br/>final text + phase"]
    finalize_text["Finalize TextBlock<br/>active_block=None<br/>active_text_part_type=None"]
    text_end["TextEndEvent"]
    agent_text_end["MessageUpdateEvent(stream_event=text_end)"]

    raw_message_added --> normalized_message_added --> start_text --> text_start --> agent_text_start
    raw_part_added --> normalized_part --> activate_part
    raw_output_delta --> normalized_output_delta --> part_guard
    raw_refusal_delta --> normalized_refusal_delta --> part_guard
    activate_part --> part_guard --> text_delta --> agent_text_delta
    raw_message_done --> normalized_message_done --> finalize_text --> text_end --> agent_text_end
```

Unsupported content parts normalize to `MESSAGE_TEXT_PART(part_type=None)`. That clears text accumulation until another supported `output_text` or `refusal` part becomes active.

## Tool Call Events

```mermaid
flowchart TD
    raw_tool_added["SDK: ResponseOutputItemAddedEvent(item=function_call)<br/>Subscription: response.output_item.added item.type=function_call"]
    normalized_tool_added["TOOL_CALL_ADDED<br/>provider_item_id + call_id + name + arguments"]
    start_tool["Assembler creates ToolCallBlock<br/>active_block=ToolCallBlock"]
    tool_start["ToolCallStartEvent"]
    agent_tool_start_update["MessageUpdateEvent(stream_event=tool_call_start)"]

    raw_tool_delta["SDK: ResponseFunctionCallArgumentsDeltaEvent<br/>Subscription: response.function_call_arguments.delta"]
    normalized_tool_delta["TOOL_CALL_ARGUMENTS_DELTA"]
    tool_delta_event["ToolCallDeltaEvent"]
    agent_tool_delta_update["MessageUpdateEvent(stream_event=tool_call_delta)"]

    raw_tool_args_done["SDK: ResponseFunctionCallArgumentsDoneEvent<br/>Subscription: response.function_call_arguments.done"]
    normalized_tool_args_done["TOOL_CALL_ARGUMENTS_DONE<br/>parsed arguments"]
    replace_args["Replace ToolCallBlock.arguments<br/>no StreamEvent"]

    raw_tool_done["SDK: ResponseOutputItemDoneEvent(item=function_call)<br/>Subscription: response.output_item.done item.type=function_call"]
    normalized_tool_done["TOOL_CALL_DONE<br/>final tool call data"]
    finalize_tool["Finalize ToolCallBlock<br/>active_block=None"]
    tool_end["ToolCallEndEvent"]
    agent_tool_end_update["MessageUpdateEvent(stream_event=tool_call_end)"]

    raw_tool_added --> normalized_tool_added --> start_tool --> tool_start --> agent_tool_start_update
    raw_tool_delta --> normalized_tool_delta --> tool_delta_event --> agent_tool_delta_update
    raw_tool_args_done --> normalized_tool_args_done --> replace_args
    raw_tool_done --> normalized_tool_done --> finalize_tool --> tool_end --> agent_tool_end_update
```

`ToolCallDeltaEvent` reports the raw argument delta for UI streaming. Parsed arguments are stored on the `ToolCallBlock` when `TOOL_CALL_ARGUMENTS_DONE` or `TOOL_CALL_DONE` arrives.

## Terminal Events And Agent Finalization

```mermaid
flowchart TD
    raw_completed["SDK: ResponseCompletedEvent<br/>Subscription: response.completed or response.done(status=completed)"]
    normalized_completed["COMPLETED<br/>stop_reason=stop | tool_use"]
    stream_done["StreamDoneEvent(message=AssistantMessage)"]
    message_end["MessageEndEvent(message=AssistantTurn)"]

    has_tools{"AssistantTurn contains ToolCallBlock?"}
    no_tools["TurnEndEvent(tool_results=[])"]
    tool_start["ToolExecutionStartEvent"]
    tool_end["ToolExecutionEndEvent"]
    tool_result["Append ToolResultTurn to history"]
    tool_turn_end["TurnEndEvent(tool_results=[...])"]
    follow_up["Agent starts another provider stream"]

    raw_incomplete_length["SDK: ResponseIncompleteEvent<br/>Subscription: response.incomplete or response.done(status=incomplete)<br/>reason=max_output_tokens"]
    normalized_incomplete_length["INCOMPLETE(stop_reason=length)"]

    raw_incomplete_error["SDK: ResponseIncompleteEvent<br/>Subscription: response.incomplete or response.done(status=incomplete)<br/>reason=content_filter"]
    normalized_incomplete_error["INCOMPLETE(stop_reason=error)"]
    stream_error_from_incomplete["StreamErrorEvent(error=AssistantMessage)"]

    raw_failed["SDK: ResponseFailedEvent<br/>SDK: ResponseErrorEvent<br/>Subscription: response.failed<br/>Subscription: error"]
    normalized_failed["FAILED(message)"]
    stream_error["StreamErrorEvent(error=AssistantMessage)"]
    error_message_end["MessageEndEvent(message=AssistantTurn status=error)"]
    error_turn_end["TurnEndEvent(tool_results=[])"]

    raw_completed --> normalized_completed --> stream_done --> message_end --> has_tools
    raw_incomplete_length --> normalized_incomplete_length --> stream_done
    has_tools -- no --> no_tools
    has_tools -- yes --> tool_start --> tool_end --> tool_result --> tool_turn_end --> follow_up

    raw_incomplete_error --> normalized_incomplete_error --> stream_error_from_incomplete --> error_message_end --> error_turn_end
    raw_failed --> normalized_failed --> stream_error --> error_message_end
```

The agent appends the finalized assistant turn to history on `StreamDoneEvent` and `StreamErrorEvent`. If the completed assistant turn contains tool calls, the agent executes tools, appends tool-result turns, ends the current turn, and then requests a follow-up assistant stream.

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
