Remaining tool-call work

Replay serialization
- Serialize prior assistant tool calls from `ToolCallBlock` to OpenAI function-call input items in `ai/openai/serialization.py`.
- Serialize `ToolResultTurn` to OpenAI function-call output items in `ai/openai/serialization.py`.
- Extend `serialize_history_items()` to replay completed assistant turns that contain text, reasoning, and tool-call blocks.

Agent wiring
- Extend the `StreamFn` protocol and `Agent.run()` plumbing to accept `tools`.
- Thread `tools` into the provider call in `agent/agent.py`.
- Add explicit handling for tool-call stream events in the agent event dispatcher.
- Keep `_build_assistant_turn()` preserving tool-call blocks so assistant turns can be replayed after a tool-use stop.

Tool execution scaffolding
- Add at least a minimal tool registry or dispatch mapping instead of hardcoding tool execution paths later.
- Implement the first concrete tool and its schema only after the provider and replay plumbing are in place.

Tests
- Add serialization tests for replaying assistant tool calls and `ToolResultTurn`.

Important shape constraints
- `ToolResultTurn.call_id` must exactly match the `call_id` on the corresponding assistant `ToolCallBlock`.
- Do not introduce a combined tool turn that stores both call and result together.
- Keep the history shape as:
  - `UserMessage`
  - `AssistantTurn` with text, reasoning, and tool-call blocks
  - `ToolResultTurn`
