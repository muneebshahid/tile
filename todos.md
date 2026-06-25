# Historical Todos

This file used to track tool-call and server-streaming work from the old
direction. Most of those items are now implemented or superseded.

Use `roadmaps/plan.local.md` as the current planning source for Ori.

Kept constraints that still matter:

- `ToolResultTurn.call_id` must match the corresponding assistant
  `ToolCallBlock.call_id`.
- Do not merge assistant tool calls and tool results into one replay history
  object.
- Keep model-visible history as `UserMessage`, `AssistantTurn`, and
  `ToolResultTurn`.
- Clients are control and transport surfaces; provider calls and tool execution
  stay runtime-side.
