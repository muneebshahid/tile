"""Narrowing helpers for optional payloads on in-band lifecycle end events."""

from tile.types.conversation import AssistantTurn
from tile.types.tool_execution import ToolExecutionOutcome


def expect_turn(turn: AssistantTurn | None) -> AssistantTurn:
    """Return an in-band end event's assistant turn, asserting presence."""

    assert turn is not None
    return turn


def expect_outcome(outcome: ToolExecutionOutcome | None) -> ToolExecutionOutcome:
    """Return an in-band end event's tool outcome, asserting presence."""

    assert outcome is not None
    return outcome
