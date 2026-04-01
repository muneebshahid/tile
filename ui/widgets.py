from __future__ import annotations

from collections.abc import Sequence

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static, TextArea


class TranscriptMessageWidget(Vertical):
    """Base widget for a single transcript message."""

    can_focus = False

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(self.text, classes="message-body", markup=False)


class UserMessageWidget(TranscriptMessageWidget):
    """Visual representation of a user-authored message."""

    DEFAULT_CLASSES = "transcript-message user-message"


class AgentMessageWidget(TranscriptMessageWidget):
    """Visual representation of an agent-authored message."""

    DEFAULT_CLASSES = "transcript-message agent-message"


class OutputSection(VerticalScroll):
    """Scrollable transcript composed of role-specific message widgets."""

    can_focus = False

    def __init__(
        self,
        messages: Sequence[TranscriptMessageWidget] | None = None,
    ) -> None:
        super().__init__(id="output")
        self._messages = list(messages or [])

    def compose(self) -> ComposeResult:
        yield from self._messages

    async def append_message(self, message: TranscriptMessageWidget) -> None:
        """Append a new message to the transcript."""

        self._messages.append(message)
        await self.mount(message)
        self.scroll_end(animate=False)


class InputSection(TextArea):
    """Editable prompt area shown below the output."""

    class Submitted(Message):
        """Message sent when the user submits the input area."""

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(self) -> None:
        super().__init__(id="input")

    def on_key(self, event: events.Key) -> None:
        """Convert Enter into a submit event instead of a newline."""

        if event.key != "enter":
            return

        event.prevent_default()
        event.stop()

        if not (prompt := self.text.strip()):
            return

        self.post_message(self.Submitted(prompt))
