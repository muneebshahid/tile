"""Transcript and input widgets for the Textual application."""

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
    BODY_CLASS = "message-body"

    def __init__(self, text: str) -> None:
        self._text = text
        super().__init__()

    @property
    def text(self) -> str:
        """Return the current message text."""

        return self._text

    def compose(self) -> ComposeResult:
        """Render the message body."""

        yield Static(self._text, classes=self.BODY_CLASS, markup=False)

    def set_text(self, text: str) -> None:
        """Replace the rendered message body text in place."""

        self._text = text
        if self.is_mounted:
            self._body().update(text)

    def append_text(self, delta: str) -> None:
        """Append streamed text to the rendered message body."""

        self.set_text(f"{self._text}{delta}")

    def _body(self) -> Static:
        """Return the mounted body widget."""

        return self.query_one(f".{self.BODY_CLASS}", Static)


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
        empty_state_message: str | None = None,
    ) -> None:
        super().__init__(id="output")
        self._messages = list(messages or [])
        self._empty_state_widget = _build_empty_state_widget(empty_state_message)

    def compose(self) -> ComposeResult:
        """Render transcript messages or the empty-state placeholder."""

        if self._messages:
            yield from self._messages
            return

        if self._empty_state_widget is not None:
            yield self._empty_state_widget

    async def add_message(self, message: TranscriptMessageWidget) -> None:
        """Add a message to the end of the transcript."""

        await self._remove_empty_state()
        self._messages.append(message)
        await self.mount(message)
        self.scroll_end(animate=False)

    async def remove_message(self, message: TranscriptMessageWidget) -> None:
        """Remove a message from the transcript."""

        if message not in self._messages:
            return

        self._messages.remove(message)
        await message.remove()
        self.scroll_end(animate=False)

    async def _remove_empty_state(self) -> None:
        """Unmount the empty-state placeholder before adding real content."""
        if self._empty_state_widget is not None and self._empty_state_widget.is_mounted:
            await self._empty_state_widget.remove()


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

        if event.key in {"alt+enter", "ctrl+j"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return

        if event.key != "enter":
            return

        event.prevent_default()
        event.stop()

        if not (prompt := self.text.strip()):
            return

        self.post_message(self.Submitted(prompt))


def _build_empty_state_widget(
    empty_state_message: str | None,
) -> AgentMessageWidget | None:
    """Create the optional transcript empty-state widget."""

    if empty_state_message is None:
        return None

    return AgentMessageWidget(empty_state_message)
