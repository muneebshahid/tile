"""Application entrypoint for the Textual chat UI."""

from collections.abc import Sequence
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding

from agent.agent import Agent
from agent.tools import build_tools
from agent.types import (
    AgentEndEvent,
    AgentEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
)
from ai.openai.provider import stream_api
from ai.types.stream import AssistantBlock, TextBlock, TextDeltaEvent, TextStartEvent
from settings import settings
from ui.textual_driver import PiyLinuxDriver
from ui.widgets import (
    AgentMessageWidget,
    InputSection,
    OutputSection,
    UserMessageWidget,
    TranscriptMessageWidget,
)


def create_agent() -> Agent:
    """Build the default agent used by the UI."""

    cwd = Path.cwd().resolve()
    return Agent(
        stream_fn=stream_api,
        model=settings.openai_model,
        tools=build_tools(cwd),
        cwd=cwd,
    )


class PiyApp(App[None]):
    """Root Textual application for piy."""

    TITLE = "piy"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding(
            key="ctrl+c",
            action="quit",
            description="Quit",
            show=False,
            system=True,
        ),
    ]

    def __init__(self, agent: Agent | None = None) -> None:
        super().__init__(driver_class=PiyLinuxDriver)
        self._agent = agent or create_agent()
        self._active_widget: AgentMessageWidget | None = None
        self._is_running = False

    async def on_mount(self) -> None:
        """Focus the input area when the app starts."""

        self.query_one(InputSection).focus()

    async def on_input_section_submitted(
        self,
        event: InputSection.Submitted,
    ) -> None:
        """Handle prompt submission from the input widget."""

        if self._is_running:
            return

        user_message = self._agent.add_user_message(event.text)
        await self._add_to_output(UserMessageWidget(user_message.content))
        self._is_running = True
        self._clear_input()
        self.run_worker(self._consume_agent_events(), exclusive=True)

    def compose(self) -> ComposeResult:
        """Compose the application layout."""

        yield OutputSection(empty_state_message="Welcome to piy.")
        yield InputSection()

    async def _consume_agent_events(self) -> None:
        """Consume agent events until the current run completes."""

        async for event in self._agent.run():
            await self._handle_agent_event(event)

    async def _handle_agent_event(self, event: AgentEvent) -> None:
        """Route agent events into incremental transcript updates."""

        match event:
            case MessageStartEvent():
                self._active_widget = None
            case MessageUpdateEvent():
                await self._handle_message_update(event)
            case MessageEndEvent():
                await self._handle_message_finalized(event)
            case AgentEndEvent():
                self._is_running = False

    async def _handle_message_update(self, event: MessageUpdateEvent) -> None:
        """Apply streamed text updates to the active assistant widget."""

        match event.stream_event:
            case TextStartEvent():
                active_widget = await self._get_or_activate_active_widget()
                active_widget.set_text(
                    _extract_assistant_text(event.message.blocks, None)
                )
            case TextDeltaEvent(delta=delta):
                active_widget = await self._get_or_activate_active_widget()
                active_widget.append_text(delta)
                self.query_one(OutputSection).scroll_end(animate=False)

    async def _handle_message_finalized(self, event: MessageEndEvent) -> None:
        """Reconcile the active widget with the final assistant message."""
        message = event.message

        if text := _extract_assistant_text(message.blocks, message.error_message):
            active_widget = await self._get_or_activate_active_widget()
            active_widget.set_text(text)
            self.query_one(OutputSection).scroll_end(animate=False)

        self._active_widget = None

    async def _get_or_activate_active_widget(self) -> AgentMessageWidget:
        """Return the active assistant widget, activating it when needed."""

        if self._active_widget is None:
            self._active_widget = AgentMessageWidget("")
            await self._add_to_output(self._active_widget)

        return self._active_widget

    async def _add_to_output(self, message: TranscriptMessageWidget) -> None:
        """Append a transcript message and keep the latest content visible."""

        output = self.query_one(OutputSection)
        await output.add_message(message)

    def _clear_input(self) -> None:
        """Clear the input area after submission."""

        self.query_one(InputSection).clear()


def _extract_assistant_text(
    blocks: Sequence[AssistantBlock],
    error_message: str | None,
) -> str:
    """Extract the user-visible assistant text from structured content."""

    text = "".join(block.text for block in blocks if isinstance(block, TextBlock))
    return text or error_message or ""
