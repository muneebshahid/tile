from textual.app import App, ComposeResult

from agent.agent import Agent
from agent.types import AgentEndEvent, AgentEvent, MessageEndEvent, MessageUpdateEvent
from ai.openai.provider import stream
from ai.types.conversation import (
    AssistantTurn,
    ConversationItem,
    ToolResultTurn,
    UserMessage,
)
from ai.types.stream import AssistantBlock, AssistantMessage, TextBlock
from settings import settings
from ui.widgets import (
    AgentMessageWidget,
    InputSection,
    OutputSection,
    UserMessageWidget,
)


def create_agent() -> Agent:
    return Agent(
        stream_fn=stream,
        model=settings.openai_model,
    )


class PiyApp(App[None]):
    """Root Textual application for piy."""

    TITLE = "piy"
    CSS_PATH = "app.tcss"

    def __init__(self, agent: Agent | None = None) -> None:
        super().__init__()
        self._agent = agent or create_agent()
        self._active_message: AssistantMessage | None = None
        self._is_running = False

    def on_mount(self) -> None:
        """Focus the input area when the app starts."""

        self.query_one(InputSection).focus()

    async def on_input_section_submitted(
        self,
        event: InputSection.Submitted,
    ) -> None:
        """Handle prompt submission from the input widget."""

        if self._is_running:
            return

        self._agent.add_user_message(event.text)
        self._clear_input()
        await self._refresh_output()
        self._is_running = True
        self.run_worker(self._consume_agent_events(), exclusive=True)

    def compose(self) -> ComposeResult:
        yield OutputSection(messages=self._build_output_messages())
        yield InputSection()

    async def _consume_agent_events(self) -> None:
        async for event in self._agent.run():
            await self._handle_agent_event(event)

    async def _handle_agent_event(self, event: AgentEvent) -> None:
        if isinstance(event, MessageUpdateEvent):
            self._active_message = event.message
        elif isinstance(event, MessageEndEvent):
            self._active_message = None
        elif isinstance(event, AgentEndEvent):
            self._is_running = False

        await self._refresh_output()

    async def _refresh_output(self) -> None:
        output = self.query_one(OutputSection)
        await output.set_messages(self._build_output_messages())

    def _clear_input(self) -> None:
        self.query_one(InputSection).clear()

    def _build_output_messages(self) -> list[UserMessageWidget | AgentMessageWidget]:
        messages = self._build_history_widgets()

        if active_message := self._build_active_message_widget():
            messages.append(active_message)

        if messages:
            return messages

        return [AgentMessageWidget("Welcome to piy.")]

    def _build_history_widgets(self) -> list[UserMessageWidget | AgentMessageWidget]:
        messages: list[UserMessageWidget | AgentMessageWidget] = []

        for item in self._agent.history:
            if widget := self._build_history_widget(item):
                messages.append(widget)

        return messages

    def _build_history_widget(
        self,
        item: ConversationItem,
    ) -> UserMessageWidget | AgentMessageWidget | None:
        if isinstance(item, UserMessage):
            return UserMessageWidget(item.content)
        if isinstance(item, AssistantTurn):
            if text := _extract_assistant_text(item.content, item.error_message):
                return AgentMessageWidget(text)
        if isinstance(item, ToolResultTurn):
            return None
        return None

    def _build_active_message_widget(self) -> AgentMessageWidget | None:
        if self._active_message is None:
            return None

        if text := _extract_assistant_text(
            self._active_message.content,
            self._active_message.error_message,
        ):
            return AgentMessageWidget(text)

        return None


def _extract_assistant_text(
    blocks: list[AssistantBlock],
    error_message: str | None,
) -> str:
    text = "".join(block.text for block in blocks if isinstance(block, TextBlock))
    return text or error_message or ""
