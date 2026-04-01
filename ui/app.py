from textual.app import App, ComposeResult

from agent.agent import Agent
from ai.openai.provider import stream
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
        self._output_messages = [
            AgentMessageWidget("Welcome to piy."),
        ]

    def on_mount(self) -> None:
        """Focus the input area when the app starts."""

        self.query_one(InputSection).focus()

    async def on_input_section_submitted(
        self,
        event: InputSection.Submitted,
    ) -> None:
        """Handle prompt submission from the input widget."""

        await self._append_prompt_to_history(event.text)
        self._clear_input()

    def compose(self) -> ComposeResult:
        yield OutputSection(messages=self._output_messages)
        yield InputSection()

    async def _append_prompt_to_history(self, prompt: str) -> None:
        output = self.query_one(OutputSection)
        await output.append_message(UserMessageWidget(prompt))

    def _clear_input(self) -> None:
        self.query_one(InputSection).clear()
