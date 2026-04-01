from textual.app import App, ComposeResult

from ui.widgets import (
    AgentMessageWidget,
    InputSection,
    OutputSection,
    UserMessageWidget,
)


class PiyApp(App[None]):
    """Root Textual application for piy."""

    TITLE = "piy"
    CSS_PATH = "app.tcss"

    def __init__(self) -> None:
        super().__init__()
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
        """Compose the MVP UI shell."""

        yield OutputSection(messages=self._output_messages)
        yield InputSection()

    async def _append_prompt_to_history(self, prompt: str) -> None:
        output = self.query_one(OutputSection)
        await output.append_message(UserMessageWidget(prompt))

    def _clear_input(self) -> None:
        self.query_one(InputSection).clear()
