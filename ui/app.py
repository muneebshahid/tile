from textual.app import App, ComposeResult

from ui.widgets import InputTextArea, OutputTextArea


class PiyApp(App[None]):
    """Root Textual application for piy."""

    TITLE = "piy"
    CSS_PATH = "app.tcss"

    def on_mount(self) -> None:
        """Focus the input area when the app starts."""

        self.query_one(InputTextArea).focus()

    def compose(self) -> ComposeResult:
        """Compose the MVP UI shell."""

        yield OutputTextArea()
        yield InputTextArea()
