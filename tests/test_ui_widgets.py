import asyncio
from typing import cast

from textual import events
from textual.app import App, ComposeResult
from textual.content import Content
from textual.widgets import Static

from ui.widgets import AgentMessageWidget, InputSection


class InputHarnessApp(App[None]):
    """Minimal app used to verify input widget behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputSection()

    def on_mount(self) -> None:
        self.query_one(InputSection).focus()

    def on_input_section_submitted(self, event: InputSection.Submitted) -> None:
        self.submissions.append(event.text)


class MessageHarnessApp(App[None]):
    """Minimal app used to verify transcript widget rendering behavior."""

    def __init__(self, message_widget: AgentMessageWidget) -> None:
        super().__init__()
        self.message_widget = message_widget

    def compose(self) -> ComposeResult:
        yield self.message_widget


def test_input_text_area_posts_submitted_message_on_enter() -> None:
    async def _run() -> None:
        app = InputHarnessApp()

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)

            input_area.load_text("Hello from widget")
            await pilot.press("enter")
            await pilot.pause()

            assert app.submissions == ["Hello from widget"]

    asyncio.run(_run())


def test_input_text_area_inserts_newline_on_alt_enter() -> None:
    async def _run() -> None:
        app = InputHarnessApp()

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)

            input_area.load_text("Hello")
            input_area.move_cursor((0, 5))
            input_area.on_key(events.Key("alt+enter", None))
            await pilot.pause()

            assert app.submissions == []
            assert input_area.text == "Hello\n"

    asyncio.run(_run())


def test_input_text_area_inserts_newline_on_ctrl_j() -> None:
    async def _run() -> None:
        app = InputHarnessApp()

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)

            input_area.load_text("Hello")
            input_area.move_cursor((0, 5))
            input_area.on_key(events.Key("ctrl+j", None))
            await pilot.pause()

            assert app.submissions == []
            assert input_area.text == "Hello\n"

    asyncio.run(_run())


def test_transcript_message_widget_updates_rendered_body_in_place() -> None:
    async def _run() -> None:
        message_widget = AgentMessageWidget("Hello")
        app = MessageHarnessApp(message_widget)

        async with app.run_test() as pilot:
            await pilot.pause()

            message_widget.append_text(" world")
            await pilot.pause()

            assert message_widget.text == "Hello world"
            content = cast(Content, message_widget.query_one(Static).render())
            assert content.plain == "Hello world"

    asyncio.run(_run())
