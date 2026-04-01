import asyncio
from unittest.mock import patch

from main import main
from ui import PiyApp
from ui.widgets import (
    AgentMessageWidget,
    InputSection,
    OutputSection,
    UserMessageWidget,
)


def test_piy_app_can_be_constructed() -> None:
    app = PiyApp()

    assert app.title == "piy"
    composed_widgets = list(app.compose())
    output_widget = composed_widgets[0]

    assert len(composed_widgets) == 2
    assert isinstance(output_widget, OutputSection)
    assert output_widget.id == "output"
    assert len(output_widget._messages) == 1
    assert isinstance(output_widget._messages[0], AgentMessageWidget)
    assert isinstance(composed_widgets[1], InputSection)
    assert composed_widgets[1].id == "input"
    assert composed_widgets[1].read_only is False


def test_main_uses_piy_app() -> None:
    with patch.object(PiyApp, "run") as run_mock:
        main()

    run_mock.assert_called_once_with()


def test_piy_app_focuses_input_on_mount() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test() as pilot:
            await pilot.pause()

            assert app.focused is app.query_one(InputSection)

    asyncio.run(_run())


def test_clicking_output_does_not_move_focus_from_input() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#output")
            await pilot.pause()

            assert app.focused is app.query_one(InputSection)

    asyncio.run(_run())


def test_pressing_enter_moves_input_text_into_output_history() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test() as pilot:
            input_area = app.query_one(InputSection)
            output_area = app.query_one(OutputSection)

            input_area.load_text("Hello, piy!")
            await pilot.press("enter")
            await pilot.pause()

            assert len(output_area.children) == 2
            message = output_area.children[-1]
            assert isinstance(message, UserMessageWidget)
            assert message.text == "Hello, piy!"
            assert input_area.text == ""

    asyncio.run(_run())


def test_input_area_expands_for_wrapped_content() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test(size=(40, 20)) as pilot:
            input_area = app.query_one(InputSection)

            await pilot.pause()
            initial_height = input_area.size.height

            input_area.load_text(
                "This is a long prompt that should wrap and expand the input area "
                "beyond a single line."
            )
            await pilot.pause()

            assert input_area.size.height > initial_height

    asyncio.run(_run())
