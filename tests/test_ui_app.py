import asyncio
from unittest.mock import patch

from main import main
from ui import PiyApp
from ui.widgets import InputTextArea, OutputTextArea


def test_piy_app_can_be_constructed() -> None:
    app = PiyApp()

    assert app.title == "piy"
    composed_widgets = list(app.compose())

    assert len(composed_widgets) == 2
    assert isinstance(composed_widgets[0], OutputTextArea)
    assert composed_widgets[0].id == "output"
    assert composed_widgets[0].read_only is True
    assert isinstance(composed_widgets[1], InputTextArea)
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

            assert app.focused is app.query_one(InputTextArea)

    asyncio.run(_run())


def test_clicking_output_does_not_move_focus_from_input() -> None:
    async def _run() -> None:
        app = PiyApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#output")
            await pilot.pause()

            assert app.focused is app.query_one(InputTextArea)

    asyncio.run(_run())
