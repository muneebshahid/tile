from unittest.mock import patch

from main import main
from ui import PiyApp


def test_piy_app_can_be_constructed() -> None:
    app = PiyApp()

    assert app.title == "piy"


def test_main_uses_piy_app() -> None:
    with patch.object(PiyApp, "run") as run_mock:
        main()

    run_mock.assert_called_once_with()
