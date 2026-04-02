from textual.events import Key

from ui.textual_driver import PiyXTermParser


def test_parser_maps_escape_enter_to_alt_enter() -> None:
    parser = PiyXTermParser(debug=False)

    events = [*parser.feed("\x1b\r"), *parser.feed("")]

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, Key)
    assert event.key == "alt+enter"
    assert event.character is None
