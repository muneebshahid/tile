"""The following part of the code was vibesloped, there might be a better
way to do this.

Textual driver customizations for modified Enter handling.

Textual's default Linux driver uses ``XTermParser`` to translate raw terminal
input into ``textual.events.Key`` instances. In this environment, the terminal
emits the key combination we want to treat as "newline" (``shift+enter``) as
``ESC + CR`` (``\x1b\r``). That is effectively the same byte sequence as Alt+Enter.

The stock Textual parser normalizes that sequence into a plain ``enter`` event
before the application code sees it. Once that normalization happens, the input
widget can no longer distinguish between:

- Enter, which should submit the prompt
- Alt+Enter, which should insert a newline

To solve this,``piy`` needs a small parser customization at the driver boundary.

This module provides that customization:

- ``PiyXTermParser`` maps raw ``ESC + CR`` to ``alt+enter``
- ``PiyLinuxDriver`` installs that parser in place of Textual's default one

That keeps the application layer simple: the input widget only needs to handle
``alt+enter`` and ``ctrl+j`` as newline keys, while plain ``enter`` remains a
submit action.
"""

from __future__ import annotations

import os
import selectors
from codecs import getincrementaldecoder

from textual import events
from textual._loop import loop_last
from textual._parser import ParseError
from textual._xterm_parser import XTermParser
from textual.drivers.linux_driver import LinuxDriver


class PiyXTermParser(XTermParser):
    """Interpret ``ESC + CR`` as ``alt+enter`` instead of plain Enter.

    The terminal used with ``piy`` sends the modified Enter combination as an
    "Alt-prefixed carriage return" sequence. Textual's base parser treats that
    as an ordinary Enter key, which erases the distinction the input widget
    needs.

    This override preserves that distinction by translating the specific raw
    sequence into a synthetic ``alt+enter`` key event. All other sequences are
    delegated to Textual unchanged.
    """

    def _sequence_to_key_events(
        self,
        sequence: str,
        alt: bool = False,
    ):
        """Convert a parsed terminal sequence into one or more key events.

        Args:
            sequence: The raw character sequence decoded by ``XTermParser``.
            alt: Whether the parser already determined that the sequence is
                Alt-prefixed.

        Yields:
            ``textual.events.Key`` instances for the sequence.

        The only customized case is ``alt=True`` with ``sequence == "\\r"``.
        That combination represents the terminal's Alt+Enter encoding in this
        environment, so we emit ``alt+enter`` explicitly.
        """

        if alt and sequence == "\r":
            yield events.Key("alt+enter", None)
            return

        yield from super()._sequence_to_key_events(sequence, alt)


class PiyLinuxDriver(LinuxDriver):
    """Linux driver that installs ``PiyXTermParser`` for terminal input.

    Textual's ``LinuxDriver`` hardcodes ``XTermParser`` inside
    ``run_input_thread``. Because the key normalization problem happens inside
    that parser, the clean fix is to override the driver method and swap in our
    parser implementation while preserving the rest of Textual's event loop.
    """

    def run_input_thread(self) -> None:
        """Read terminal bytes, parse them, and dispatch Textual events.

        This method intentionally mirrors Textual's default implementation with
        one behavioral change: it constructs ``PiyXTermParser`` instead of the
        stock ``XTermParser``. That keeps the driver behavior aligned with
        upstream Textual while giving ``piy`` control over the modified Enter
        sequence that needs to survive parsing.
        """

        selector = selectors.SelectSelector()
        selector.register(self.fileno, selectors.EVENT_READ)

        fileno = self.fileno
        event_read = selectors.EVENT_READ

        parser = PiyXTermParser(self._debug)
        feed = parser.feed
        tick = parser.tick

        utf8_decoder = getincrementaldecoder("utf-8")().decode
        decode = utf8_decoder
        read = os.read

        def process_selector_events(
            selector_events: list[tuple[selectors.SelectorKey, int]],
            final: bool = False,
        ) -> None:
            """Decode available terminal input and forward parsed events.

            Args:
                selector_events: Ready file-descriptor events returned by the
                    selector.
                final: Whether this is the final drain pass during shutdown.

            The nested function matches Textual's original structure. Keeping it
            local avoids broadening the driver's public API while making the
            byte-decoding and parser-feeding flow easier to read.
            """

            for last, (_selector_key, mask) in loop_last(selector_events):
                if mask & event_read:
                    unicode_data = decode(read(fileno, 1024 * 4), final=final and last)
                    if not unicode_data:
                        break
                    for event in feed(unicode_data):
                        self.process_message(event)
            for event in tick():
                self.process_message(event)

        try:
            while not self.exit_event.is_set():
                process_selector_events(selector.select(0.1))
            selector.unregister(self.fileno)
            process_selector_events(selector.select(0.1), final=True)
        finally:
            selector.close()
            try:
                for event in feed(""):
                    pass
            except (EOFError, ParseError):
                pass
