from textual.widgets import TextArea


class OutputTextArea(TextArea):
    """Read-only transcript area shown above the input."""

    can_focus = False

    def __init__(self) -> None:
        super().__init__(
            text="",
            id="output",
            read_only=True,
            soft_wrap=True,
            show_line_numbers=False,
            show_cursor=False,
            placeholder="Welcome to Piy!",
        )


class InputTextArea(TextArea):
    """Editable prompt area shown below the output."""

    def __init__(self) -> None:
        super().__init__(
            text="",
            id="input",
            read_only=False,
            soft_wrap=True,
            show_line_numbers=False,
        )
