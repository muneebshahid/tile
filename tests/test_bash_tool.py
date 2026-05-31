"""Tests for the default shell command tool scaffold."""

import agent.tools.bash as bash


def test_bash_schema_requires_only_command() -> None:
    """Require only the command so callers can omit optional timeout."""

    assert bash.tool.input_schema["required"] == ["command"]


def test_bash_schema_exposes_command_controls() -> None:
    """Expose shell command inputs without execution-injected fields."""

    properties = bash.tool.input_schema["properties"]

    assert bash.tool.name == "bash"
    assert isinstance(properties, dict)
    assert set(properties) == {"command", "timeout"}
