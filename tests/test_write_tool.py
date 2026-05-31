"""Tests for the default file write tool scaffold."""

import pytest

import agent.tools.write as write


def test_write_schema_requires_path_and_content() -> None:
    """Require the file path and replacement content."""

    assert write.tool.input_schema["required"] == ["path", "content"]


def test_write_schema_exposes_write_controls() -> None:
    """Expose only path and content inputs."""

    properties = write.tool.input_schema["properties"]

    assert write.tool.name == "write"
    assert isinstance(properties, dict)
    assert set(properties) == {"path", "content"}
