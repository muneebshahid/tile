"""Tests for the default file edit tool scaffold."""

from pathlib import Path

import pytest

import agent.tools.edit as edit


def test_edit_schema_requires_path_and_edits() -> None:
    """Require a target path and at least one edit collection argument."""

    assert edit.tool.input_schema["required"] == ["path", "edits"]


def test_edit_schema_exposes_edit_controls() -> None:
    """Expose path plus edits inputs."""

    properties = edit.tool.input_schema["properties"]

    assert edit.tool.name == "edit"
    assert isinstance(properties, dict)
    assert set(properties) == {"path", "edits"}


def test_edit_schema_describes_each_replacement() -> None:
    """Expose oldText and newText for every edit item."""

    properties = edit.tool.input_schema["properties"]
    assert isinstance(properties, dict)

    edits_schema = properties["edits"]
    assert isinstance(edits_schema, dict)

    item_schema = edits_schema["items"]
    assert isinstance(item_schema, dict)

    item_properties = item_schema["properties"]
    assert isinstance(item_properties, dict)

    assert set(item_properties) == {"oldText", "newText"}
    assert item_schema["required"] == ["oldText", "newText"]
    assert item_schema["additionalProperties"] is False


def test_edit_resolves_relative_path_against_supplied_cwd(tmp_path: Path) -> None:
    """Resolve relative edit paths against the supplied tool cwd."""

    assert edit._resolve_path("sample.txt", tmp_path) == tmp_path / "sample.txt"


def test_edit_expands_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand home-directory markers in edit paths."""

    monkeypatch.setenv("HOME", str(tmp_path))

    assert edit._resolve_path("~/sample.txt", Path.cwd()) == tmp_path / "sample.txt"


def test_edit_normalizes_unicode_spaces(tmp_path: Path) -> None:
    """Resolve paths typed with uncommon Unicode spaces."""

    file_path = tmp_path / "my file.txt"
    requested_path = str(file_path).replace(" ", "\u00a0")

    assert edit._resolve_path(requested_path, Path.cwd()) == file_path


def test_edit_strips_at_prefix(tmp_path: Path) -> None:
    """Strip leading at signs from referenced edit paths."""

    assert edit._resolve_path("@sample.txt", tmp_path) == tmp_path / "sample.txt"
