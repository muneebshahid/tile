"""Tests for the default file edit tool."""

from pathlib import Path

import pytest

import ori.tools.edit as edit
from ori.types.tools import EditDetails, ToolResult, ToolTextContent


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


@pytest.mark.asyncio
async def test_edit_replaces_text_in_file(tmp_path: Path) -> None:
    """Replace a unique exact text block in a file."""

    file_path = _write_text(tmp_path / "sample.txt", "Hello, world!")

    result = _text(
        await edit.fn(
            path=str(file_path),
            edits=[{"oldText": "world", "newText": "testing"}],
            cwd=Path.cwd(),
        )
    )

    assert file_path.read_text(encoding="utf-8") == "Hello, testing!"
    assert result == f"Successfully replaced 1 block(s) in {file_path}."


@pytest.mark.asyncio
async def test_edit_replaces_multiple_disjoint_blocks(tmp_path: Path) -> None:
    """Apply multiple replacements against the original file content."""

    file_path = _write_text(tmp_path / "sample.txt", "alpha\nbeta\ngamma\n")

    await edit.fn(
        path=str(file_path),
        edits=[
            {"oldText": "alpha\n", "newText": "ALPHA\n"},
            {"oldText": "gamma\n", "newText": "GAMMA\n"},
        ],
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "ALPHA\nbeta\nGAMMA\n"


@pytest.mark.asyncio
async def test_edit_returns_unified_diff_details(tmp_path: Path) -> None:
    """Return a standard unified diff in edit result details."""

    file_path = _write_text(tmp_path / "sample.txt", "alpha\nbeta\ngamma\n")

    tool_result = await edit.fn(
        path=str(file_path),
        edits=[{"oldText": "beta\n", "newText": "BETA\n"}],
        cwd=Path.cwd(),
    )

    details = _edit_details(tool_result)
    assert details.type == "edit"
    assert details.diff == (
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
    )


@pytest.mark.asyncio
async def test_edit_matches_against_original_content(tmp_path: Path) -> None:
    """Match later edits against original content instead of earlier replacements."""

    file_path = _write_text(tmp_path / "sample.txt", "foo\nbar\nbaz\n")

    await edit.fn(
        path=str(file_path),
        edits=[
            {"oldText": "foo\n", "newText": "foo bar\n"},
            {"oldText": "bar\n", "newText": "BAR\n"},
        ],
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "foo bar\nBAR\nbaz\n"


@pytest.mark.asyncio
async def test_edit_raises_if_text_not_found(tmp_path: Path) -> None:
    """Raise when oldText is not present in the original file."""

    file_path = _write_text(tmp_path / "sample.txt", "Hello, world!")

    with pytest.raises(RuntimeError, match="Could not find the exact text"):
        await edit.fn(
            path=str(file_path),
            edits=[{"oldText": "missing", "newText": "testing"}],
            cwd=Path.cwd(),
        )


@pytest.mark.asyncio
async def test_edit_raises_if_text_is_not_unique(tmp_path: Path) -> None:
    """Raise when oldText appears more than once."""

    file_path = _write_text(tmp_path / "sample.txt", "foo foo foo")

    with pytest.raises(RuntimeError, match="Found 3 occurrences"):
        await edit.fn(
            path=str(file_path),
            edits=[{"oldText": "foo", "newText": "bar"}],
            cwd=Path.cwd(),
        )


@pytest.mark.asyncio
async def test_edit_raises_if_edits_are_empty(tmp_path: Path) -> None:
    """Reject empty edit lists."""

    file_path = _write_text(tmp_path / "sample.txt", "hello\nworld\n")

    with pytest.raises(RuntimeError, match="edits must contain at least one"):
        await edit.fn(path=str(file_path), edits=[], cwd=Path.cwd())


@pytest.mark.asyncio
async def test_edit_raises_if_old_text_is_empty(tmp_path: Path) -> None:
    """Reject empty oldText values."""

    file_path = _write_text(tmp_path / "sample.txt", "hello\nworld\n")

    with pytest.raises(RuntimeError, match="oldText must not be empty"):
        await edit.fn(
            path=str(file_path),
            edits=[{"oldText": "", "newText": "replacement"}],
            cwd=Path.cwd(),
        )


@pytest.mark.asyncio
async def test_edit_raises_if_regions_overlap(tmp_path: Path) -> None:
    """Reject overlapping multi-edit ranges."""

    file_path = _write_text(tmp_path / "sample.txt", "one\ntwo\nthree\n")

    with pytest.raises(RuntimeError, match="overlap"):
        await edit.fn(
            path=str(file_path),
            edits=[
                {"oldText": "one\ntwo\n", "newText": "ONE\nTWO\n"},
                {"oldText": "two\nthree\n", "newText": "TWO\nTHREE\n"},
            ],
            cwd=Path.cwd(),
        )


@pytest.mark.asyncio
async def test_edit_does_not_partially_apply_when_one_edit_fails(
    tmp_path: Path,
) -> None:
    """Leave the file unchanged when any requested edit is invalid."""

    original_content = "alpha\nbeta\ngamma\n"
    file_path = _write_text(tmp_path / "sample.txt", original_content)

    with pytest.raises(RuntimeError, match="Could not find"):
        await edit.fn(
            path=str(file_path),
            edits=[
                {"oldText": "alpha\n", "newText": "ALPHA\n"},
                {"oldText": "missing\n", "newText": "MISSING\n"},
            ],
            cwd=Path.cwd(),
        )

    assert file_path.read_text(encoding="utf-8") == original_content


@pytest.mark.asyncio
async def test_edit_matches_lf_text_and_preserves_crlf(tmp_path: Path) -> None:
    """Match LF oldText against CRLF files and preserve CRLF on write."""

    file_path = _write_text(tmp_path / "sample.txt", "first\r\nsecond\r\nthird\r\n")

    await edit.fn(
        path=str(file_path),
        edits=[{"oldText": "second\n", "newText": "REPLACED\n"}],
        cwd=Path.cwd(),
    )

    assert _read_text(file_path) == "first\r\nREPLACED\r\nthird\r\n"


@pytest.mark.asyncio
async def test_edit_preserves_utf8_bom(tmp_path: Path) -> None:
    """Preserve a leading UTF-8 BOM after editing."""

    file_path = _write_text(tmp_path / "sample.txt", "\ufefffirst\nsecond\nthird\n")

    await edit.fn(
        path=str(file_path),
        edits=[{"oldText": "second\n", "newText": "REPLACED\n"}],
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "\ufefffirst\nREPLACED\nthird\n"


@pytest.mark.asyncio
async def test_edit_fuzzy_matches_trailing_whitespace(tmp_path: Path) -> None:
    """Fallback to fuzzy matching when trailing whitespace differs."""

    file_path = _write_text(
        tmp_path / "sample.txt", "line one   \nline two  \nline three\n"
    )

    await edit.fn(
        path=str(file_path),
        edits=[{"oldText": "line one\nline two\n", "newText": "replaced\n"}],
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "replaced\nline three\n"


@pytest.mark.asyncio
async def test_edit_fuzzy_matches_smart_quotes(tmp_path: Path) -> None:
    """Fallback to fuzzy matching for smart quote variants."""

    file_path = _write_text(
        tmp_path / "sample.txt", "console.log(\u2018hello\u2019);\n"
    )

    await edit.fn(
        path=str(file_path),
        edits=[
            {
                "oldText": "console.log('hello');",
                "newText": "console.log('world');",
            }
        ],
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "console.log('world');\n"


@pytest.mark.asyncio
async def test_edit_fuzzy_matches_unicode_dashes_and_spaces(tmp_path: Path) -> None:
    """Fallback to fuzzy matching for Unicode dashes and spaces."""

    file_path = _write_text(
        tmp_path / "sample.txt", "hello\u00a0world\nrange: 1\u20135\n"
    )

    await edit.fn(
        path=str(file_path),
        edits=[
            {
                "oldText": "hello world\nrange: 1-5\n",
                "newText": "hello universe\nrange: 10-50\n",
            }
        ],
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "hello universe\nrange: 10-50\n"


@pytest.mark.asyncio
async def test_edit_fuzzy_matches_unicode_compatibility_forms(tmp_path: Path) -> None:
    """Fallback to fuzzy matching for NFKC-equivalent text."""

    file_path = _write_text(tmp_path / "sample.txt", "ＡＢＣ１２３\ncafe\u0301\n")

    await edit.fn(
        path=str(file_path),
        edits=[{"oldText": "ABC123\ncafé\n", "newText": "XYZ789\ncoffee\n"}],
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "XYZ789\ncoffee\n"


@pytest.mark.asyncio
async def test_edit_prefers_exact_match_over_fuzzy_match(tmp_path: Path) -> None:
    """Use original content when all edits match exactly."""

    file_path = _write_text(
        tmp_path / "sample.txt",
        "const x = 'exact';\nconst y = \u2018other\u2019;\n",
    )

    await edit.fn(
        path=str(file_path),
        edits=[
            {
                "oldText": "const x = 'exact';",
                "newText": "const x = 'changed';",
            }
        ],
        cwd=Path.cwd(),
    )

    assert (
        file_path.read_text(encoding="utf-8")
        == "const x = 'changed';\nconst y = \u2018other\u2019;\n"
    )


@pytest.mark.asyncio
async def test_edit_fuzzy_matches_multiple_edits(tmp_path: Path) -> None:
    """Use fuzzy-normalized content for all edits once fallback is needed."""

    file_path = _write_text(
        tmp_path / "sample.txt",
        "console.log(\u2018hello\u2019);\nhello\u00a0world\n",
    )

    await edit.fn(
        path=str(file_path),
        edits=[
            {
                "oldText": "console.log('hello');\n",
                "newText": "console.log('world');\n",
            },
            {"oldText": "hello world\n", "newText": "hello universe\n"},
        ],
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == (
        "console.log('world');\nhello universe\n"
    )


def _write_text(path: Path, content: str) -> Path:
    """Write test content to a UTF-8 text file."""

    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)
    return path


def _read_text(path: Path) -> str:
    """Read test content without newline translation."""

    with path.open("r", encoding="utf-8", newline="") as file:
        return file.read()


def _text(result: ToolResult) -> str:
    """Return the single text block from a tool result."""

    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, ToolTextContent)
    return content.text


def _edit_details(result: ToolResult) -> EditDetails:
    """Return edit details from a tool result."""

    assert isinstance(result.details, EditDetails)
    return result.details
