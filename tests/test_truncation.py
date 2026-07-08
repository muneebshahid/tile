"""Tests for shared tool output truncation helpers."""

import tile.tools.support.truncation as truncation


def test_append_notice_block_returns_text_without_notices() -> None:
    """Return unchanged text when there are no notices to append."""

    assert truncation.append_notice_block("content", []) == "content"


def test_append_notice_block_formats_joined_notices() -> None:
    """Append notices in the shared bracketed block format."""

    assert truncation.append_notice_block("content", ["first", "second"]) == (
        "content\n\n[first. second]"
    )


def test_truncate_to_byte_limit_keeps_complete_lines() -> None:
    """Truncate over-limit output at line boundaries instead of mid-line."""

    assert truncation.truncate_to_byte_limit("a.txt\nb.txt", byte_limit=11) == (
        "a.txt\nb.txt",
        False,
    )
    assert truncation.truncate_to_byte_limit("a.txt\nb.txt", byte_limit=10) == (
        "a.txt",
        True,
    )


def test_truncate_head_reports_line_limit() -> None:
    """Report line-limit metadata while keeping leading complete lines."""

    result = truncation.truncate_head("a\nb\nc", max_lines=2, max_bytes=100)

    assert result.content == "a\nb"
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.output_lines == 2
    assert result.total_lines == 3


def test_truncate_head_reports_byte_limit() -> None:
    """Report byte-limit metadata while keeping leading complete lines."""

    result = truncation.truncate_head("abcd\nefgh", max_lines=100, max_bytes=6)

    assert result.content == "abcd"
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.output_bytes == 4


def test_truncate_head_reports_first_line_exceeds_limit() -> None:
    """Return no partial content when the first line exceeds the byte limit."""

    result = truncation.truncate_head("abcdef\nsecond", max_lines=100, max_bytes=5)

    assert result.content == ""
    assert result.truncated is True
    assert result.edge_line_exceeds_limit is True
    assert result.keep == "head"


def test_truncate_tail_reports_line_limit() -> None:
    """Report line-limit metadata while keeping trailing complete lines."""

    result = truncation.truncate_tail("a\nb\nc", max_lines=2, max_bytes=100)

    assert result.content == "b\nc"
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.output_lines == 2
    assert result.total_lines == 3


def test_truncate_tail_reports_byte_limit() -> None:
    """Report byte-limit metadata while keeping trailing complete lines."""

    result = truncation.truncate_tail("abcd\nefgh", max_lines=100, max_bytes=6)

    assert result.content == "efgh"
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.output_bytes == 4


def test_truncate_tail_reports_last_line_exceeds_limit() -> None:
    """Return no partial content when the final line exceeds the byte limit."""

    result = truncation.truncate_tail("first\nabcdef", max_lines=100, max_bytes=5)

    assert result.content == ""
    assert result.truncated is True
    assert result.edge_line_exceeds_limit is True
    assert result.keep == "tail"
