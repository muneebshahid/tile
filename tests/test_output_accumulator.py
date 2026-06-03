"""Tests for streaming tool output accumulation."""

import pytest

from agent.tools.output_accumulator import OutputAccumulator
from agent.tools.truncation import truncate_tail


def test_accumulate_decodes_split_utf8_characters() -> None:
    """Decode partial UTF-8 characters across chunk boundaries."""

    output = OutputAccumulator()
    content = "é\nok".encode("utf-8")

    output.accumulate(content[:1])
    output.accumulate(content[1:])
    snapshot = output.finish()

    assert snapshot.content == "é\nok"
    assert snapshot.truncation.truncated is False


def test_accumulate_keeps_bounded_tail_with_global_totals() -> None:
    """Keep only rolling tail text while preserving full output totals."""

    text = "one\ntwo\nthree"
    output = OutputAccumulator(
        max_lines=100,
        max_bytes=6,
    )

    output.accumulate(text.encode("utf-8"))
    snapshot = output.finish()

    assert snapshot.content == "three"
    assert snapshot.truncation.truncated is True
    assert snapshot.truncation.truncated_by == "bytes"
    assert snapshot.truncation.total_lines == 3
    assert snapshot.truncation.total_bytes == len(text.encode("utf-8"))


def test_accumulate_trims_combined_tail_across_chunks() -> None:
    """Trim after appending the incoming chunk to the existing tail."""

    output = OutputAccumulator(
        max_lines=100,
        max_bytes=6,
    )

    output.accumulate("one\n".encode("utf-8"))
    output.accumulate("two\n".encode("utf-8"))
    output.accumulate("three".encode("utf-8"))
    snapshot = output.finish()

    assert snapshot.content == "three"
    assert snapshot.truncation.total_lines == 3


def test_accumulate_reports_global_line_truncation_when_snapshot_fits() -> None:
    """Report line truncation when rolling trim already dropped earlier lines."""

    output = OutputAccumulator(
        max_lines=2,
        max_bytes=10,
    )

    output.accumulate(b"aaaaaaaaaaaaaaaaaaaaaaaa\nx\ny")
    snapshot = output.finish()
    tail_truncation = truncate_tail(output._snapshot_text(), max_lines=2, max_bytes=10)

    assert snapshot.content == "x\ny"
    assert tail_truncation.truncated is False
    assert tail_truncation.truncated_by is None
    assert snapshot.truncation.truncated is True
    assert snapshot.truncation.truncated_by == "lines"
    assert snapshot.truncation.total_lines == 3
    assert snapshot.truncation.output_lines == 2


def test_accumulate_reports_global_byte_truncation_when_snapshot_fits() -> None:
    """Report byte truncation when line-boundary cleanup leaves a small snapshot."""

    output = OutputAccumulator(
        max_lines=100,
        max_bytes=10,
    )

    output.accumulate(b"aaaaaaaaaaaaaaaaaaaaaaaa\nok")
    snapshot = output.finish()
    tail_truncation = truncate_tail(
        output._snapshot_text(), max_lines=100, max_bytes=10
    )

    assert snapshot.content == "ok"
    assert tail_truncation.truncated is False
    assert tail_truncation.truncated_by is None
    assert snapshot.truncation.truncated is True
    assert snapshot.truncation.truncated_by == "bytes"
    assert snapshot.truncation.total_bytes == 27
    assert snapshot.truncation.output_bytes == 2


def test_accumulate_rejects_chunks_after_finish() -> None:
    """Reject writes after the accumulator has been finalized."""

    output = OutputAccumulator()
    output.finish()

    with pytest.raises(RuntimeError, match="after finish"):
        output.accumulate(b"late")
