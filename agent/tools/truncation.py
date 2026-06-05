"""Shared truncation helpers for built-in tool output."""

from collections.abc import Iterator

from tools.types import Truncation, TruncationKeep, TruncationReason

# Maximum lines to keep before appending a truncation notice.
OUTPUT_LINE_LIMIT: int = 2000
# Maximum UTF-8 bytes to keep before appending a truncation notice.
OUTPUT_BYTE_LIMIT: int = 50 * 1024
# Human-readable label for the shared output byte limit.
OUTPUT_BYTE_LIMIT_LABEL: str = "50.0KB"
# Maximum characters to keep from one grep result text line.
GREP_LINE_CHARACTER_LIMIT: int = 500


def truncate_head(
    text: str,
    max_lines: int = OUTPUT_LINE_LIMIT,
    max_bytes: int = OUTPUT_BYTE_LIMIT,
) -> Truncation:
    """Return leading complete lines constrained by line and byte limits."""

    return truncate_text(text, keep="head", max_lines=max_lines, max_bytes=max_bytes)


def truncate_tail(
    text: str,
    max_lines: int = OUTPUT_LINE_LIMIT,
    max_bytes: int = OUTPUT_BYTE_LIMIT,
) -> Truncation:
    """Return trailing complete lines constrained by line and byte limits."""

    return truncate_text(text, keep="tail", max_lines=max_lines, max_bytes=max_bytes)


def truncate_text(
    text: str,
    *,
    keep: TruncationKeep,
    max_lines: int = OUTPUT_LINE_LIMIT,
    max_bytes: int = OUTPUT_BYTE_LIMIT,
) -> Truncation:
    """Return one edge of text constrained by line and byte limits."""

    total_bytes = len(text.encode("utf-8"))
    lines = text.split("\n")
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return Truncation(
            content=text,
            truncated=False,
            truncated_by=None,
            keep=keep,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            edge_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    edge_line = lines[0] if keep == "head" else lines[-1]
    if len(edge_line.encode("utf-8")) > max_bytes:
        return Truncation(
            content="",
            truncated=True,
            truncated_by="bytes",
            keep=keep,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            edge_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines, truncated_by = _select_output_lines(
        lines,
        keep,
        max_lines,
        max_bytes,
    )
    content = "\n".join(output_lines)
    return Truncation(
        content=content,
        truncated=True,
        truncated_by=truncated_by,
        keep=keep,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=len(content.encode("utf-8")),
        edge_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_to_byte_limit(
    text: str,
    byte_limit: int = OUTPUT_BYTE_LIMIT,
) -> tuple[str, bool]:
    """Return text capped to complete lines within the UTF-8 byte limit."""

    if len(text.encode("utf-8")) <= byte_limit:
        return text, False

    output_lines: list[str] = []
    output_bytes = 0
    for line in text.split("\n"):
        separator_bytes = 1 if output_lines else 0
        line_bytes = len(line.encode("utf-8"))
        if output_bytes + separator_bytes + line_bytes > byte_limit:
            break

        output_lines.append(line)
        output_bytes += separator_bytes + line_bytes

    return "\n".join(output_lines), True


def truncate_line(
    line: str,
    character_limit: int = GREP_LINE_CHARACTER_LIMIT,
) -> tuple[str, bool]:
    """Return one line capped to a character limit with a truncation marker."""

    if len(line) <= character_limit:
        return line, False
    return f"{line[:character_limit]}... [truncated]", True


def format_size(byte_count: int) -> str:
    """Format a byte count with compact binary units."""

    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / (1024 * 1024):.1f}MB"


def _select_output_lines(
    lines: list[str],
    keep: TruncationKeep,
    max_lines: int,
    max_bytes: int,
) -> tuple[list[str], TruncationReason]:
    """Return selected output lines and the boundary that stopped selection."""

    output_lines: list[str] = []
    output_bytes = 0
    truncated_by: TruncationReason = "lines"
    for line in _iter_lines(lines, keep):
        if len(output_lines) >= max_lines:
            break

        separator_bytes = 1 if output_lines else 0
        line_bytes = len(line.encode("utf-8"))
        if output_bytes + separator_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break

        output_lines.append(line)
        output_bytes += separator_bytes + line_bytes

    if keep == "tail":
        output_lines.reverse()
    return output_lines, truncated_by


def _iter_lines(lines: list[str], keep: TruncationKeep) -> Iterator[str]:
    """Iterate lines from the retained edge."""

    if keep == "head":
        return iter(lines)
    return reversed(lines)
