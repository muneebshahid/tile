"""File edit tool for the default agent."""

import asyncio
import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ori.types.tools import EditDetails, ToolDefinition, ToolResult
from ori.tools.support.paths import resolve_to_cwd

UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
FUZZY_UNICODE_SPACES = re.compile(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]")
SMART_SINGLE_QUOTES = re.compile(r"[\u2018\u2019\u201A\u201B]")
SMART_DOUBLE_QUOTES = re.compile(r"[\u201C\u201D\u201E\u201F]")
UNICODE_DASHES = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
BOM = "\ufeff"
LineEnding = Literal["\n", "\r\n"]


async def fn(path: str, edits: list[dict[str, str]], *, cwd: Path) -> ToolResult:
    """Edit a file with one or more targeted text replacements."""

    resolved_path = _resolve_path(path, cwd)
    replacements = _parse_edits(edits)
    result = await _execute(resolved_path, replacements, path)
    return _build_result(result, path)


class EditReplacement(BaseModel):
    """A single exact text replacement requested by the model."""

    oldText: str
    newText: str


@dataclass(frozen=True)
class LoadedFile:
    """File content split into mutation metadata and editable text."""

    bom: str
    content: str
    line_ending: LineEnding


@dataclass(frozen=True)
class MatchedSpan[ReplacementT]:
    """A validated replacement span over character or line positions."""

    edit_index: int
    start: int
    end: int
    replacement: ReplacementT


@dataclass(frozen=True)
class EditExecutionResult:
    """Successful edit execution details."""

    path: Path
    replacement_count: int
    diff: str


@dataclass(frozen=True)
class ReplacementApplication:
    """Content before and after applying validated replacements."""

    base_content: str
    new_content: str


class MatchNotFound(RuntimeError):
    """Raised when a replacement oldText cannot be found."""


async def _execute(
    path: Path,
    replacements: list[EditReplacement],
    display_path: str,
) -> EditExecutionResult:
    """Apply validated replacements to a file asynchronously."""

    return await asyncio.to_thread(_edit_file, path, replacements, display_path)


def _parse_edits(edits: list[dict[str, str]]) -> list[EditReplacement]:
    """Parse raw edit dictionaries into replacement models."""

    replacements = [EditReplacement.model_validate(edit) for edit in edits]
    if not replacements:
        raise RuntimeError("edits must contain at least one replacement")
    return replacements


def _build_result(result: EditExecutionResult, path: str) -> ToolResult:
    """Build a successful edit result."""

    text = f"Successfully replaced {result.replacement_count} block(s) in {path}."
    return ToolResult.text(text, details=EditDetails(diff=result.diff))


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a user path against the tool working directory."""

    normalized_path = _normalize_unicode_spaces(_normalize_at_prefix(path))
    return resolve_to_cwd(normalized_path, cwd)


def _edit_file(
    path: Path,
    replacements: list[EditReplacement],
    display_path: str,
) -> EditExecutionResult:
    """Read, edit, and write a UTF-8 text file."""

    loaded_file = _load_file(path)
    application = _apply_replacements_with_fallback(
        loaded_file.content,
        replacements,
        display_path,
    )
    _write_loaded_file(path, loaded_file, application.new_content)
    diff = _generate_unified_diff(
        application.base_content,
        application.new_content,
        display_path,
    )
    return EditExecutionResult(
        path=path,
        replacement_count=len(replacements),
        diff=diff,
    )


def _load_file(path: Path) -> LoadedFile:
    """Read a UTF-8 text file and capture BOM and line-ending metadata."""

    raw_content = _read_text(path)
    bom, content = _strip_bom(raw_content)
    return LoadedFile(
        bom=bom,
        content=_normalize_to_lf(content),
        line_ending=_detect_line_ending(content),
    )


def _apply_replacements_with_fallback(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> ReplacementApplication:
    """Apply exact replacements, then retry fuzzy whole-line matching if needed."""

    _validate_non_empty_old_text(replacements, display_path)
    try:
        return _apply_exact_replacements(content, replacements, display_path)
    except MatchNotFound:
        return _apply_fuzzy_replacements(content, replacements, display_path)


def _apply_exact_replacements(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> ReplacementApplication:
    """Apply replacements matched exactly in character space."""

    spans = _match_exact_spans(content, replacements, display_path)
    new_content = _splice_text_spans(content, spans)
    return _build_application(content, new_content, replacements, display_path)


def _apply_fuzzy_replacements(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> ReplacementApplication:
    """Apply replacements matched as fuzzy-normalized whole lines.

    Normalization is used only to locate matches; replacements splice the
    original lines so untouched regions stay byte-identical.
    """

    original_lines = content.split("\n")
    spans = _match_fuzzy_spans(original_lines, replacements, display_path)
    new_content = "\n".join(_splice_line_spans(original_lines, spans))
    return _build_application(content, new_content, replacements, display_path)


def _build_application(
    content: str,
    new_content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> ReplacementApplication:
    """Reject no-op replacements and pair original with edited content."""

    if content == new_content:
        raise RuntimeError(_no_change_error(display_path, len(replacements)))
    return ReplacementApplication(base_content=content, new_content=new_content)


def _match_exact_spans(
    content: str,
    replacements: list[EditReplacement],
    display_path: str,
) -> list[MatchedSpan[str]]:
    """Find a unique character span for every requested replacement."""

    spans: list[MatchedSpan[str]] = []
    for index, replacement in enumerate(replacements):
        old_text = _normalize_to_lf(replacement.oldText)
        starts = _find_text_starts(content, old_text)
        start = _require_unique_match(starts, display_path, index, len(replacements))
        spans.append(
            MatchedSpan(
                edit_index=index,
                start=start,
                end=start + len(old_text),
                replacement=_normalize_to_lf(replacement.newText),
            )
        )
    _validate_non_overlapping(spans, display_path)
    return spans


def _match_fuzzy_spans(
    original_lines: list[str],
    replacements: list[EditReplacement],
    display_path: str,
) -> list[MatchedSpan[list[str]]]:
    """Find a unique fuzzy-normalized line window for every requested replacement."""

    normalized_lines = [_normalize_for_fuzzy_match(line) for line in original_lines]
    spans: list[MatchedSpan[list[str]]] = []
    for index, replacement in enumerate(replacements):
        window = [
            _normalize_for_fuzzy_match(line)
            for line in _window_lines(replacement.oldText)
        ]
        starts = _find_window_starts(normalized_lines, window)
        start = _require_unique_match(starts, display_path, index, len(replacements))
        spans.append(
            MatchedSpan(
                edit_index=index,
                start=start,
                end=start + len(window),
                replacement=_window_lines(replacement.newText),
            )
        )
    _validate_non_overlapping(spans, display_path)
    return spans


def _find_text_starts(content: str, old_text: str) -> list[int]:
    """Return non-overlapping character offsets where the old text occurs."""

    starts: list[int] = []
    offset = content.find(old_text)
    while offset != -1:
        starts.append(offset)
        offset = content.find(old_text, offset + len(old_text))
    return starts


def _find_window_starts(normalized_lines: list[str], window: list[str]) -> list[int]:
    """Return start indexes where the window matches consecutive lines."""

    span = len(window)
    return [
        start
        for start in range(len(normalized_lines) - span + 1)
        if normalized_lines[start : start + span] == window
    ]


def _window_lines(text: str) -> list[str]:
    """Split edit text into whole lines, treating a trailing newline as closure."""

    lines = _normalize_to_lf(text).split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    return lines


def _require_unique_match(
    starts: list[int],
    display_path: str,
    edit_index: int,
    total_edits: int,
) -> int:
    """Return the single match start or raise a lookup error."""

    if not starts:
        raise MatchNotFound(_not_found_error(display_path, edit_index, total_edits))
    if len(starts) > 1:
        raise RuntimeError(
            _duplicate_error(display_path, edit_index, total_edits, len(starts))
        )
    return starts[0]


def _validate_non_overlapping[ReplacementT](
    spans: list[MatchedSpan[ReplacementT]],
    display_path: str,
) -> None:
    """Reject overlapping replacement spans."""

    ordered = sorted(spans, key=lambda span: span.start)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.end > current.start:
            raise RuntimeError(
                _overlap_error(previous.edit_index, current.edit_index, display_path)
            )


def _splice_text_spans(content: str, spans: list[MatchedSpan[str]]) -> str:
    """Apply character spans in reverse order so earlier offsets stay stable."""

    result = content
    for span in _spans_in_reverse_order(spans):
        result = result[: span.start] + span.replacement + result[span.end :]
    return result


def _splice_line_spans(
    lines: list[str],
    spans: list[MatchedSpan[list[str]]],
) -> list[str]:
    """Apply line spans in reverse order so earlier indexes stay stable."""

    result = list(lines)
    for span in _spans_in_reverse_order(spans):
        result[span.start : span.end] = span.replacement
    return result


def _spans_in_reverse_order[ReplacementT](
    spans: list[MatchedSpan[ReplacementT]],
) -> list[MatchedSpan[ReplacementT]]:
    """Return spans sorted by descending start position."""

    return sorted(spans, key=lambda span: span.start, reverse=True)


def _write_loaded_file(path: Path, loaded_file: LoadedFile, content: str) -> None:
    """Write edited content while preserving BOM and line endings."""

    restored_content = _restore_line_endings(content, loaded_file.line_ending)
    _write_text(path, f"{loaded_file.bom}{restored_content}")


def _generate_unified_diff(old_content: str, new_content: str, path: str) -> str:
    """Generate a standard unified diff for edited LF-normalized content."""

    diff_lines = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff_lines)


def _read_text(path: Path) -> str:
    """Read UTF-8 text without newline translation."""

    with path.open("r", encoding="utf-8", newline="") as file:
        return file.read()


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text without newline translation."""

    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)


def _validate_non_empty_old_text(
    replacements: list[EditReplacement],
    display_path: str,
) -> None:
    """Reject empty oldText values."""

    for index, replacement in enumerate(replacements):
        if replacement.oldText == "":
            raise RuntimeError(
                _empty_old_text_error(display_path, index, len(replacements))
            )


def _strip_bom(content: str) -> tuple[str, str]:
    """Remove and return a leading UTF-8 BOM marker when present."""

    if content.startswith(BOM):
        return BOM, content[1:]
    return "", content


def _detect_line_ending(content: str) -> LineEnding:
    """Detect whether the first line ending is CRLF or LF."""

    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    if crlf_index < lf_index:
        return "\r\n"
    return "\n"


def _normalize_to_lf(content: str) -> str:
    """Normalize CRLF and CR line endings to LF."""

    return content.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(content: str, line_ending: LineEnding) -> str:
    """Restore LF-normalized content to the original line-ending style."""

    if line_ending == "\r\n":
        return content.replace("\n", "\r\n")
    return content


def _normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fallback matching."""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = _strip_trailing_line_whitespace(normalized)
    normalized = SMART_SINGLE_QUOTES.sub("'", normalized)
    normalized = SMART_DOUBLE_QUOTES.sub('"', normalized)
    normalized = UNICODE_DASHES.sub("-", normalized)
    return FUZZY_UNICODE_SPACES.sub(" ", normalized)


def _strip_trailing_line_whitespace(text: str) -> str:
    """Strip trailing whitespace from every line."""

    return "\n".join(line.rstrip() for line in text.split("\n"))


def _normalize_at_prefix(path: str) -> str:
    """Strip a leading at sign used when users paste referenced paths."""

    if path.startswith("@"):
        return path[1:]
    return path


def _normalize_unicode_spaces(path: str) -> str:
    """Normalize uncommon Unicode spaces to ordinary spaces."""

    return UNICODE_SPACES.sub(" ", path)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    """Return the not-found edit error message."""

    if total_edits == 1:
        return (
            f"Could not find the exact text in {path}. The old text must match "
            "exactly including all whitespace and newlines."
        )
    return (
        f"Could not find edits[{edit_index}] in {path}. The oldText must match "
        "exactly including all whitespace and newlines."
    )


def _duplicate_error(
    path: str,
    edit_index: int,
    total_edits: int,
    occurrences: int,
) -> str:
    """Return the duplicate-match edit error message."""

    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. The text "
            "must be unique. Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it "
        "unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    """Return the empty oldText edit error message."""

    if total_edits == 1:
        return f"oldText must not be empty in {path}."
    return f"edits[{edit_index}].oldText must not be empty in {path}."


def _overlap_error(first_index: int, second_index: int, path: str) -> str:
    """Return the overlapping-edits error message."""

    return (
        f"edits[{first_index}] and edits[{second_index}] overlap in {path}. "
        "Merge them into one edit or target disjoint regions."
    )


def _no_change_error(path: str, total_edits: int) -> str:
    """Return the no-change edit error message."""

    if total_edits == 1:
        return f"No changes made to {path}. The replacement produced identical content."
    return f"No changes made to {path}. The replacements produced identical content."


tool = ToolDefinition(
    name="edit",
    description=(
        "Edit a single file using exact text replacement. Every edits[].oldText "
        "must match a unique, non-overlapping region of the original file. If "
        "two changes affect the same block or nearby lines, merge them into one "
        "edit instead of emitting overlapping edits. Do not include large "
        "unchanged regions just to connect distant changes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative or absolute).",
            },
            "edits": {
                "type": "array",
                "description": (
                    "One or more targeted replacements. Each edit is matched "
                    "against the original file, not incrementally. Do not include "
                    "overlapping or nested edits. If two changes touch the same "
                    "block or nearby lines, merge them into one edit instead."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {
                            "type": "string",
                            "description": (
                                "Exact text for one targeted replacement. It "
                                "must be unique in the original file and must "
                                "not overlap with any other edits[].oldText in "
                                "the same call."
                            ),
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text for this targeted edit.",
                        },
                    },
                    "required": ["oldText", "newText"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["path", "edits"],
        "additionalProperties": False,
    },
    fn=fn,
)
