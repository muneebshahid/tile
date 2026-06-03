"""Text and image file read tool for the default agent."""

import base64
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel

from ai.types.tools import ImageMimeType, ToolDefinition, ToolImageContent, ToolResult
from agent.tools.image_processing import (
    ImageProcessingError,
    ProcessedImage,
    process_image,
)
from agent.tools.paths import resolve_to_cwd
from agent.tools.truncation import (
    OUTPUT_BYTE_LIMIT,
    OUTPUT_BYTE_LIMIT_LABEL,
    Truncation,
    format_size,
    truncate_head,
)

UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
NARROW_NO_BREAK_SPACE = "\u202f"
IMAGE_TYPE_SNIFF_BYTES = 4100
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ReadSelection(BaseModel):
    """Selected file content and line metadata for formatting."""

    content: str
    start_line: int
    total_lines: int
    user_limited_lines: int | None


async def fn(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
    *,
    cwd: Path,
) -> ToolResult:
    """Read a UTF-8 text file or supported image file."""

    resolved_path = _resolve_path(path, cwd)
    image_mime_type = _detect_supported_image_mime_type(resolved_path)
    if image_mime_type is not None:
        return _read_image(resolved_path, image_mime_type)

    content = _execute(resolved_path)
    selection = _parse_output(content, offset, limit)
    return ToolResult.text(_format_results(selection, path))


def _execute(path: Path) -> str:
    """Read a UTF-8 text file from disk."""

    return path.read_text(encoding="utf-8")


def _parse_output(
    content: str,
    offset: int | None,
    limit: int | None,
) -> ReadSelection:
    """Select the requested line window from file content."""

    lines = content.split("\n")
    start_index = _start_index(offset)
    if start_index >= len(lines):
        raise RuntimeError(
            f"Offset {offset} is beyond end of file ({len(lines)} lines total)"
        )

    selected_lines = _select_lines(lines, start_index, limit)
    return ReadSelection(
        content="\n".join(selected_lines),
        start_line=start_index + 1,
        total_lines=len(lines),
        user_limited_lines=len(selected_lines) if limit is not None else None,
    )


def _format_results(selection: ReadSelection, path: str) -> str:
    """Format selected file content with Pi-compatible continuation notices."""

    truncation = truncate_head(selection.content)
    if truncation.edge_line_exceeds_limit:
        return _format_first_line_too_large(selection, path)
    if truncation.truncated:
        return _format_truncated_selection(selection, truncation)
    if _user_limit_left_remaining_lines(selection):
        return _format_user_limited_selection(selection, truncation.content)
    return truncation.content


def _resolve_path(path: str, cwd: Path) -> Path:
    """Resolve a path with Pi-compatible user-input path variants."""

    normalized_path = _normalize_unicode_spaces(_normalize_at_prefix(path))
    resolved = _resolve_to_cwd(normalized_path, cwd)
    return _existing_path_variant(resolved)


def _read_image(path: Path, mime_type: ImageMimeType) -> ToolResult:
    """Read an image file and return base64 image content."""

    try:
        processed_image = process_image(path.read_bytes(), mime_type)
    except ImageProcessingError as error:
        return ToolResult.text(f'<file name="{path}">[Image omitted: {error}.]</file>')

    return _format_image_result(path, processed_image)


def _format_image_result(path: Path, image: ProcessedImage) -> ToolResult:
    """Format a processed image as text metadata plus image content."""

    encoded_image = base64.b64encode(image.data).decode("ascii")
    text = _format_image_text(path, image)
    return ToolResult.image(
        text,
        ToolImageContent(data=encoded_image, mime_type=image.mime_type),
    )


def _format_image_text(path: Path, image: ProcessedImage) -> str:
    """Format image metadata for the model-visible text block."""

    content = f"[{image.mime_type}]"
    dimension_note = image.dimension_note()
    if dimension_note is not None:
        content = f"{content}\n{dimension_note}"
    return f'<file name="{path}">{content}</file>'


def _detect_supported_image_mime_type(path: Path) -> ImageMimeType | None:
    """Detect supported image MIME type by sniffing the file header."""

    with path.open("rb") as file:
        header = file.read(IMAGE_TYPE_SNIFF_BYTES)
    return _detect_supported_image_mime_type_from_bytes(header)


def _detect_supported_image_mime_type_from_bytes(
    content: bytes,
) -> ImageMimeType | None:
    """Detect supported image MIME type from leading file bytes."""

    if _is_jpeg(content):
        return "image/jpeg"
    if _is_supported_png(content):
        return "image/png"
    if _is_gif(content):
        return "image/gif"
    if _is_webp(content):
        return "image/webp"
    return None


def _is_jpeg(content: bytes) -> bool:
    """Return whether bytes look like a supported JPEG image."""

    return len(content) >= 4 and content[:3] == b"\xff\xd8\xff" and content[3] != 0xF7


def _is_supported_png(content: bytes) -> bool:
    """Return whether bytes look like a non-animated PNG image."""

    return _is_png(content) and not _is_animated_png(content)


def _is_png(content: bytes) -> bool:
    """Return whether bytes look like a PNG image."""

    return (
        len(content) >= 16
        and content.startswith(PNG_SIGNATURE)
        and content[12:16] == b"IHDR"
    )


def _is_animated_png(content: bytes) -> bool:
    """Return whether PNG bytes contain an animation control chunk before image data."""

    animation_chunk_index = content.find(b"acTL")
    image_data_index = content.find(b"IDAT")
    return animation_chunk_index != -1 and (
        image_data_index == -1 or animation_chunk_index < image_data_index
    )


def _is_gif(content: bytes) -> bool:
    """Return whether bytes look like a GIF image."""

    return content.startswith(b"GIF")


def _is_webp(content: bytes) -> bool:
    """Return whether bytes look like a WEBP image."""

    return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP"


def _normalize_at_prefix(path: str) -> str:
    """Strip a leading at sign used when users paste referenced paths."""

    if path.startswith("@"):
        return path[1:]
    return path


def _normalize_unicode_spaces(path: str) -> str:
    """Normalize uncommon Unicode spaces to ordinary spaces."""

    return UNICODE_SPACES.sub(" ", path)


def _resolve_to_cwd(path: str, cwd: Path) -> Path:
    """Resolve relative paths against the current working directory."""

    return resolve_to_cwd(path, cwd)


def _existing_path_variant(path: Path) -> Path:
    """Return the first existing forgiving path variant."""

    for candidate in _path_variants(path):
        if candidate.exists():
            return candidate
    return path


def _path_variants(path: Path) -> list[Path]:
    """Return Pi-compatible path spelling variants to try."""

    macos_screenshot_path = _macos_screenshot_path(path)
    nfd_path = _nfd_path(path)
    curly_quote_path = _curly_quote_path(path)
    nfd_curly_quote_path = _curly_quote_path(nfd_path)
    return [
        path,
        macos_screenshot_path,
        nfd_path,
        curly_quote_path,
        nfd_curly_quote_path,
    ]


def _macos_screenshot_path(path: Path) -> Path:
    """Return a macOS screenshot AM/PM spacing variant."""

    return Path(
        re.sub(
            r" (AM|PM)\.",
            lambda match: f"{NARROW_NO_BREAK_SPACE}{match.group(1)}.",
            str(path),
            flags=re.IGNORECASE,
        )
    )


def _nfd_path(path: Path) -> Path:
    """Return an NFD-normalized path variant."""

    return Path(unicodedata.normalize("NFD", str(path)))


def _curly_quote_path(path: Path) -> Path:
    """Return a path variant using right single quotation marks."""

    return Path(str(path).replace("'", "\u2019"))


def _start_index(offset: int | None) -> int:
    """Convert a 1-indexed offset to a non-negative list index."""

    if offset is None:
        return 0
    return max(0, offset - 1)


def _select_lines(
    lines: list[str],
    start_index: int,
    limit: int | None,
) -> list[str]:
    """Return the requested line slice."""

    if limit is None:
        return lines[start_index:]
    end_index = min(start_index + limit, len(lines))
    return lines[start_index:end_index]


def _format_first_line_too_large(selection: ReadSelection, path: str) -> str:
    """Return guidance when the first selected line exceeds the byte limit."""

    first_line = selection.content.split("\n", maxsplit=1)[0]
    first_line_size = format_size(len(first_line.encode("utf-8")))
    return (
        f"[Line {selection.start_line} is {first_line_size}, exceeds "
        f"{OUTPUT_BYTE_LIMIT_LABEL} limit. Use bash: "
        f"sed -n '{selection.start_line}p' {path} | head -c {OUTPUT_BYTE_LIMIT}]"
    )


def _format_truncated_selection(
    selection: ReadSelection,
    truncation: Truncation,
) -> str:
    """Return truncated content with an offset continuation notice."""

    end_line = selection.start_line + truncation.output_lines - 1
    next_offset = end_line + 1
    notice = _truncation_notice(selection, truncation, end_line, next_offset)
    return f"{truncation.content}\n\n[{notice}]"


def _format_user_limited_selection(selection: ReadSelection, content: str) -> str:
    """Return user-limited content with an offset continuation notice."""

    limited_lines = selection.user_limited_lines or 0
    remaining = selection.total_lines - (
        _start_index_from_selection(selection) + limited_lines
    )
    next_offset = selection.start_line + limited_lines
    return f"{content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"


def _user_limit_left_remaining_lines(selection: ReadSelection) -> bool:
    """Return whether the caller's limit stopped before end of file."""

    if selection.user_limited_lines is None:
        return False
    return (
        _start_index_from_selection(selection) + selection.user_limited_lines
        < selection.total_lines
    )


def _truncation_notice(
    selection: ReadSelection,
    truncation: Truncation,
    end_line: int,
    next_offset: int,
) -> str:
    """Build the continuation notice for automatic truncation."""

    if truncation.truncated_by == "lines":
        return (
            f"Showing lines {selection.start_line}-{end_line} of "
            f"{selection.total_lines}. Use offset={next_offset} to continue."
        )
    return (
        f"Showing lines {selection.start_line}-{end_line} of {selection.total_lines} "
        f"({OUTPUT_BYTE_LIMIT_LABEL} limit). Use offset={next_offset} to continue."
    )


def _start_index_from_selection(selection: ReadSelection) -> int:
    """Return the zero-based start index for a selection."""

    return selection.start_line - 1


tool = ToolDefinition(
    name="read",
    description=(
        "Read the contents of a UTF-8 text file or supported image file. Text "
        "output is truncated to 2000 lines or 50KB. Use offset and limit for "
        "large text files."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read, relative or absolute.",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from, 1-indexed.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    fn=fn,
)
