"""Image processing pipeline for tool-returned images."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps

from ai.types.tools import ImageMimeType

EXIF_ORIENTATION_TAG = 274
DEFAULT_MAX_WIDTH = 2000
DEFAULT_MAX_HEIGHT = 2000
DEFAULT_MAX_BASE64_BYTES = int(4.5 * 1024 * 1024)
DEFAULT_JPEG_QUALITY = 80
DOWNSCALE_FACTOR = 0.75
JPEG_QUALITY_STEPS = (DEFAULT_JPEG_QUALITY, 85, 70, 55, 40)


def process_image(
    data: bytes,
    mime_type: ImageMimeType,
    options: ImageProcessingOptions | None = None,
) -> ProcessedImage:
    """Run the image processing pipeline before model submission."""

    effective_options = options or ImageProcessingOptions()
    try:
        image = _create_processed_image(data, mime_type)
        _apply_exif_orientation(image)
        _resize_to_fit_dimensions(image, effective_options)

        while _exceeds_base64_size(image, effective_options):
            if _is_minimum_size(image):
                raise ImageProcessingError
            _shrink_by_factor(image, effective_options)

        return image
    except Exception:
        raise ImageProcessingError from None


@dataclass(frozen=True)
class ImageProcessingOptions:
    """Limits and encoding controls for image processing."""

    max_width: int = DEFAULT_MAX_WIDTH
    max_height: int = DEFAULT_MAX_HEIGHT
    max_base64_bytes: int = DEFAULT_MAX_BASE64_BYTES
    jpeg_quality: int = DEFAULT_JPEG_QUALITY


@dataclass
class ProcessedImage:
    """Image bytes and metadata after local processing."""

    data: bytes
    mime_type: ImageMimeType
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool

    def dimension_note(self) -> str | None:
        """Return a coordinate mapping note when this image was resized."""

        if not self.was_resized:
            return None

        scale = self.original_width / self.width
        return (
            f"[Image: original {self.original_width}x{self.original_height}, "
            f"displayed at {self.width}x{self.height}. Multiply coordinates by "
            f"{scale:.2f} to map to original image.]"
        )


class ImageProcessingError(Exception):
    """Raised when an image cannot be prepared for inline model input."""

    def __init__(self) -> None:
        super().__init__("could not be resized below the inline image size limit")


def _create_processed_image(data: bytes, mime_type: ImageMimeType) -> ProcessedImage:
    """Create a processed image from original file bytes."""

    with _open_image(data) as image:
        width, height = image.size

    return ProcessedImage(
        data=data,
        mime_type=mime_type,
        original_width=width,
        original_height=height,
        width=width,
        height=height,
        was_resized=False,
    )


def _apply_exif_orientation(image: ProcessedImage) -> None:
    """Apply EXIF orientation to an image when present."""

    if image.mime_type != "image/jpeg":
        return

    with _open_image(image.data) as current_image:
        if not _has_exif_orientation(current_image):
            return

        oriented_image = ImageOps.exif_transpose(current_image)
        _replace_with_encoded_image(
            image,
            _encode_image(oriented_image, image.mime_type, DEFAULT_JPEG_QUALITY),
            *oriented_image.size,
            was_resized=False,
        )


def _resize_to_fit_dimensions(
    image: ProcessedImage,
    options: ImageProcessingOptions,
) -> None:
    """Resize once to fit inside the configured max box."""

    width, height = _fit_dimensions(image.width, image.height, options)
    if width == image.width and height == image.height:
        return

    _resize(image, width, height, options)


def _shrink_by_factor(
    image: ProcessedImage,
    options: ImageProcessingOptions,
) -> None:
    """Shrink dimensions by the progressive downscale factor."""

    width, height = _next_dimensions(image.width, image.height)
    if width == image.width and height == image.height:
        raise ImageProcessingError

    _resize(image, width, height, options)


def _resize(
    image: ProcessedImage,
    width: int,
    height: int,
    options: ImageProcessingOptions,
) -> None:
    """Resize and encode an image using ordered candidates."""

    with _open_image(image.data) as current_image:
        resized_image = current_image.resize((width, height), Image.Resampling.LANCZOS)
        encoded = _best_encoded_candidate(resized_image, options)

    _replace_with_encoded_image(image, encoded, width, height, was_resized=True)


def _replace_with_encoded_image(
    image: ProcessedImage,
    candidate: EncodedCandidate,
    width: int,
    height: int,
    *,
    was_resized: bool,
) -> None:
    """Replace processed image bytes and mutable metadata in place."""

    image.data = candidate.data
    image.mime_type = candidate.mime_type
    image.width = width
    image.height = height
    image.was_resized = was_resized


@dataclass(frozen=True)
class EncodedCandidate:
    """One encoded image candidate for the current dimensions."""

    data: bytes
    mime_type: ImageMimeType


def _best_encoded_candidate(
    image: Image.Image,
    options: ImageProcessingOptions,
) -> EncodedCandidate:
    """Return the smallest fitting candidate, or the smallest encoded fallback."""

    candidates = _encoded_candidates(image, options)
    for candidate in candidates:
        if _base64_size(candidate.data) < options.max_base64_bytes:
            return candidate
    return min(candidates, key=lambda candidate: _base64_size(candidate.data))


def _encoded_candidates(
    image: Image.Image,
    options: ImageProcessingOptions,
) -> list[EncodedCandidate]:
    """Return PNG and JPEG encodings for the current dimensions."""

    candidates = [_encode_image(image, "image/png")]
    candidates.extend(
        _encode_image(image, "image/jpeg", quality)
        for quality in _jpeg_quality_steps(options)
    )
    return candidates


def _encode_image(
    image: Image.Image,
    mime_type: ImageMimeType,
    quality: int | None = None,
) -> EncodedCandidate:
    """Encode an image as a specific MIME type."""

    output = BytesIO()
    prepared_image = _prepare_for_encoding(image, mime_type)
    save_options = _save_options(quality)
    prepared_image.save(output, format=_image_format(mime_type), **save_options)
    return EncodedCandidate(data=output.getvalue(), mime_type=mime_type)


def _prepare_for_encoding(
    image: Image.Image,
    mime_type: ImageMimeType,
) -> Image.Image:
    """Return an image mode compatible with the target encoder."""

    if mime_type == "image/jpeg" and image.mode not in ("RGB", "L"):
        return image.convert("RGB")
    return image


def _open_image(data: bytes) -> Image.Image:
    """Open bytes as a Pillow image."""

    return Image.open(BytesIO(data))


def _has_exif_orientation(image: Image.Image) -> bool:
    """Return whether an image has a meaningful EXIF orientation tag."""

    orientation = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
    return isinstance(orientation, int) and orientation != 1


def _exceeds_base64_size(
    image: ProcessedImage,
    options: ImageProcessingOptions,
) -> bool:
    """Return whether an image exceeds the configured base64 payload size."""

    return _base64_size(image.data) >= options.max_base64_bytes


def _is_minimum_size(image: ProcessedImage) -> bool:
    """Return whether an image can no longer be downscaled."""

    return image.width == 1 and image.height == 1


def _fit_dimensions(
    width: int,
    height: int,
    options: ImageProcessingOptions,
) -> tuple[int, int]:
    """Return dimensions fitted inside the configured max box."""

    fitted_width = width
    fitted_height = height

    if fitted_width > options.max_width:
        fitted_height = round((fitted_height * options.max_width) / fitted_width)
        fitted_width = options.max_width
    if fitted_height > options.max_height:
        fitted_width = round((fitted_width * options.max_height) / fitted_height)
        fitted_height = options.max_height

    return max(1, fitted_width), max(1, fitted_height)


def _next_dimensions(width: int, height: int) -> tuple[int, int]:
    """Return the next progressively smaller image dimensions."""

    next_width = 1 if width == 1 else max(1, int(width * DOWNSCALE_FACTOR))
    next_height = 1 if height == 1 else max(1, int(height * DOWNSCALE_FACTOR))
    if next_width == width and next_height == height:
        return 1, 1
    return next_width, next_height


def _jpeg_quality_steps(options: ImageProcessingOptions) -> tuple[int, ...]:
    """Return unique JPEG quality attempts in preferred order."""

    return tuple(dict.fromkeys((options.jpeg_quality, *JPEG_QUALITY_STEPS)))


def _save_options(quality: int | None) -> dict[str, int]:
    """Return Pillow save options for an encoded candidate."""

    if quality is None:
        return {}
    return {"quality": quality}


def _base64_size(data: bytes) -> int:
    """Return the byte size of the base64 representation."""

    return len(base64.b64encode(data))


def _image_format(mime_type: ImageMimeType) -> str:
    """Return the Pillow save format for a MIME type."""

    match mime_type:
        case "image/jpeg":
            return "JPEG"
        case "image/png":
            return "PNG"
        case "image/gif":
            return "GIF"
        case "image/webp":
            return "WEBP"
