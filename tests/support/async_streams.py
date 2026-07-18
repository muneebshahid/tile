"""Helpers for adapting static test data into async streams."""

from collections.abc import AsyncGenerator, Sequence
from typing import TypeVar

TItem = TypeVar("TItem")


def async_stream(
    items: Sequence[TItem],
    *,
    error: Exception | None = None,
) -> AsyncGenerator[TItem, None]:
    """Yield static test items, then raise the optional mid-stream error."""

    async def _iterate() -> AsyncGenerator[TItem, None]:
        """Yield each configured item, then fail when an error is set."""

        for item in items:
            yield item
        if error is not None:
            raise error

    return _iterate()
