from collections.abc import AsyncIterator
from typing import Protocol

from ori.types.stream_events import ProviderStreamEvent


class AsyncEventStream(Protocol):
    """Async stream of provider-originated assistant events."""

    def __aiter__(self) -> AsyncIterator[ProviderStreamEvent]: ...
