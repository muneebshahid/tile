from collections.abc import AsyncIterator
from typing import Literal, Protocol, TypedDict

from ori.types.stream_events import ProviderStreamEvent


class AsyncEventStream(Protocol):
    """Async stream of provider-originated assistant events."""

    def __aiter__(self) -> AsyncIterator[ProviderStreamEvent]: ...


class Reasoning(TypedDict, total=False):
    """App-level reasoning options passed to model providers."""

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    summary: Literal["auto", "concise", "detailed"]
