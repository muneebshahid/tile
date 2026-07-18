from collections.abc import AsyncGenerator
from typing import TypeAlias

from tile.types.stream_events import ProviderStreamEvent

AsyncEventStream: TypeAlias = AsyncGenerator[ProviderStreamEvent, None]
"""Async stream of provider-originated assistant events.

Provider streams are async generators: the consumer that iterates a
stream owns closing it, and closure must release the underlying
transport. Adapters wrapping SDK streams forward closure to the SDK
object in a ``finally``.
"""
