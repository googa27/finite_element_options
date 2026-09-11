"""Bounded, exact-key reuse of assembled operators and numerical factors."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

Key = TypeVar("Key")
Value = TypeVar("Value")


@dataclass(frozen=True)
class OperatorCacheInfo:
    """Immutable counts of actual resident entries and lifetime cache events."""

    capacity: int
    entries: int
    peak_entries: int
    hits: int
    misses: int
    evictions: int
    clears: int


class OperatorCache(Generic[Key, Value]):
    """Keep at most ``capacity`` operators, evicting the least recently used.

    Zero disables retention. Keys and numerical payloads belong to the caller;
    the cache neither approximates equality nor copies or mutates operators.
    Lookup raises ``KeyError`` on a miss. This cache is for serial solver use.
    """

    def __init__(self, capacity: int):
        """Validate the explicit entry limit before retaining any operators."""
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("operator cache capacity must be a non-negative integer")
        self._capacity = capacity
        self._entries: OrderedDict[Key, Value] = OrderedDict()
        self._peak_entries = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._clears = 0

    def __getitem__(self, key: Key) -> Value:
        """Return a resident operator and mark it as most recently used."""
        try:
            value = self._entries[key]
        except KeyError:
            self._misses += 1
            raise
        self._entries.move_to_end(key)
        self._hits += 1
        return value

    def __setitem__(self, key: Key, value: Value) -> None:
        """Retain a completed operator, releasing the oldest entry if full."""
        if self._capacity == 0:
            return
        if key not in self._entries and len(self._entries) == self._capacity:
            self._entries.popitem(last=False)
            self._evictions += 1
        self._entries[key] = value
        self._entries.move_to_end(key)
        self._peak_entries = max(self._peak_entries, len(self._entries))

    def clear(self) -> None:
        """Release resident operators while preserving lifetime event counts."""
        self._entries.clear()
        self._clears += 1

    def info(self) -> OperatorCacheInfo:
        """Snapshot resident-entry counts; these do not estimate process memory."""
        return OperatorCacheInfo(
            self._capacity,
            len(self._entries),
            self._peak_entries,
            self._hits,
            self._misses,
            self._evictions,
            self._clears,
        )
