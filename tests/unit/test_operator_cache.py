"""Bounded numerical reuse retains only resident objects and reports actual events."""

import gc
import weakref

import pytest

from finite_element_options.core.operator_cache import OperatorCache


class Payload:
    pass


@pytest.mark.parametrize("capacity", [-1, True, False, 1.5, 2.0, None, "2"])
def test_capacity_is_an_explicit_nonnegative_integer(capacity):
    with pytest.raises(ValueError, match="capacity"):
        OperatorCache(capacity)


def test_lru_evicts_actual_objects_and_counts_only_resident_entries():
    cache = OperatorCache(2)
    first, second, third = Payload(), Payload(), Payload()
    refs = [weakref.ref(value) for value in (first, second, third)]
    cache["first"], cache["second"] = first, second
    assert cache["first"] is first  # second is now least recently used
    cache["third"] = third
    del second
    gc.collect()
    assert refs[1]() is None and refs[0]() is first and refs[2]() is third
    with pytest.raises(KeyError):
        cache["second"]
    info = cache.info()
    assert (info.capacity, info.entries, info.peak_entries) == (2, 2, 2)
    assert (info.hits, info.misses, info.evictions) == (1, 1, 1)
    cache.clear()
    del first, third
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert cache.info().entries == 0 and cache.info().clears == 1
    assert info.entries == 2  # immutable point-in-time diagnostics


def test_zero_capacity_retains_no_payload_and_always_misses():
    cache = OperatorCache(0)
    payload = Payload()
    ref = weakref.ref(payload)
    cache["key"] = payload
    del payload
    gc.collect()
    assert ref() is None
    with pytest.raises(KeyError):
        cache["key"]
    assert cache.info().entries == cache.info().peak_entries == 0
    assert cache.info().misses == 1 and cache.info().evictions == 0
