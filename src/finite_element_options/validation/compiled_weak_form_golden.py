"""Packaged golden fixture data for compiled weak-form screening."""

from __future__ import annotations

import base64
from functools import lru_cache
import json
import zlib
from typing import Any

_GOLDEN_FIXTURE_ZLIB_BASE64 = (
    "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"
)


@lru_cache(maxsize=1)
def packaged_golden_fixture() -> dict[str, Any]:
    """Return the exact public Black-Scholes compiled weak-form golden fixture."""

    raw = zlib.decompress(base64.b64decode(_GOLDEN_FIXTURE_ZLIB_BASE64))
    loaded = json.loads(raw.decode("utf-8"))
    if not isinstance(loaded, dict):  # defensive guard for corrupted package data
        raise TypeError("compiled weak-form golden fixture must decode to an object")
    return loaded
