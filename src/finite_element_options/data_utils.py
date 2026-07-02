"""Helpers for market data and solution snapshots.

This module centralises utilities for constructing pandas DataFrames and
xarray DataArrays used throughout the project.  It also provides simple
serialisation helpers enabling reproducible experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------------
# Market data helpers


def make_market_dataframe(
    strikes: Iterable[float],
    maturities: Iterable[float],
    prices: Iterable[float],
) -> pd.DataFrame:
    """Return a canonical market DataFrame.

    Parameters
    ----------
    strikes, maturities, prices:
        One-dimensional sequences describing the option surface.
    """

    return pd.DataFrame({"strike": strikes, "maturity": maturities, "price": prices})


def df_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Serialise ``df`` to ``path`` in CSV format."""

    df.to_csv(path, index=False)


def df_to_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Serialise ``df`` to ``path`` in Parquet format."""

    df.to_parquet(path, index=False)


def df_from_csv(path: str | Path) -> pd.DataFrame:
    """Load market data from ``path`` stored in CSV format."""

    return pd.read_csv(path)


def df_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load market data from ``path`` stored in Parquet format."""

    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Solution snapshot helpers


def snapshot(
    array: np.ndarray,
    time_grid: Iterable[float],
    space_grid: Iterable[float],
) -> xr.DataArray:
    """Convert ``array`` into an :class:`xarray.DataArray` snapshot.

    Parameters
    ----------
    array:
        Two-dimensional ``time`` × ``space`` array of option values.
    time_grid, space_grid:
        Coordinate vectors for the respective axes.
    """

    return xr.DataArray(
        array,
        coords={"time": np.asarray(list(time_grid)), "space": np.asarray(list(space_grid))},
        dims=("time", "space"),
    )
