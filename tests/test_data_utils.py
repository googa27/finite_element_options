"""Tests for data utility helpers."""

import pandas as pd

from src.data_utils import (
    make_market_dataframe,
    df_to_csv,
    df_from_csv,
    df_to_parquet,
    df_from_parquet,
    snapshot,
)

import numpy as np


def test_dataframe_roundtrip(tmp_path):
    df = make_market_dataframe([1, 2], [0.5, 0.6], [0.1, 0.2])
    csv_path = tmp_path / "market.csv"
    pq_path = tmp_path / "market.parquet"
    df_to_csv(df, csv_path)
    df_to_parquet(df, pq_path)
    assert df.equals(df_from_csv(csv_path))
    assert df.equals(df_from_parquet(pq_path))


def test_snapshot_creation():
    arr = np.arange(6).reshape(2, 3)
    time = [0, 1]
    space = [10, 20, 30]
    da = snapshot(arr, time, space)
    assert da.dims == ("time", "space")
    assert np.all(da[0].values == arr[0])
