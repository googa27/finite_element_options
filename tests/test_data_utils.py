"""Tests for data utility helpers."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from finite_element_options.data_utils import (
    df_from_csv,
    df_from_parquet,
    df_to_csv,
    df_to_parquet,
    make_market_dataframe,
    snapshot,
    snapshot_from_netcdf,
    snapshot_to_netcdf,
)


def _metadata() -> dict[str, str]:
    return {
        "run_id": "run-001",
        "units": "CLP",
        "currency": "CLP",
        "quote_type": "option_price",
        "source": "public-synthetic",
        "model": "black_scholes",
        "config_hash": "cfg-abc",
        "backend": "scikit-fem",
        "convergence_state": "converged",
    }


def _manifest(path):
    return json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())


def test_market_dataframe_validation_and_metadata() -> None:
    df = make_market_dataframe([90, 100], [0.5, 1.0], [1.1, 2.2], metadata=_metadata())

    assert list(df.columns) == ["strike", "maturity", "price"]
    assert df.attrs["schema_version"] == "finite-element-options.market-data.v1"
    assert df.attrs["run_id"] == "run-001"
    assert df.attrs["coordinates"] == ["strike", "maturity"]

    with pytest.raises(ValueError, match="same length"):
        make_market_dataframe([90], [0.5, 1.0], [1.1], metadata=_metadata())
    with pytest.raises(ValueError, match="finite"):
        make_market_dataframe([90], [0.5], [np.nan], metadata=_metadata())
    with pytest.raises(ValueError, match="duplicate"):
        make_market_dataframe([90, 90], [0.5, 0.5], [1.1, 1.2], metadata=_metadata())


def test_dataframe_roundtrip_writes_hash_linked_manifest(tmp_path) -> None:
    df = make_market_dataframe([90, 100], [0.5, 1.0], [1.1, 2.2], metadata=_metadata())
    csv_path = tmp_path / "market.csv"
    pq_path = tmp_path / "market.parquet"

    csv_manifest = df_to_csv(df, csv_path)
    parquet_manifest = df_to_parquet(df, pq_path)

    assert (
        csv_manifest["data_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    )
    assert (
        parquet_manifest["data_sha256"]
        == hashlib.sha256(pq_path.read_bytes()).hexdigest()
    )
    assert _manifest(csv_path)["run_id"] == "run-001"
    assert _manifest(pq_path)["metadata"]["quote_type"] == "option_price"
    embedded = pq.read_schema(pq_path).metadata[b"finite_element_options.metadata"]
    embedded_metadata = json.loads(embedded.decode("utf-8"))
    assert (
        embedded_metadata["schema_version"] == "finite-element-options.market-data.v1"
    )
    assert embedded_metadata["run_id"] == "run-001"
    assert embedded_metadata["units"] == "CLP"

    csv_df = df_from_csv(csv_path)
    parquet_df = df_from_parquet(pq_path)
    assert df.equals(csv_df)
    assert df.equals(parquet_df)
    assert csv_df.attrs["run_id"] == "run-001"
    assert parquet_df.attrs["backend"] == "scikit-fem"


def test_dataframe_load_rejects_missing_or_tampered_manifest(tmp_path) -> None:
    df = make_market_dataframe([90], [0.5], [1.1], metadata=_metadata())
    bare_path = tmp_path / "bare.csv"
    df.to_csv(bare_path, index=False)

    with pytest.raises(FileNotFoundError, match="manifest"):
        df_from_csv(bare_path)

    good_path = tmp_path / "market.csv"
    df_to_csv(df, good_path)
    good_path.write_text(good_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        df_from_csv(good_path)


def test_atomic_write_failure_leaves_no_partial_dataframe_files(
    tmp_path, monkeypatch
) -> None:
    df = make_market_dataframe([90], [0.5], [1.1], metadata=_metadata())
    path = tmp_path / "market.csv"

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("late writer failure")

    monkeypatch.setattr(type(df), "to_csv", boom)

    with pytest.raises(RuntimeError, match="late writer failure"):
        df_to_csv(df, path)

    assert not path.exists()
    assert not path.with_suffix(path.suffix + ".manifest.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_concurrent_style_writers_cannot_overwrite_existing_artifact(tmp_path) -> None:
    first = make_market_dataframe([90], [0.5], [1.1], metadata=_metadata())
    second = make_market_dataframe(
        [100], [1.0], [2.2], metadata={**_metadata(), "run_id": "run-002"}
    )
    path = tmp_path / "market.parquet"

    df_to_parquet(first, path)
    with pytest.raises(FileExistsError, match="already exists"):
        df_to_parquet(second, path)

    loaded = df_from_parquet(path)
    assert loaded.attrs["run_id"] == "run-001"
    assert loaded.iloc[0]["strike"] == 90


def test_snapshot_validation_and_metadata() -> None:
    arr = np.arange(6, dtype=float).reshape(2, 3)

    da = snapshot(
        arr,
        [0, 1],
        [90, 100, 110],
        time_name="tau",
        space_name="spot",
        metadata=_metadata(),
    )

    assert da.dims == ("tau", "spot")
    assert np.all(da.sel(tau=0).values == arr[0])
    assert da.attrs["schema_version"] == "finite-element-options.solution-snapshot.v1"
    assert da.attrs["coordinates"] == ["tau", "spot"]
    assert da.attrs["run_id"] == "run-001"
    assert da.attrs["convergence_state"] == "converged"

    with pytest.raises(ValueError, match="shape"):
        snapshot(arr, [0], [90, 100, 110], metadata=_metadata())
    with pytest.raises(ValueError, match="finite"):
        snapshot(np.array([[np.nan]]), [0], [90], metadata=_metadata())


def test_snapshot_netcdf_roundtrip_and_hash_validation(tmp_path) -> None:
    da = snapshot(
        np.arange(6, dtype=float).reshape(2, 3),
        [0, 1],
        [90, 100, 110],
        time_name="tau",
        space_name="spot",
        metadata=_metadata(),
    )
    path = tmp_path / "solution.nc"

    manifest = snapshot_to_netcdf(da, path)

    assert manifest["data_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = snapshot_from_netcdf(path)
    assert loaded.dims == ("tau", "spot")
    assert loaded.attrs["run_id"] == "run-001"
    assert loaded.attrs["backend"] == "scikit-fem"
    assert np.allclose(loaded.values, da.values)

    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        snapshot_from_netcdf(path)
