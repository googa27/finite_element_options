"""Helpers for market data and solution snapshots.

The helpers in this module validate tabular inputs and solution arrays before
writing, publish artifacts via same-directory temporary files, and pair every
artifact with a hash-linked manifest sidecar. The sidecar is the completion
marker used by readers: a data file without a matching manifest is treated as an
incomplete artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import xarray as xr

MARKET_DATA_SCHEMA_VERSION = "finite-element-options.market-data.v1"
SOLUTION_SNAPSHOT_SCHEMA_VERSION = "finite-element-options.solution-snapshot.v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "finite-element-options.artifact-manifest.v1"

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Manifest and atomic write helpers


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_roundtripable(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(mapping), default=_json_default, sort_keys=True))


def _manifest_path(path: str | Path) -> Path:
    artifact = Path(path)
    return artifact.with_suffix(artifact.suffix + ".manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _artifact_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd: int | None = None
    acquired = False
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        acquired = True
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        if fd is not None:
            os.close(fd)
        if acquired:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _missing_io_extra_message() -> str:
    """Return a useful optional-dependency error message for Parquet helpers."""

    return (
        "Parquet artifact helpers require the 'io' extra: "
        "install finite-element-options[io] to enable pyarrow-backed reads/writes."
    )


def _restore_publish_backup(backup_path: Path, final_path: Path) -> None:
    if backup_path.exists():
        os.replace(backup_path, final_path)


def _artifact_manifest(
    *,
    path: Path,
    artifact_type: str,
    metadata: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "data_path": path.name,
        "data_sha256": _sha256(path),
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": metadata.get("run_id", "unknown-run"),
        "metadata": _json_roundtripable(metadata),
    }
    if extra:
        payload.update(_json_roundtripable(extra))
    return payload


def _atomic_publish(
    path: str | Path,
    *,
    artifact_type: str,
    metadata: Mapping[str, Any],
    writer: Callable[[Path], None],
    overwrite: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(final_path)
    tmp_data = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")
    tmp_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    backup_data = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.bak")
    backup_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.bak"
    )

    with _artifact_lock(final_path):
        if not overwrite and (final_path.exists() or manifest_path.exists()):
            raise FileExistsError(f"artifact already exists: {final_path}")
        data_backed_up = False
        manifest_backed_up = False
        data_published = False
        try:
            writer(tmp_data)
            manifest = _artifact_manifest(
                path=tmp_data,
                artifact_type=artifact_type,
                metadata=metadata,
                extra=extra,
            )
            manifest["data_path"] = final_path.name
            manifest["data_sha256"] = _sha256(tmp_data)
            tmp_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, default=_json_default)
                + "\n",
                encoding="utf-8",
            )
            if final_path.exists():
                os.replace(final_path, backup_data)
                data_backed_up = True
            if manifest_path.exists():
                os.replace(manifest_path, backup_manifest)
                manifest_backed_up = True
            os.replace(tmp_data, final_path)
            data_published = True
            os.replace(tmp_manifest, manifest_path)
            backup_data.unlink(missing_ok=True)
            backup_manifest.unlink(missing_ok=True)
            return manifest
        except Exception:
            if data_published:
                final_path.unlink(missing_ok=True)
            if not manifest_backed_up:
                manifest_path.unlink(missing_ok=True)
            _restore_publish_backup(backup_data, final_path)
            _restore_publish_backup(backup_manifest, manifest_path)
            raise
        finally:
            tmp_data.unlink(missing_ok=True)
            tmp_manifest.unlink(missing_ok=True)
            if not data_backed_up:
                backup_data.unlink(missing_ok=True)
            if not manifest_backed_up:
                backup_manifest.unlink(missing_ok=True)


def _load_manifest(path: str | Path, expected_artifact_type: str) -> dict[str, Any]:
    final_path = Path(path)
    manifest_path = _manifest_path(final_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest sidecar missing for artifact: {final_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest schema_version is incompatible")
    if manifest.get("artifact_type") != expected_artifact_type:
        raise ValueError(
            f"manifest artifact_type {manifest.get('artifact_type')!r} does not match "
            f"{expected_artifact_type!r}"
        )
    if manifest.get("data_path") != final_path.name:
        raise ValueError("manifest data_path does not match artifact filename")
    actual_hash = _sha256(final_path)
    if manifest.get("data_sha256") != actual_hash:
        raise ValueError("artifact hash does not match manifest")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manifest metadata must be an object")
    return manifest


def _write_parquet_with_metadata(
    df: pd.DataFrame, path: Path, metadata: Mapping[str, Any]
) -> None:
    """Write a Parquet table with finite-element-options metadata in the schema."""

    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - covered via import monkeypatch.
        raise ImportError(_missing_io_extra_message()) from exc

    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = table.schema.metadata or {}
    schema_metadata = {
        **existing,
        b"finite_element_options.metadata": json.dumps(
            _json_roundtripable(metadata), sort_keys=True, default=_json_default
        ).encode("utf-8"),
    }
    pq.write_table(table.replace_schema_metadata(schema_metadata), path)


def _read_parquet_metadata(path: str | Path) -> dict[str, Any]:
    """Read finite-element-options metadata embedded in a Parquet schema."""

    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - covered via import monkeypatch.
        raise ImportError(_missing_io_extra_message()) from exc

    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(b"finite_element_options.metadata")
    if raw is None:
        raise ValueError("Parquet artifact missing finite_element_options metadata")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Parquet metadata payload must be an object")
    return payload


# ---------------------------------------------------------------------------
# Market data helpers


def _base_market_metadata(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    supplied = dict(metadata or {})
    base = {
        "schema_version": MARKET_DATA_SCHEMA_VERSION,
        "run_id": supplied.get("run_id", f"run-{uuid.uuid4().hex}"),
        "units": supplied.get("units", "unspecified"),
        "currency": supplied.get("currency", "unspecified"),
        "quote_type": supplied.get("quote_type", "option_price"),
        "source": supplied.get("source", "unspecified"),
        "coordinates": ["strike", "maturity"],
    }
    base.update(supplied)
    base["schema_version"] = MARKET_DATA_SCHEMA_VERSION
    base["coordinates"] = ["strike", "maturity"]
    return _json_roundtripable(base)


def _validate_market_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = ["strike", "maturity", "price"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"market data missing required columns: {missing}")
    clean = df.loc[:, required].astype(float)
    lengths = {len(clean[column]) for column in required}
    if len(lengths) != 1:
        raise ValueError("strike, maturity and price must have the same length")
    values = clean.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("market data values must be finite")
    if (clean["strike"] <= 0).any():
        raise ValueError("strike values must be positive")
    if (clean["maturity"] < 0).any():
        raise ValueError("maturity values must be non-negative")
    if clean.duplicated(subset=["strike", "maturity"]).any():
        raise ValueError("market data contains duplicate strike/maturity keys")
    return clean


def make_market_dataframe(
    strikes: Iterable[float],
    maturities: Iterable[float],
    prices: Iterable[float],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a validated canonical market DataFrame.

    Parameters
    ----------
    strikes, maturities, prices:
        One-dimensional sequences describing the option surface.
    metadata:
        Optional run/source/units metadata stored in ``DataFrame.attrs`` and in
        artifact manifests when the table is written.
    """

    strike_arr = np.asarray(list(strikes), dtype=float)
    maturity_arr = np.asarray(list(maturities), dtype=float)
    price_arr = np.asarray(list(prices), dtype=float)
    if not (len(strike_arr) == len(maturity_arr) == len(price_arr)):
        raise ValueError("strike, maturity and price must have the same length")
    df = pd.DataFrame(
        {"strike": strike_arr, "maturity": maturity_arr, "price": price_arr}
    )
    clean = _validate_market_dataframe(df)
    clean.attrs.update(_base_market_metadata(metadata))
    return clean


def _market_metadata(df: pd.DataFrame) -> dict[str, Any]:
    metadata = _base_market_metadata(df.attrs)
    return metadata


def df_to_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Serialise ``df`` and a hash-linked manifest to CSV."""

    clean = _validate_market_dataframe(df)
    metadata = _market_metadata(df)

    def writer(tmp_path: Path) -> None:
        clean.to_csv(tmp_path, index=False)

    return _atomic_publish(
        path,
        artifact_type="market-data.csv",
        metadata=metadata,
        writer=writer,
        overwrite=overwrite,
        extra={
            "columns": list(clean.columns),
            "dtypes": clean.dtypes.astype(str).to_dict(),
        },
    )


def df_to_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Serialise ``df`` and a hash-linked manifest to Parquet."""

    clean = _validate_market_dataframe(df)
    metadata = _market_metadata(df)

    def writer(tmp_path: Path) -> None:
        _write_parquet_with_metadata(clean, tmp_path, metadata)

    return _atomic_publish(
        path,
        artifact_type="market-data.parquet",
        metadata=metadata,
        writer=writer,
        overwrite=overwrite,
        extra={
            "columns": list(clean.columns),
            "dtypes": clean.dtypes.astype(str).to_dict(),
        },
    )


def df_from_csv(path: str | Path) -> pd.DataFrame:
    """Load manifest-validated market data stored in CSV format."""

    manifest = _load_manifest(path, "market-data.csv")
    df = _validate_market_dataframe(pd.read_csv(path))
    df.attrs.update(manifest["metadata"])
    return df


def df_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load manifest-validated market data stored in Parquet format."""

    manifest = _load_manifest(path, "market-data.parquet")
    parquet_metadata = _read_parquet_metadata(path)
    if parquet_metadata.get("schema_version") != MARKET_DATA_SCHEMA_VERSION:
        raise ValueError("Parquet schema metadata is incompatible")
    if parquet_metadata.get("run_id") != manifest["metadata"].get("run_id"):
        raise ValueError("Parquet schema metadata does not match manifest")
    df = _validate_market_dataframe(pd.read_parquet(path))
    df.attrs.update(manifest["metadata"])
    return df


# ---------------------------------------------------------------------------
# Solution snapshot helpers


def _base_solution_metadata(
    *,
    coordinates: list[str],
    shape: tuple[int, ...],
    dtype: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    supplied = dict(metadata or {})
    base = {
        "schema_version": SOLUTION_SNAPSHOT_SCHEMA_VERSION,
        "run_id": supplied.get("run_id", f"run-{uuid.uuid4().hex}"),
        "units": supplied.get("units", "unspecified"),
        "model": supplied.get("model", "unspecified"),
        "config_hash": supplied.get("config_hash", "unspecified"),
        "backend": supplied.get("backend", "unspecified"),
        "convergence_state": supplied.get("convergence_state", "unknown"),
        "coordinates": coordinates,
        "shape": list(shape),
        "dtype": dtype,
    }
    base.update(supplied)
    base["schema_version"] = SOLUTION_SNAPSHOT_SCHEMA_VERSION
    base["coordinates"] = coordinates
    base["shape"] = list(shape)
    base["dtype"] = dtype
    return _json_roundtripable(base)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, str) and value and value[0] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _encode_attrs_for_netcdf(attrs: Mapping[str, Any]) -> dict[str, Any]:
    encoded: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, str | int | float | np.integer | np.floating):
            encoded[key] = value
        else:
            encoded[key] = json.dumps(value, default=_json_default, sort_keys=True)
    return encoded


def snapshot(
    array: np.ndarray,
    time_grid: Iterable[float],
    space_grid: Iterable[float],
    *,
    time_name: str = "time",
    space_name: str = "space",
    metadata: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Convert ``array`` into a validated :class:`xarray.DataArray` snapshot.

    Parameters
    ----------
    array:
        Two-dimensional time × space array of option values.
    time_grid, space_grid:
        Coordinate vectors for the respective axes.
    time_name, space_name:
        Coordinate/dimension names, e.g. ``tau`` and ``spot``.
    metadata:
        Optional run/model/backend/convergence metadata stored in attributes and
        artifact manifests when the snapshot is written.
    """

    values = np.asarray(array)
    time = np.asarray(list(time_grid), dtype=float)
    space = np.asarray(list(space_grid), dtype=float)
    if values.ndim != 2:
        raise ValueError("solution snapshot array must be two-dimensional")
    if values.shape != (len(time), len(space)):
        raise ValueError(
            "solution snapshot shape must match time and space coordinate lengths"
        )
    if (
        not np.isfinite(values).all()
        or not np.isfinite(time).all()
        or not np.isfinite(space).all()
    ):
        raise ValueError("solution snapshot values and coordinates must be finite")
    if len(set([time_name, space_name])) != 2:
        raise ValueError("solution snapshot coordinate names must be distinct")
    attrs = _base_solution_metadata(
        coordinates=[time_name, space_name],
        shape=values.shape,
        dtype=str(values.dtype),
        metadata=metadata,
    )
    return xr.DataArray(
        values,
        coords={time_name: time, space_name: space},
        dims=(time_name, space_name),
        name="solution",
        attrs=attrs,
    )


def _validate_snapshot(data: xr.DataArray) -> xr.DataArray:
    if data.ndim != 2:
        raise ValueError("solution snapshot must be two-dimensional")
    if len(data.dims) != 2 or len(set(data.dims)) != 2:
        raise ValueError("solution snapshot must have two distinct coordinates")
    if not np.isfinite(np.asarray(data.values, dtype=float)).all():
        raise ValueError("solution snapshot values must be finite")
    for dim in data.dims:
        if dim not in data.coords:
            raise ValueError(f"solution snapshot missing coordinate: {dim}")
        if not np.isfinite(np.asarray(data.coords[dim].values, dtype=float)).all():
            raise ValueError(f"solution snapshot coordinate {dim!r} must be finite")
    attrs = {key: _decode_attr(value) for key, value in data.attrs.items()}
    if attrs.get("schema_version") != SOLUTION_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("solution snapshot schema_version is missing or incompatible")
    if attrs.get("coordinates") != list(data.dims):
        raise ValueError("solution snapshot coordinate metadata does not match dims")
    if attrs.get("shape") != list(data.shape):
        raise ValueError("solution snapshot shape metadata does not match data")
    data.attrs.clear()
    data.attrs.update(attrs)
    return data


def snapshot_to_netcdf(
    data: xr.DataArray,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Serialise a solution snapshot and hash-linked manifest to NetCDF."""

    clean = _validate_snapshot(data.copy(deep=True))
    metadata = _json_roundtripable(clean.attrs)

    def writer(tmp_path: Path) -> None:
        encoded = clean.copy(deep=True)
        encoded.attrs = _encode_attrs_for_netcdf(encoded.attrs)
        encoded.to_netcdf(tmp_path, engine="scipy")

    return _atomic_publish(
        path,
        artifact_type="solution-snapshot.netcdf",
        metadata=metadata,
        writer=writer,
        overwrite=overwrite,
        extra={
            "coordinates": list(clean.dims),
            "shape": list(clean.shape),
            "dtype": str(clean.dtype),
        },
    )


def snapshot_from_netcdf(path: str | Path) -> xr.DataArray:
    """Load a manifest-validated solution snapshot from NetCDF."""

    manifest = _load_manifest(path, "solution-snapshot.netcdf")
    loaded = xr.open_dataarray(path, engine="scipy").load()
    loaded.attrs.update(manifest["metadata"])
    return _validate_snapshot(loaded)
