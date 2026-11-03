"""Validated plot exports and bounded portable-runtime delivery metadata.

The standalone native driver injects its report reader, so this outer artifact
owner cannot silently bypass field validation or import native solver packages.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import platform
import sys
import tarfile
from typing import Any


ARCHIVE_MAX_BYTES = 500 * 1024**2


def source_hashes(driver: Path) -> dict[str, str]:
    """Bind both exact source files, independent of their copied directory."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (driver, Path(__file__))
    }


def export_report(path: Path, *, read_report: Callable[[Path], dict[str, Any]]) -> None:
    """Render historical-named deliverables only from accepted measured fields."""
    report = read_report(path)
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = report["measurements"]
    backend = report["backend"]
    rms = float(np.sqrt(np.mean((arrays["u"] - arrays["exact"]) ** 2)))
    poisson = backend == "fenicsx"
    data = path.parent / f"{backend}_result.npz"
    image = path.parent / ("fenicsx_poisson.png" if poisson else "dedalus_heat.png")
    fig, ax = plt.subplots(figsize=(6, 5) if poisson else (7, 4))
    try:
        if poisson:
            contour = ax.tricontourf(arrays["x"], arrays["y"], arrays["u"], levels=25)
            fig.colorbar(contour, ax=ax, label="u")
            ax.set(xlabel="x", ylabel="y", aspect="equal")
        else:
            order = np.argsort(arrays["x"])
            ax.plot(arrays["x"][order], arrays["exact"][order], "--", label="exact")
            ax.plot(arrays["x"][order], arrays["u"][order], label="Dedalus")
            ax.legend()
            ax.set(xlabel="x", ylabel="u(x, 0.5)")
        title = "FEniCSx 0.11 Poisson" if poisson else "Dedalus two-mode heat equation"
        ax.set_title(f"{title}, nodal RMS={rms:.2e}")
        fig.tight_layout()
        fig.savefig(image, dpi=150, bbox_inches=None, facecolor="white")
    finally:
        plt.close(fig)
    np.savez_compressed(data, **arrays, rms=rms)
    manifest = {
        "schema": 1,
        "backend": backend,
        "phase": report["phase"],
        "script_sha256": report["script_sha256"],
        "source_sha256": report["source_sha256"],
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "data_sha256": report["data_sha256"],
        "matplotlib": matplotlib.__version__,
        "nodes": arrays["u"].size,
        "nodal_rms": rms,
        "fixture": (
            "Validated 64x64 finest P1 grid; successor of historical 24x24 plot."
            if poisson
            else "Validated Fourier64/two-mode/T0.5/40-step finest field; successor of historical Fourier96/single-mode/dt0.002 plot."
        ),
        "limits": "Nodal plot RMS is distinct from integrated convergence L2; plots do not establish native execution independently of the bound report/runtime evidence.",
        "exports": [
            {
                "name": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            for artifact in (data, image)
        ],
    }
    (path.parent / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


def archive_manifest(
    archive: Path,
    evidence: Path,
    *,
    read_report: Callable[[Path], dict[str, Any]],
    max_bytes: int = ARCHIVE_MAX_BYTES,
) -> None:
    """Measure a bounded archive; actual relocated replay remains a later gate."""
    report = read_report(evidence / "original/report.json")
    if (
        archive.is_symlink()
        or not archive.is_file()
        or not 0 < archive.stat().st_size <= min(max_bytes, ARCHIVE_MAX_BYTES)
    ):
        raise ValueError(
            "native archive must be a regular nonempty file no larger than500MiB"
        )
    if archive.name != "native-runtime.tar.xz":
        raise ValueError(
            "native archive must use the declared native-runtime.tar.xz name"
        )
    try:
        with tarfile.open(archive, mode="r:xz"):
            pass
    except (tarfile.TarError, OSError, EOFError) as error:
        raise ValueError("native archive must be a readable tar.xz file") from error
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    source = (evidence / "source.txt").read_text().splitlines()
    if len(source) != 2 or any(
        len(oid) != 40 or any(c not in "0123456789abcdef" for c in oid)
        for oid in source
    ):
        raise ValueError("archive source commit/tree identity is incomplete")
    packages = evidence / "conda-explicit.txt"
    manifest = {
        "schema": 1,
        "filename": archive.name,
        "format": "tar.xz",
        "compression": {"level": 6, "threads": 2},
        "max_bytes": ARCHIVE_MAX_BYTES,
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "source_commit": source[0],
        "source_tree": source[1],
        "platform": sys.platform,
        "architecture": platform.machine(),
        "producer_report_sha256": hashlib.sha256(
            (evidence / "original/report.json").read_bytes()
        ).hexdigest(),
        "script_sha256": report["script_sha256"],
        "source_sha256": report["source_sha256"],
        "package_list_sha256": hashlib.sha256(packages.read_bytes()).hexdigest(),
        "serial_mpi_environment": {"OMPI_MCA_pml": "ob1", "OMPI_MCA_btl": "self,tcp"},
        "distribution_gate": "Measured before testing; upload allowed only after successful same-run unpack and isolated complete-field native replay. Measurement alone does not establish delivery or replay.",
        "retention": "Separate public Actions artifact, compression0,1day;≤500MiB raw archive leaves margin below512MiB binary readback limit; upload/quota failure propagates.",
    }
    (evidence / "archive-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (evidence / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n")
    (evidence / "SOURCE_SHA256SUMS").write_text(
        "".join(
            f"{digest}  {name}\n" for name, digest in report["source_sha256"].items()
        )
    )
    (evidence / "README-runtime.md").write_text(
        "# Tested native FEniCSx environment\n\n"
        "This Linux x86_64 archive is tied to the exact source and package list in archive-manifest.json. "
        "It includes matplotlib-base3.11.2 for both original and relocated plot exports. "
        "The accompanying small numeric evidence records the same-run producer/relocated validation and exports. "
        "It is a synthetic demonstration runtime, not validation of pricing, calibration or distributed performance.\n\n"
        "Verify the archive before extracting into a new independent prefix:\n\n"
        "Standard tar.xz from maintained conda-pack0.9.2 replaces the former gzip artifact; "
        "preset6/two compression threads and a500MiB raw cap leave room for metadata/ZIP overhead below512MiB. "
        "A size measurement or header check is not independent complete artifact readback.\n\n"
        "```sh\nsha256sum --check SHA256SUMS\nmkdir runtime\n"
        "tar -xJf native-runtime.tar.xz -C runtime\n"
        "env -u PYTHONPATH -u PYTHONHOME runtime/bin/python -E -s runtime/bin/conda-unpack\n"
        "source runtime/bin/activate\nexport OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp\n"
        "python -I -c 'import dolfinx, matplotlib; print(dolfinx.__version__, matplotlib.__version__)'\n```\n\n"
        "Use source_commit to retrieve both scripts/validate_native_pde.py and "
        "scripts/native_pde_artifacts.py into one separate caller-owned directory, "
        "preserving both filenames. Compare sha256sum for both files with the "
        "source_sha256 mapping in archive-manifest.json before execution. "
        "The isolated driver loads only this exact sibling; no PYTHONPATH is needed. "
        "Then rerun the synthetic solve/export:\n\n"
        "```sh\nsha256sum --check SOURCE_SHA256SUMS\n"
        'export MPLCONFIGDIR="$PWD/matplotlib-cache"\n'
        'python -I validate_native_pde.py solve --backend fenicsx --output replay --phase original --expected-prefix "$CONDA_PREFIX"\n'
        "python -I validate_native_pde.py export replay/report.json\n```\n\n"
        "Do not move the environment again after conda-unpack. Other operating systems/architectures "
        "and MPI scaling are outside this evidence. Actions retention is one day; no delivery is inferred "
        "if the upload fails.\n"
    )
