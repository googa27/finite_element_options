"""Lazy pyMOR adapter for POD and affine Galerkin projection."""

from __future__ import annotations

from importlib.metadata import version
from time import perf_counter

import numpy as np
import scipy.linalg as sla  # type: ignore[import-untyped]
import scipy.sparse as sps  # type: ignore[import-untyped]

from .contracts import PODProjection


INSTALL_HINT = "install finite-element-options[reduction] to use the pyMOR adapter"


def build_pod_projection(
    *,
    snapshots: np.ndarray,
    mass: sps.csc_matrix,
    operator_constant: sps.csc_matrix,
    operator_variance: sps.csc_matrix,
    mass_boundary: np.ndarray,
    constant_boundary: np.ndarray,
    variance_boundary: np.ndarray,
    initial: np.ndarray,
    output_weights: np.ndarray,
    max_basis_size: int,
    pod_rtol: float,
) -> PODProjection:
    """Build a mass-POD basis and project every affine term with pyMOR."""

    adapter_started = perf_counter()
    try:
        from pymor.algorithms.pod import pod
        from pymor.algorithms.projection import project
        from pymor.algorithms.to_matrix import to_matrix
        from pymor.operators.numpy import NumpyMatrixOperator
        from pymor.vectorarrays.numpy import NumpyVectorSpace
    except ImportError as exc:  # pragma: no cover - tested in isolated wheel probe
        raise ModuleNotFoundError(INSTALL_HINT) from exc

    snapshot_matrix = np.asarray(snapshots, dtype=float)
    if snapshot_matrix.ndim != 2 or snapshot_matrix.shape[1] < 2:
        raise ValueError("snapshots must have shape (full_dofs, at least_two_snapshots)")
    if snapshot_matrix.shape[0] != mass.shape[0]:
        raise ValueError("snapshot and mass dimensions differ")
    if not np.all(np.isfinite(snapshot_matrix)):
        raise FloatingPointError("snapshots must be finite")

    vector_space = NumpyVectorSpace(snapshot_matrix.shape[0])
    snapshot_array = vector_space.from_numpy(snapshot_matrix)
    mass_operator = NumpyMatrixOperator(sps.csc_matrix(mass))
    setup_seconds = perf_counter() - adapter_started
    started = perf_counter()
    basis, singular_values = pod(
        snapshot_array,
        product=mass_operator,
        modes=max_basis_size,
        rtol=pod_rtol,
    )
    pod_seconds = perf_counter() - started
    if len(basis) < 1:
        raise RuntimeError("pyMOR POD returned an empty basis")
    snapshot_energy = float(np.sum(snapshot_matrix * (mass @ snapshot_matrix)))
    captured_energy = float(np.sum(np.asarray(singular_values, dtype=float) ** 2))
    captured_energy_fraction = captured_energy / snapshot_energy

    started = perf_counter()
    constant_operator = NumpyMatrixOperator(sps.csc_matrix(operator_constant))
    variance_operator = NumpyMatrixOperator(sps.csc_matrix(operator_variance))
    reduced_mass = np.asarray(to_matrix(project(mass_operator, basis, basis)), dtype=float)
    reduced_constant = np.asarray(to_matrix(project(constant_operator, basis, basis)), dtype=float)
    reduced_variance = np.asarray(to_matrix(project(variance_operator, basis, basis)), dtype=float)
    basis_matrix = np.asarray(basis.to_numpy(), dtype=float)
    reduced_mass_boundary = basis_matrix.T @ np.asarray(mass_boundary, dtype=float)
    reduced_constant_boundary = basis_matrix.T @ np.asarray(constant_boundary, dtype=float)
    reduced_variance_boundary = basis_matrix.T @ np.asarray(variance_boundary, dtype=float)
    reduced_initial = sla.solve(
        reduced_mass,
        basis_matrix.T @ (mass @ np.asarray(initial, dtype=float)),
        assume_a="sym",
    )
    reduced_outputs = np.asarray(output_weights, dtype=float) @ basis_matrix
    projection_seconds = perf_counter() - started

    arrays = (
        basis_matrix,
        np.asarray(singular_values, dtype=float),
        reduced_mass,
        reduced_constant,
        reduced_variance,
        reduced_mass_boundary,
        reduced_constant_boundary,
        reduced_variance_boundary,
        reduced_initial,
        reduced_outputs,
    )
    for array in arrays:
        array.setflags(write=False)
    return PODProjection(
        library="pymor",
        library_version=version("pymor"),
        basis=basis_matrix,
        singular_values=arrays[1],
        reduced_mass=reduced_mass,
        reduced_operator_constant=reduced_constant,
        reduced_operator_variance=reduced_variance,
        reduced_mass_boundary=reduced_mass_boundary,
        reduced_constant_boundary=reduced_constant_boundary,
        reduced_variance_boundary=reduced_variance_boundary,
        reduced_initial=reduced_initial,
        reduced_outputs=reduced_outputs,
        captured_energy_fraction=captured_energy_fraction,
        setup_seconds=setup_seconds,
        pod_seconds=pod_seconds,
        projection_seconds=projection_seconds,
    )


__all__ = ["INSTALL_HINT", "build_pod_projection"]
