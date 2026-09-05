"""Fixed-space Black--Scholes assembly and deterministic identity helpers."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps  # type: ignore[import-untyped]
from skfem import Basis

from finite_element_options.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.mesh import create_mesh
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.validation.evidence.serialization import (
    canonical_json_sha256,
    quantize_json_floats,
)

from .contracts import PymorBlackScholesConfig


def space_solver(config: PymorBlackScholesConfig, volatility: float) -> SpaceSolver:
    """Assemble the repository's P2 Black--Scholes FEM space at one volatility."""

    dynamics = DynamicsParametersBlackScholes(r=config.rate, q=0.0, sig=volatility)
    option = EuropeanOptionBs(k=config.strike, q=0.0, mkt=Market(r=config.rate))
    mesh, finite_element_config = create_mesh([config.domain_max], config.refinement_level)
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], config.domain_max),
        }
    )
    return SpaceSolver(
        mesh,
        dynamics,
        option,
        is_call=True,
        config=finite_element_config,
    )


def output_weights(
    config: PymorBlackScholesConfig,
    basis: Basis,
) -> np.ndarray:
    """Build centered finite-bump price, Delta, and Gamma functionals."""

    lower = _value_weights(basis, config.spot - config.greek_bump)
    center = _value_weights(basis, config.spot)
    upper = _value_weights(basis, config.spot + config.greek_bump)
    delta = (upper - lower) / (2.0 * config.greek_bump)
    gamma = (upper - 2.0 * center + lower) / (config.greek_bump**2)
    return np.vstack((center, delta, gamma))


def decomposition_hash(
    config: PymorBlackScholesConfig,
    *arrays: sps.csc_matrix | np.ndarray,
) -> str:
    """Hash a 10-significant-digit canonical affine-system representation."""

    payload: dict[str, object] = {"study_input_hash": config.input_hash, "arrays": []}
    records: list[dict[str, object]] = []
    for array in arrays:
        if sps.issparse(array):
            sparse = sps.csc_matrix(array)
            shape = sparse.shape
            if shape is None:  # pragma: no cover - SciPy matrices always have a shape
                raise ValueError("sparse affine term has no shape")
            records.append(
                {
                    "shape": [int(shape[0]), int(shape[1])],
                    "data": quantize_json_floats(sparse.data, significant_digits=10),
                    "indices": sparse.indices.tolist(),
                    "indptr": sparse.indptr.tolist(),
                }
            )
        else:
            dense = np.asarray(array)
            records.append(
                {
                    "shape": list(dense.shape),
                    "data": quantize_json_floats(dense, significant_digits=10),
                }
            )
    payload["arrays"] = records
    return canonical_json_sha256(payload)


def _value_weights(basis: Basis, target: float) -> np.ndarray:
    probe = basis.probes(np.array([[target]], dtype=float))
    return np.asarray(probe.toarray(), dtype=float).ravel()


__all__ = ["decomposition_hash", "output_weights", "space_solver"]
