"""Affine 1D Black--Scholes full- and reduced-order benchmark systems."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import numpy as np
import scipy.linalg as sla  # type: ignore[import-untyped]
import scipy.sparse as sps  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]
import scipy.stats as spst  # type: ignore[import-untyped]

from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs

from .assembly import (
    decomposition_hash as build_decomposition_hash,
    output_weights as build_output_weights,
    space_solver,
)
from .contracts import (
    FullOrderSolution,
    OptionOutputs,
    PODProjection,
    PymorBlackScholesConfig,
    ReducedOrderSolution,
    ROMEnvelopeError,
)
from .pymor_adapter import build_pod_projection


@dataclass(frozen=True, slots=True)
class AffineBlackScholesSystem:
    """Fixed-space affine full-order system used by the adoption gate."""

    config: PymorBlackScholesConfig
    mass: sps.csc_matrix
    operator_constant: sps.csc_matrix
    operator_variance: sps.csc_matrix
    coordinates: np.ndarray
    interior: np.ndarray
    left_boundary: int
    right_boundary: int
    output_weights: np.ndarray
    output_boundary_weights: np.ndarray
    decomposition_hash: str
    boundary_policy: str = "volatility-independent-asymptotic-call"

    @property
    def full_dofs(self) -> int:
        """Return the full finite-element degree count including boundaries."""

        return int(self.coordinates.size)

    @property
    def interior_dofs(self) -> int:
        """Return the full-order interior dimension."""

        return int(self.interior.size)

    def assemble_direct_operator(self, volatility: float) -> sps.csc_matrix:
        """Assemble the existing FEM operator directly for parity evidence."""

        sigma = self._validated_volatility(volatility, require_envelope=False)
        direct = sps.csc_matrix(space_solver(self.config, sigma).stiffness)
        return sps.csc_matrix(direct[self.interior][:, self.interior])

    def assemble_affine_operator(self, volatility: float) -> sps.csc_matrix:
        """Reconstruct the interior operator from its exact variance-affine terms."""

        sigma = self._validated_volatility(volatility, require_envelope=False)
        operator = self.operator_constant + sigma * sigma * self.operator_variance
        return sps.csc_matrix(operator[self.interior][:, self.interior])

    def prepare_full_order(self, volatility: float) -> PreparedFullOrderSolver:
        """Assemble and factorize one parameter-specific full-order system."""

        sigma = self._validated_volatility(volatility, require_envelope=False)
        config = self.config
        dt = config.maturity / config.time_steps
        operator = self.operator_constant + sigma * sigma * self.operator_variance
        lhs = self.mass - config.theta * dt * operator
        rhs_operator = self.mass + (1.0 - config.theta) * dt * operator
        interior = self.interior
        lhs_ii = sps.csc_matrix(lhs[interior][:, interior])
        return PreparedFullOrderSolver(
            system=self,
            volatility=sigma,
            lhs_interior=lhs_ii,
            rhs_interior=sps.csc_matrix(rhs_operator[interior][:, interior]),
            lhs_right=np.asarray(lhs[interior, self.right_boundary].toarray()).ravel(),
            rhs_right=np.asarray(rhs_operator[interior, self.right_boundary].toarray()).ravel(),
            factorized=spla.splu(lhs_ii),
        )

    def solve_full_order(
        self,
        volatility: float,
        *,
        capture_snapshots: bool = False,
    ) -> FullOrderSolution:
        """Solve one cold full-order query including assembly and factorization."""

        started = perf_counter()
        solution = self.prepare_full_order(volatility).solve(capture_snapshots=capture_snapshots)
        return FullOrderSolution(
            outputs=solution.outputs,
            final_interior=solution.final_interior,
            snapshots=solution.snapshots,
            elapsed_seconds=perf_counter() - started,
            residual_linf=solution.residual_linf,
            linear_solves=solution.linear_solves,
            operator_nnz=solution.operator_nnz,
        )

    def analytical_outputs(self, volatility: float) -> OptionOutputs:
        """Return analytical Black--Scholes price, Delta, and Gamma references."""

        sigma = self._validated_volatility(volatility, require_envelope=False)
        option = EuropeanOptionBs(k=self.config.strike, q=0.0, mkt=Market(r=self.config.rate))
        variance = sigma * sigma
        gamma = spst.norm.pdf(option.d1(self.config.maturity, self.config.spot, variance)) / (
            self.config.spot * sigma * np.sqrt(self.config.maturity)
        )
        return OptionOutputs(
            price=float(option.call_from_volatility(self.config.maturity, self.config.spot, sigma)),
            delta=float(
                option.call_delta_from_volatility(self.config.maturity, self.config.spot, sigma)
            ),
            gamma=float(gamma),
        )

    def _validated_volatility(self, volatility: float, *, require_envelope: bool) -> float:
        return _validated_volatility(
            self.config,
            volatility,
            require_envelope=require_envelope,
        )

    def _right_boundary_value(self, tau: float) -> float:
        return _right_boundary_value(self.config, tau)


@dataclass(frozen=True, slots=True)
class PreparedFullOrderSolver:
    """Parameter-specific factorization for a strict cached-FOM timing baseline."""

    system: AffineBlackScholesSystem
    volatility: float
    lhs_interior: sps.csc_matrix
    rhs_interior: sps.csc_matrix
    lhs_right: np.ndarray
    rhs_right: np.ndarray
    factorized: Any

    def solve(self, *, capture_snapshots: bool = False) -> FullOrderSolution:
        """March with an already assembled and factorized full-order operator."""

        started = perf_counter()
        config = self.system.config
        dt = config.maturity / config.time_steps
        current = np.maximum(
            self.system.coordinates[self.system.interior] - config.strike,
            0.0,
        )
        snapshots: list[np.ndarray] | None = [current.copy()] if capture_snapshots else None
        last_rhs = np.empty_like(current)
        for index in range(config.time_steps):
            start = index * dt
            end = (index + 1) * dt
            last_rhs = (
                self.rhs_interior @ current
                + self.rhs_right * self.system._right_boundary_value(start)
                - self.lhs_right * self.system._right_boundary_value(end)
            )
            current = np.asarray(self.factorized.solve(last_rhs), dtype=float)
            if snapshots is not None:
                snapshots.append(current.copy())
        boundary = np.array([0.0, self.system._right_boundary_value(config.maturity)])
        values = (
            self.system.output_weights @ current + self.system.output_boundary_weights @ boundary
        )
        return FullOrderSolution(
            outputs=OptionOutputs(*np.asarray(values, dtype=float)),
            final_interior=current,
            snapshots=None if snapshots is None else np.column_stack(snapshots),
            elapsed_seconds=perf_counter() - started,
            residual_linf=float(np.max(np.abs(self.lhs_interior @ current - last_rhs))),
            linear_solves=config.time_steps,
            operator_nnz=int(self.lhs_interior.nnz),
        )


@dataclass(frozen=True, slots=True)
class TrainedPymorROM:
    """Online-efficient POD/Galerkin model with fail-closed envelope checks."""

    config: PymorBlackScholesConfig
    full_dofs_count: int
    output_boundary_weights: np.ndarray
    projection: PODProjection
    training_seconds: float
    snapshot_count: int

    @property
    def library(self) -> str:
        """Return the maintained MOR implementation name."""

        return self.projection.library

    @property
    def basis_size(self) -> int:
        """Return reduced dimension."""

        return self.projection.basis_size

    @property
    def full_dofs(self) -> int:
        """Return full finite-element dimension."""

        return self.full_dofs_count

    @property
    def offline_seconds(self) -> float:
        """Return FOM training, POD, and projection wall time."""

        return (
            self.training_seconds
            + self.projection.setup_seconds
            + self.projection.pod_seconds
            + self.projection.projection_seconds
        )

    def solve(self, volatility: float) -> OptionOutputs:
        """Solve only reduced systems after checking the training envelope."""

        return self.solve_with_diagnostics(volatility).outputs

    def solve_with_diagnostics(self, volatility: float) -> ReducedOrderSolution:
        """Solve the reduced system and report its final linear residual."""

        sigma = _validated_volatility(self.config, volatility, require_envelope=True)
        config = self.config
        projection = self.projection
        dt = config.maturity / config.time_steps
        reduced_operator = (
            projection.reduced_operator_constant
            + sigma * sigma * projection.reduced_operator_variance
        )
        lhs = projection.reduced_mass - config.theta * dt * reduced_operator
        rhs_operator = projection.reduced_mass + (1.0 - config.theta) * dt * reduced_operator
        lhs_right = projection.reduced_mass_boundary - config.theta * dt * (
            projection.reduced_constant_boundary
            + sigma * sigma * projection.reduced_variance_boundary
        )
        rhs_right = projection.reduced_mass_boundary + (1.0 - config.theta) * dt * (
            projection.reduced_constant_boundary
            + sigma * sigma * projection.reduced_variance_boundary
        )
        factorized = sla.lu_factor(lhs)
        current = projection.reduced_initial.copy()
        last_rhs = np.empty_like(current)
        for index in range(config.time_steps):
            start = index * dt
            end = (index + 1) * dt
            last_rhs = (
                rhs_operator @ current
                + rhs_right * _right_boundary_value(config, start)
                - lhs_right * _right_boundary_value(config, end)
            )
            current = sla.lu_solve(factorized, last_rhs)
        boundary = np.array([0.0, _right_boundary_value(config, config.maturity)])
        values = projection.reduced_outputs @ current + self.output_boundary_weights @ boundary
        return ReducedOrderSolution(
            outputs=OptionOutputs(*np.asarray(values, dtype=float)),
            residual_linf=float(np.max(np.abs(lhs @ current - last_rhs))),
            linear_solves=config.time_steps,
            reduced_dimension=self.basis_size,
        )


def build_affine_black_scholes_system(
    config: PymorBlackScholesConfig,
) -> AffineBlackScholesSystem:
    """Build variance-affine FEM terms on one fixed line/P2 space."""

    low = space_solver(config, config.volatility_min)
    high = space_solver(config, config.volatility_max)
    low_operator = sps.csc_matrix(low.stiffness)
    high_operator = sps.csc_matrix(high.stiffness)
    low_variance = config.volatility_min**2
    high_variance = config.volatility_max**2
    operator_variance = (high_operator - low_operator) / (high_variance - low_variance)
    operator_constant = low_operator - low_variance * operator_variance
    mass = sps.csc_matrix(low.mass)
    coordinates = np.asarray(low.Vh.doflocs[0], dtype=float)
    left = int(np.argmin(np.abs(coordinates)))
    right = int(np.argmin(np.abs(coordinates - config.domain_max)))
    interior = np.setdiff1d(np.arange(coordinates.size), np.array([left, right]))
    full_output_weights = build_output_weights(config, low.Vh)
    boundaries = np.array([left, right])
    output_weights = full_output_weights[:, interior]
    output_boundary_weights = full_output_weights[:, boundaries]
    decomposition_hash = build_decomposition_hash(
        config,
        mass,
        sps.csc_matrix(operator_constant),
        sps.csc_matrix(operator_variance),
        coordinates,
        interior,
        output_weights,
        output_boundary_weights,
    )
    return AffineBlackScholesSystem(
        config=config,
        mass=mass,
        operator_constant=sps.csc_matrix(operator_constant),
        operator_variance=sps.csc_matrix(operator_variance),
        coordinates=coordinates,
        interior=interior,
        left_boundary=left,
        right_boundary=right,
        output_weights=output_weights,
        output_boundary_weights=output_boundary_weights,
        decomposition_hash=decomposition_hash,
    )


def train_pymor_rom(
    system: AffineBlackScholesSystem,
    config: PymorBlackScholesConfig,
) -> TrainedPymorROM:
    """Generate FOM snapshots and delegate POD/projection to pyMOR."""

    if config.input_hash != system.config.input_hash:
        raise ValueError("system and training configuration hashes differ")
    started = perf_counter()
    snapshots: list[np.ndarray] = []
    for volatility in config.training_volatilities:
        solution = system.solve_full_order(volatility, capture_snapshots=True)
        if solution.snapshots is None:  # pragma: no cover - guaranteed by call
            raise RuntimeError("training solve did not return snapshots")
        snapshots.append(np.ascontiguousarray(solution.snapshots[:, :: config.snapshot_stride]))
        del solution
    snapshot_matrix = np.column_stack(snapshots)
    training_seconds = perf_counter() - started
    interior = system.interior
    projection = build_pod_projection(
        snapshots=snapshot_matrix,
        mass=sps.csc_matrix(system.mass[interior][:, interior]),
        operator_constant=sps.csc_matrix(system.operator_constant[interior][:, interior]),
        operator_variance=sps.csc_matrix(system.operator_variance[interior][:, interior]),
        mass_boundary=np.asarray(system.mass[interior, system.right_boundary].toarray()).ravel(),
        constant_boundary=np.asarray(
            system.operator_constant[interior, system.right_boundary].toarray()
        ).ravel(),
        variance_boundary=np.asarray(
            system.operator_variance[interior, system.right_boundary].toarray()
        ).ravel(),
        initial=np.maximum(system.coordinates[interior] - config.strike, 0.0),
        output_weights=system.output_weights,
        max_basis_size=config.max_basis_size,
        pod_rtol=config.pod_rtol,
    )
    output_boundary_weights = system.output_boundary_weights.copy()
    output_boundary_weights.setflags(write=False)
    return TrainedPymorROM(
        config=config,
        full_dofs_count=system.full_dofs,
        output_boundary_weights=output_boundary_weights,
        projection=projection,
        training_seconds=training_seconds,
        snapshot_count=int(snapshot_matrix.shape[1]),
    )


def _validated_volatility(
    config: PymorBlackScholesConfig,
    volatility: float,
    *,
    require_envelope: bool,
) -> float:
    sigma = float(volatility)
    if not isfinite(sigma) or sigma <= 0.0:
        raise ValueError("volatility must be finite and positive")
    if require_envelope and not config.volatility_min <= sigma <= config.volatility_max:
        raise ROMEnvelopeError(
            f"volatility {sigma} is outside "
            f"[{config.volatility_min}, {config.volatility_max}]; "
            "use full_order_fem"
        )
    return sigma


def _right_boundary_value(config: PymorBlackScholesConfig, tau: float) -> float:
    return config.domain_max - config.strike * np.exp(-config.rate * tau)


__all__ = [
    "AffineBlackScholesSystem",
    "PreparedFullOrderSolver",
    "TrainedPymorROM",
    "build_affine_black_scholes_system",
    "train_pymor_rom",
]
