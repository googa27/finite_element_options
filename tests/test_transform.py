import numpy as np
import pytest
import skfem as fem

from finite_element_options.core.config import Config
from finite_element_options.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)
from finite_element_options.core.dynamics_heston import DynamicsParametersHeston
from finite_element_options.core.market import Market
from finite_element_options.core.vanilla_bs import EuropeanOptionBs
from finite_element_options.space.boundary import DirichletBC
from finite_element_options.space.forms import PDEForms
from finite_element_options.space.solver import SpaceSolver
from finite_element_options.time_integration.stepper import ThetaScheme
from finite_element_options.transform import (
    CoordinateTransform,
    Identity,
    LogPrice,
    SqrtVol,
    TimeToMaturity,
)


def test_log_price_cycle():
    vals = np.array([0.5, 1.0, 2.0])
    trans = LogPrice()
    np.testing.assert_allclose(vals, trans.untransform(trans.transform(vals)))


def test_sqrt_vol_cycle():
    vals = np.array([0.04, 0.16, 0.25])
    trans = SqrtVol()
    np.testing.assert_allclose(vals, trans.untransform(trans.transform(vals)))


def test_time_to_maturity_cycle():
    T = 1.0
    vals = np.array([0.0, 0.4, 1.0])
    trans = TimeToMaturity(maturity=T)
    np.testing.assert_allclose(vals, trans.untransform(trans.transform(vals)))


def test_coordinate_transform_state_cycle():
    ct = CoordinateTransform(price=LogPrice(), vol=SqrtVol())
    coords = np.array([[1.0, 2.0], [0.04, 0.09]])
    transformed = ct.transform_state(coords)
    back = ct.untransform_state(transformed)
    np.testing.assert_allclose(coords, back)


def test_custom_mapping_without_derivatives_fails_closed_for_coefficients():
    """Coordinate-only custom maps must not silently define PDE coefficients."""

    class ShiftOnly:
        def transform(self, x):
            return x + 1.0

        def untransform(self, x):
            return x - 1.0

    dynamics = DynamicsParametersBlackScholes(r=0.05, q=0.01, sig=0.2)
    transform = CoordinateTransform(price=ShiftOnly())

    with pytest.raises(TypeError, match="derivative and second_derivative"):
        transform.transformed_coefficients(dynamics, np.array([[1.0]]))


def test_log_price_black_scholes_transformed_generator_coefficients():
    """Log-price coordinates use Ito chain-rule coefficients, not spot coefficients."""

    dynamics = DynamicsParametersBlackScholes(r=0.05, q=0.01, sig=0.2)
    transform = CoordinateTransform(price=LogPrice())
    log_spot = np.log(np.array([[80.0, 100.0, 120.0]]))

    diffusion, diffusion_divergence, drift = transform.transformed_coefficients(
        dynamics,
        log_spot,
    )

    np.testing.assert_allclose(diffusion[0][0], np.full(3, dynamics.sig**2))
    np.testing.assert_allclose(diffusion_divergence[0], np.zeros(3), atol=1.0e-14)
    np.testing.assert_allclose(
        drift[0],
        np.full(3, dynamics.r - dynamics.q - 0.5 * dynamics.sig**2),
    )


def test_log_price_form_matches_native_log_black_scholes_form():
    """Weak-form assembly must consume transformed coefficients in log space."""

    physical = DynamicsParametersBlackScholes(r=0.05, q=0.01, sig=0.2)
    class NativeLogBlackScholes:
        r = physical.r

        def A(self, x):
            return [[physical.sig**2 + 0.0 * x]]

        def dA(self, x):
            return [0.0 * x]

        def b(self, x):
            return [physical.r - physical.q - 0.5 * physical.sig**2 + 0.0 * x]

    log_dynamics = NativeLogBlackScholes()

    payoff = EuropeanOptionBs(k=100.0, q=physical.q, mkt=Market(r=physical.r))
    mesh = fem.MeshLine(np.linspace(np.log(50.0), np.log(150.0), 6))
    basis = fem.CellBasis(mesh, fem.ElementLineP2())

    transformed_form = PDEForms(
        is_call=True,
        payoff=payoff,
        dynamics=physical,
        transform=CoordinateTransform(price=LogPrice()),
    ).l_bil().assemble(basis)
    native_log_form = PDEForms(
        is_call=True,
        payoff=payoff,
        dynamics=log_dynamics,
        transform=CoordinateTransform(price=Identity()),
    ).l_bil().assemble(basis)

    np.testing.assert_allclose(transformed_form.toarray(), native_log_form.toarray())


def test_log_price_black_scholes_solution_matches_spot_solution_after_interpolation():
    """Physical and log-price solves agree at shared spot locations."""

    dynamics = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    payoff = EuropeanOptionBs(k=1.0, q=dynamics.q, mkt=Market(r=dynamics.r))
    time_grid = np.linspace(0.0, 1.0, 9)
    stepper = ThetaScheme(theta=0.5)

    physical_space = SpaceSolver(
        fem.MeshLine(np.linspace(0.0, 2.0, 17)),
        dynamics,
        payoff,
        is_call=True,
        config=Config(elem=fem.ElementLineP2()),
    )
    log_space = SpaceSolver(
        fem.MeshLine(np.linspace(np.log(0.25), np.log(2.0), 17)),
        dynamics,
        payoff,
        is_call=True,
        transform=CoordinateTransform(price=LogPrice()),
        config=Config(elem=fem.ElementLineP2()),
    )

    physical_solution = stepper.solve(
        time_grid,
        physical_space,
        boundary_condition=DirichletBC([]),
    )
    log_solution = stepper.solve(
        time_grid,
        log_space,
        boundary_condition=DirichletBC([]),
    )

    spot = np.array([0.75, 1.0, 1.25])
    physical_dofs = physical_space.Vh.doflocs[0]
    log_dofs = log_space.Vh.doflocs[0]
    physical_order = np.argsort(physical_dofs)
    log_order = np.argsort(log_dofs)
    physical_prices = np.interp(
        spot,
        physical_dofs[physical_order],
        physical_solution[-1, physical_order],
    )
    log_prices = np.interp(
        np.log(spot),
        log_dofs[log_order],
        log_solution[-1, log_order],
    )

    np.testing.assert_allclose(log_prices, physical_prices, rtol=3.0e-2, atol=3.0e-3)


def test_heston_log_sqrt_transform_coefficients_and_psd_covariance():
    """Log-spot/sqrt-variance coordinates include Hessian drift corrections."""

    dynamics = DynamicsParametersHeston(
        r=0.04,
        q=0.01,
        kappa=1.7,
        theta=0.05,
        sig=0.3,
        rho=-0.4,
    )
    physical_state = np.array([[80.0, 100.0, 120.0], [0.04, 0.09, 0.16]])
    transformed_state = CoordinateTransform(
        price=LogPrice(),
        vol=SqrtVol(),
    ).transform_state(physical_state)

    diffusion, diffusion_divergence, drift = CoordinateTransform(
        price=LogPrice(),
        vol=SqrtVol(),
    ).transformed_coefficients(dynamics, transformed_state)

    sqrt_v = np.sqrt(physical_state[1])
    np.testing.assert_allclose(diffusion[0][0], physical_state[1])
    np.testing.assert_allclose(diffusion[0][1], 0.5 * dynamics.rho * dynamics.sig * sqrt_v)
    np.testing.assert_allclose(diffusion[1][0], 0.5 * dynamics.rho * dynamics.sig * sqrt_v)
    np.testing.assert_allclose(diffusion[1][1], np.full(3, dynamics.sig**2 / 4.0))
    np.testing.assert_allclose(
        diffusion_divergence[0],
        np.full(3, 0.5 * dynamics.rho * dynamics.sig),
    )
    np.testing.assert_allclose(diffusion_divergence[1], np.zeros(3), atol=1.0e-14)
    np.testing.assert_allclose(drift[0], dynamics.r - dynamics.q - 0.5 * physical_state[1])
    np.testing.assert_allclose(
        drift[1],
        dynamics.kappa * (dynamics.theta - physical_state[1]) / (2.0 * sqrt_v)
        - dynamics.sig**2 / (8.0 * sqrt_v),
    )

    for point in range(physical_state.shape[1]):
        covariance = np.array(
            [[np.asarray(entry)[point] for entry in row] for row in diffusion],
            dtype=float,
        )
        assert np.linalg.eigvalsh(covariance).min() >= -1.0e-14


def test_heston_log_sqrt_diffusion_divergence_matches_finite_difference():
    """Returned divergence is the row divergence of transformed diffusion."""

    dynamics = DynamicsParametersHeston(
        r=0.04,
        q=0.01,
        kappa=1.7,
        theta=0.05,
        sig=0.3,
        rho=-0.4,
    )
    transform = CoordinateTransform(price=LogPrice(), vol=SqrtVol())
    transformed_state = transform.transform_state(np.array([[100.0], [0.09]]))
    diffusion, diffusion_divergence, _drift = transform.transformed_coefficients(
        dynamics,
        transformed_state,
    )

    def diffusion_entry(point, row: int, col: int) -> float:
        matrix, _divergence, _drift = transform.transformed_coefficients(dynamics, point)
        return float(np.asarray(matrix[row][col]).reshape(-1)[0])

    eps = 1.0e-5
    finite_difference_divergence = []
    for row in range(2):
        row_divergence = 0.0
        for col in range(2):
            plus = transformed_state.copy()
            minus = transformed_state.copy()
            plus[col, 0] += eps
            minus[col, 0] -= eps
            row_divergence += (
                diffusion_entry(plus, row, col) - diffusion_entry(minus, row, col)
            ) / (2.0 * eps)
        finite_difference_divergence.append(row_divergence)

    assert float(np.asarray(diffusion[0][0]).reshape(-1)[0]) == pytest.approx(0.09)
    np.testing.assert_allclose(
        [float(np.asarray(item).reshape(-1)[0]) for item in diffusion_divergence],
        finite_difference_divergence,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_sqrt_variance_transform_rejects_singular_boundary_before_assembly():
    dynamics = DynamicsParametersHeston(
        r=0.04,
        q=0.01,
        kappa=1.7,
        theta=0.05,
        sig=0.3,
        rho=-0.4,
    )
    singular_state = np.array([[np.log(100.0)], [0.0]])

    with pytest.raises(ValueError, match="strictly positive"):
        CoordinateTransform(price=LogPrice(), vol=SqrtVol()).transformed_coefficients(
            dynamics,
            singular_state,
        )


def test_sqrt_variance_transform_rejects_negative_transformed_coordinate():
    """Negative root-variance coordinates are outside the sqrt transform range."""

    dynamics = DynamicsParametersHeston(
        r=0.04,
        q=0.01,
        kappa=1.7,
        theta=0.05,
        sig=0.3,
        rho=-0.4,
    )
    negative_root_state = np.array([[np.log(100.0)], [-0.2]])

    with pytest.raises(ValueError, match="non-negative"):
        CoordinateTransform(price=LogPrice(), vol=SqrtVol()).transformed_coefficients(
            dynamics,
            negative_root_state,
        )


def test_heston_log_sqrt_space_solver_rejects_mesh_touching_singular_boundary():
    """Mesh nodes touching sqrt-variance zero are rejected before assembly."""

    dynamics = DynamicsParametersHeston(
        r=0.04,
        q=0.01,
        kappa=1.7,
        theta=0.05,
        sig=0.3,
        rho=-0.4,
    )
    payoff = EuropeanOptionBs(k=100.0, q=dynamics.q, mkt=Market(r=dynamics.r))
    mesh = fem.MeshTri().init_tensor(
        x=np.linspace(np.log(50.0), np.log(150.0), 3),
        y=np.linspace(0.0, 0.3, 3),
    )

    with pytest.raises(ValueError, match="strictly positive"):
        SpaceSolver(
            mesh,
            dynamics,
            payoff,
            is_call=True,
            transform=CoordinateTransform(price=LogPrice(), vol=SqrtVol()),
        )
