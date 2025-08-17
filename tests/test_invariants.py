"""Property-based tests for payoff and boundary condition invariants."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps
from hypothesis import given, strategies as st

from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.space.boundary import apply_dirichlet


@given(
    s=st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
    k=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_payoff_invariants(s: float, k: float) -> None:
    """European option payoffs are non-negative and match definitions."""
    mkt = Market(r=0.01)
    payoff = EuropeanOptionBs(k=k, q=0.0, mkt=mkt)

    call = payoff.call_payoff(s)
    put = payoff.put_payoff(s)

    assert call >= 0.0
    assert put >= 0.0
    assert call == pytest.approx(max(s - k, 0.0))
    assert put == pytest.approx(max(k - s, 0.0))


@given(
    n=st.integers(min_value=2, max_value=6),
    data=st.data(),
)
def test_apply_dirichlet_enforces_values(n: int, data: st.DataObject) -> None:
    """apply_dirichlet sets enforced rows, columns and vector entries."""
    dofs = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            min_size=1,
            max_size=n,
            unique=True,
        )
    )

    A = sps.csr_matrix(np.arange(n * n, dtype=float).reshape(n, n))
    b = np.arange(n, dtype=float)
    x = np.arange(n, dtype=float) + 0.5

    class _Basis:
        def get_dofs(self, boundaries):
            return np.array(dofs)

    A_enf, b_enf = apply_dirichlet(A.copy(), b.copy(), _Basis(), ["bd"], x)

    for idx in dofs:
        row = A_enf.getrow(idx).toarray().ravel()
        expected = np.zeros(n)
        expected[idx] = 1.0
        assert np.allclose(row, expected)
        assert b_enf[idx] == pytest.approx(x[idx])

    for idx in set(range(n)) - set(dofs):
        assert np.allclose(
            A_enf.getrow(idx).toarray(), A.getrow(idx).toarray()
        )
        assert b_enf[idx] == pytest.approx(b[idx])
