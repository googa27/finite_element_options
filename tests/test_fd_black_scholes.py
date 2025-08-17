import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.fdsolver import solve_system, delta  # noqa: E402
from src.core.dynamics_black_scholes import (
    DynamicsParametersBlackScholes,
)  # noqa: E402
from src.core.market import Market  # noqa: E402
from src.core.vanilla_bs import EuropeanOptionBs  # noqa: E402


def test_fd_black_scholes_price():
    dh = DynamicsParametersBlackScholes(r=0.03, q=0.0, sig=0.2)
    mkt = Market(r=dh.r)
    bsopt = EuropeanOptionBs(k=1.0, q=dh.q, mkt=mkt)
    s_grid = np.linspace(0.0, 2.0, 101)
    t = np.linspace(0.0, 1.0, 50)
    v = solve_system(s_grid, t, dh, bsopt, is_call=True)

    s0 = 1.0
    idx = np.argmin(np.abs(s_grid - s0))
    price_num = v.sel(time=t[-1], space=s0, method="nearest").item()
    price_exact = bsopt.call(t[-1], s0, dh.sig ** 2)
    assert price_num == pytest.approx(price_exact, rel=1e-2)

    d_num = delta(v.sel(time=t[-1]).values, s_grid[1] - s_grid[0])[idx]
    d_exact = bsopt.call_delta(t[-1], s0, dh.sig ** 2)
    assert d_num == pytest.approx(d_exact, rel=2e-2)
