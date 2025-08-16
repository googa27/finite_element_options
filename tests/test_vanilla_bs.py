import numpy as np
import pytest

from pathlib import Path
import sys

# Ensure project root is on ``sys.path`` so ``src`` can be imported when the
# package isn't installed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vanilla_bs import EuropeanOptionBs
from src.market import Market


@pytest.fixture
def bs_option():
    mkt = Market(r=0.05)
    return EuropeanOptionBs(k=100, q=0.0, mkt=mkt)


def test_call_payoff(bs_option):
    assert bs_option.call_payoff(120) == pytest.approx(20.0)
    assert bs_option.call_payoff(80) == pytest.approx(0.0)


def test_put_payoff(bs_option):
    assert bs_option.put_payoff(80) == pytest.approx(20.0)
    assert bs_option.put_payoff(120) == pytest.approx(0.0)


def test_call_price(bs_option):
    price = bs_option.call(th=1.0, s=100.0, v=0.04)
    assert price == pytest.approx(10.4506, rel=1e-4)


def test_put_price(bs_option):
    price = bs_option.put(th=1.0, s=100.0, v=0.04)
    assert price == pytest.approx(5.5735, rel=1e-4)


def test_delta(bs_option):
    delta = bs_option.call_delta(th=1.0, s=100.0, v=0.04)
    assert delta == pytest.approx(0.6368, rel=1e-3)


def test_discount_factor():
    mkt = Market(r=0.05)
    assert mkt.D(1.0) == pytest.approx(np.exp(-0.05))
