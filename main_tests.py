import pytest
from src.vanilla_bs import EuropeanOptionBs
from src.market import Market


def test_call_payoff():
    mkt = Market(r=0.05)
    opt = EuropeanOptionBs(k=100, q=0.0, mkt=mkt)
    assert opt.call_payoff(120) == pytest.approx(20.0)
    assert opt.call_payoff(80) == pytest.approx(0.0)


def test_put_payoff():
    mkt = Market(r=0.05)
    opt = EuropeanOptionBs(k=100, q=0.0, mkt=mkt)
    assert opt.put_payoff(80) == pytest.approx(20.0)
    assert opt.put_payoff(120) == pytest.approx(0.0)
