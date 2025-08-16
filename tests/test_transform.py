import numpy as np

from src.transform import (
    CoordinateTransform,
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
