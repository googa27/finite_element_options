"""Black-Scholes vanilla option pricing utilities.

The public analytical oracle is explicit about the stochastic variable used at
its boundary: option prices can be requested from either volatility ``sigma`` or
variance ``sigma**2``. Greeks name the differentiated variable so FEM/FD
benchmarks cannot compare volatility vega against variance sensitivity by
accident.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import special

from .interfaces import Payoff
from .market import Market

_NEAR_ZERO_STD = 1.0e-12
_KINK_ATOL_SCALE = 1.0e-14


def _as_float_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _is_scalar_input(*values: Any) -> bool:
    return all(np.asarray(value).ndim == 0 for value in values)


def _as_output(value: np.ndarray, *, scalar: bool = False) -> float | np.ndarray:
    array = np.asarray(value, dtype=float)
    if scalar:
        return float(array.reshape(-1)[0])
    return array


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class EuropeanOptionBs(Payoff):
    """European option priced with the Black-Scholes model.

    Parameters are stored as attributes via ``dataclass`` which promotes a more
    declarative and immutable design.  Price APIs are explicit about whether the
    stochastic input is volatility or variance; legacy ``call``/``put`` methods
    remain variance-based for compatibility with existing solver tests.
    """

    k: float
    q: float
    mkt: Market

    def __post_init__(self) -> None:
        """Validate immutable Black-Scholes oracle parameters."""

        if not np.isfinite(self.k) or self.k <= 0.0:
            raise ValueError("strike k must be finite and positive")
        if not np.isfinite(self.q):
            raise ValueError("dividend/carry rate q must be finite")
        if not np.isfinite(self.mkt.r):
            raise ValueError("market rate r must be finite")

    @property
    def r(self) -> float:
        """Risk-free rate of the associated market."""

        return self.mkt.r

    def dividend_discount(self, th: float) -> float:
        """Dividend discount factor ``exp(-q * th)``."""

        maturity = self._validate_maturity(th)
        return np.exp(-self.q * maturity)

    def forward_price(self, th: float, s: float | np.ndarray) -> float | np.ndarray:
        """Forward price of the underlying asset."""

        maturity = self._validate_maturity(th)
        spot = self._validate_spot(s)
        forward = (
            spot * self.dividend_discount(maturity) / self.mkt.discount_factor(maturity)
        )
        return _as_output(forward, scalar=_is_scalar_input(s))

    def d1(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Black-Scholes ``d1`` for strictly positive variance and maturity."""

        maturity = self._validate_maturity(th)
        variance_array = self._validate_variance(variance)
        if maturity <= 0.0 or np.any(variance_array <= 0.0):
            raise ValueError("d1 requires positive maturity and positive variance")
        spot, variance_b = np.broadcast_arrays(self._validate_spot(s), variance_array)
        volatility = np.sqrt(variance_b)
        d1, _ = self._d_terms(maturity, spot, volatility)
        return _as_output(d1, scalar=_is_scalar_input(s, variance))

    def d2(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Black-Scholes ``d2`` for strictly positive variance and maturity."""

        maturity = self._validate_maturity(th)
        variance_array = self._validate_variance(variance)
        if maturity <= 0.0 or np.any(variance_array <= 0.0):
            raise ValueError("d2 requires positive maturity and positive variance")
        spot, variance_b = np.broadcast_arrays(self._validate_spot(s), variance_array)
        volatility = np.sqrt(variance_b)
        _, d2 = self._d_terms(maturity, spot, volatility)
        return _as_output(d2, scalar=_is_scalar_input(s, variance))

    def call_from_volatility(
        self, th: float, s: float | np.ndarray, volatility: float | np.ndarray
    ) -> float | np.ndarray:
        """Price a European call from volatility ``sigma``."""

        return _as_output(
            self._price_from_volatility(th, s, volatility, is_call=True),
            scalar=_is_scalar_input(s, volatility),
        )

    def call_from_variance(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Price a European call from variance ``sigma**2``."""

        return self.call_from_volatility(
            th, s, np.sqrt(self._validate_variance(variance))
        )

    def call(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Compatibility wrapper: call price from variance ``sigma**2``."""

        return self.call_from_variance(th, s, variance)

    def call_payoff(self, s: float | np.ndarray) -> float | np.ndarray:
        """Intrinsic value of a call option at spot price ``s``."""

        return _as_output(
            np.maximum(0.0, self._validate_spot(s) - self.k),
            scalar=_is_scalar_input(s),
        )

    def call_delta_from_volatility(
        self, th: float, s: float | np.ndarray, volatility: float | np.ndarray
    ) -> float | np.ndarray:
        """Spot delta of a European call from volatility ``sigma``."""

        return _as_output(
            self._delta_from_volatility(th, s, volatility, is_call=True),
            scalar=_is_scalar_input(s, volatility),
        )

    def call_delta_from_variance(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Spot delta of a European call from variance ``sigma**2``."""

        return self.call_delta_from_volatility(
            th, s, np.sqrt(self._validate_variance(variance))
        )

    def call_delta(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Compatibility wrapper: call spot delta from variance ``sigma**2``."""

        return self.call_delta_from_variance(th, s, variance)

    def put_from_volatility(
        self, th: float, s: float | np.ndarray, volatility: float | np.ndarray
    ) -> float | np.ndarray:
        """Price a European put from volatility ``sigma``."""

        return _as_output(
            self._price_from_volatility(th, s, volatility, is_call=False),
            scalar=_is_scalar_input(s, volatility),
        )

    def put_from_variance(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Price a European put from variance ``sigma**2``."""

        return self.put_from_volatility(
            th, s, np.sqrt(self._validate_variance(variance))
        )

    def put(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Compatibility wrapper: put price from variance ``sigma**2``."""

        return self.put_from_variance(th, s, variance)

    def put_payoff(self, s: float | np.ndarray) -> float | np.ndarray:
        """Intrinsic value of a put option at spot price ``s``."""

        return _as_output(
            np.maximum(0.0, self.k - self._validate_spot(s)),
            scalar=_is_scalar_input(s),
        )

    def put_delta_from_volatility(
        self, th: float, s: float | np.ndarray, volatility: float | np.ndarray
    ) -> float | np.ndarray:
        """Spot delta of a European put from volatility ``sigma``."""

        return _as_output(
            self._delta_from_volatility(th, s, volatility, is_call=False),
            scalar=_is_scalar_input(s, volatility),
        )

    def put_delta_from_variance(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Spot delta of a European put from variance ``sigma**2``."""

        return self.put_delta_from_volatility(
            th, s, np.sqrt(self._validate_variance(variance))
        )

    def put_delta(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Compatibility wrapper: put spot delta from variance ``sigma**2``."""

        return self.put_delta_from_variance(th, s, variance)

    def vega_volatility(
        self, th: float, s: float | np.ndarray, volatility: float | np.ndarray
    ) -> float | np.ndarray:
        """Derivative of option value with respect to volatility ``sigma``."""

        maturity = self._validate_maturity(th)
        spot, sigma = np.broadcast_arrays(
            self._validate_spot(s), self._validate_volatility(volatility)
        )
        spot = np.atleast_1d(spot)
        sigma = np.atleast_1d(sigma)
        result = np.zeros_like(spot, dtype=float)
        if maturity == 0.0:
            return _as_output(result, scalar=_is_scalar_input(s, volatility))

        total_std = sigma * np.sqrt(maturity)
        regular = (total_std > _NEAR_ZERO_STD) & (spot > 0.0)
        if np.any(regular):
            _, d2 = self._d_terms(maturity, spot[regular], sigma[regular])
            result[regular] = (
                self.k
                * self.mkt.discount_factor(maturity)
                * _normal_pdf(d2)
                * np.sqrt(maturity)
            )

        near_zero = ~regular
        if np.any(near_zero):
            forward = self._forward_array(maturity, spot[near_zero])
            atm = self._at_kink(forward)
            result[near_zero] = np.where(
                atm,
                self.k
                * self.mkt.discount_factor(maturity)
                * np.sqrt(maturity)
                / np.sqrt(2.0 * np.pi),
                0.0,
            )
        return _as_output(result, scalar=_is_scalar_input(s, volatility))

    def sensitivity_variance(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Derivative of option value with respect to variance ``sigma**2``."""

        variance_array = self._validate_variance(variance)
        if np.any(variance_array <= 0.0):
            raise ValueError("variance sensitivity is undefined at zero variance")
        volatility = np.sqrt(variance_array)
        return _as_output(
            self.vega_volatility(th, s, volatility) / (2.0 * volatility),
            scalar=_is_scalar_input(s, variance),
        )

    def vega(
        self, th: float, s: float | np.ndarray, variance: float | np.ndarray
    ) -> float | np.ndarray:
        """Compatibility wrapper: volatility vega from variance ``sigma**2``."""

        return self.vega_volatility(th, s, np.sqrt(self._validate_variance(variance)))

    def _validate_maturity(self, th: float) -> float:
        maturity = float(_as_float_array("maturity", th))
        if maturity < 0.0:
            raise ValueError("maturity must be non-negative")
        return maturity

    def _validate_spot(self, s: float | np.ndarray) -> np.ndarray:
        spot = _as_float_array("spot", s)
        if np.any(spot < 0.0):
            raise ValueError("spot must be non-negative")
        return spot

    def _validate_volatility(self, volatility: float | np.ndarray) -> np.ndarray:
        sigma = _as_float_array("volatility", volatility)
        if np.any(sigma < 0.0):
            raise ValueError("volatility must be non-negative")
        return sigma

    def _validate_variance(self, variance: float | np.ndarray) -> np.ndarray:
        variance_array = _as_float_array("variance", variance)
        if np.any(variance_array < 0.0):
            raise ValueError("variance must be non-negative")
        return variance_array

    def _forward_array(self, maturity: float, spot: np.ndarray) -> np.ndarray:
        return spot * np.exp((self.r - self.q) * maturity)

    def _d_terms(
        self, maturity: float, spot: np.ndarray, volatility: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        total_std = volatility * np.sqrt(maturity)
        forward = self._forward_array(maturity, spot)
        d1 = (
            np.log(forward / self.k) + 0.5 * np.square(volatility) * maturity
        ) / total_std
        d2 = d1 - total_std
        return d1, d2

    def _price_from_volatility(
        self,
        th: float,
        s: float | np.ndarray,
        volatility: float | np.ndarray,
        *,
        is_call: bool,
    ) -> np.ndarray:
        maturity = self._validate_maturity(th)
        spot, sigma = np.broadcast_arrays(
            self._validate_spot(s), self._validate_volatility(volatility)
        )
        spot = np.atleast_1d(spot)
        sigma = np.atleast_1d(sigma)
        dividend_df = self.dividend_discount(maturity)
        riskfree_df = self.mkt.discount_factor(maturity)
        if maturity == 0.0:
            payoff = (
                np.maximum(spot - self.k, 0.0)
                if is_call
                else np.maximum(self.k - spot, 0.0)
            )
            return payoff.astype(float)

        forward_discounted = spot * dividend_df
        strike_discounted = self.k * riskfree_df
        deterministic = (
            np.maximum(forward_discounted - strike_discounted, 0.0)
            if is_call
            else np.maximum(strike_discounted - forward_discounted, 0.0)
        )
        result = deterministic.astype(float)

        total_std = sigma * np.sqrt(maturity)
        regular = (total_std > _NEAR_ZERO_STD) & (spot > 0.0)
        if np.any(regular):
            d1, d2 = self._d_terms(maturity, spot[regular], sigma[regular])
            if is_call:
                result[regular] = spot[regular] * dividend_df * special.ndtr(
                    d1
                ) - self.k * riskfree_df * special.ndtr(d2)
            else:
                result[regular] = self.k * riskfree_df * special.ndtr(-d2) - spot[
                    regular
                ] * dividend_df * special.ndtr(-d1)
        return result

    def _delta_from_volatility(
        self,
        th: float,
        s: float | np.ndarray,
        volatility: float | np.ndarray,
        *,
        is_call: bool,
    ) -> np.ndarray:
        maturity = self._validate_maturity(th)
        spot, sigma = np.broadcast_arrays(
            self._validate_spot(s), self._validate_volatility(volatility)
        )
        spot = np.atleast_1d(spot)
        sigma = np.atleast_1d(sigma)
        dividend_df = self.dividend_discount(maturity)
        forward = self._forward_array(maturity, spot)
        deterministic = self._deterministic_delta(forward, dividend_df, is_call=is_call)
        if maturity == 0.0:
            return deterministic

        result = deterministic.astype(float)
        total_std = sigma * np.sqrt(maturity)
        regular = (total_std > _NEAR_ZERO_STD) & (spot > 0.0)
        if np.any(regular):
            d1, _ = self._d_terms(maturity, spot[regular], sigma[regular])
            if is_call:
                result[regular] = dividend_df * special.ndtr(d1)
            else:
                result[regular] = dividend_df * (special.ndtr(d1) - 1.0)
        return result

    def _deterministic_delta(
        self, forward: np.ndarray, dividend_df: float, *, is_call: bool
    ) -> np.ndarray:
        greater = forward > self.k
        lower = forward < self.k
        kink = self._at_kink(forward)
        if is_call:
            return np.where(
                greater, dividend_df, np.where(kink, 0.5 * dividend_df, 0.0)
            )
        return np.where(lower, -dividend_df, np.where(kink, -0.5 * dividend_df, 0.0))

    def _at_kink(self, forward: np.ndarray) -> np.ndarray:
        return np.isclose(
            forward,
            self.k,
            rtol=0.0,
            atol=_KINK_ATOL_SCALE * max(float(self.k), 1.0),
        )
