"""Black-Scholes vanilla option pricing utilities."""

from dataclasses import dataclass

import numpy as np
import scipy.stats as spst

from src.market import Market


@dataclass(frozen=True)
class EuropeanOptionBs:
    """European option priced with the Black-Scholes model.

    Parameters are stored as attributes via ``dataclass`` which promotes a more
    declarative and immutable design.  All computations are performed through
    instance methods operating solely on the provided state and arguments.
    """

    k: float
    q: float
    mkt: Market

    @property
    def r(self) -> float:
        """Risk-free rate of the associated market."""

        return self.mkt.r

    def dividend_discount(self, th: float) -> float:
        """Dividend discount factor ``exp(-q * th)``."""

        return np.exp(-self.q * th)

    def forward_price(self, th: float, s: float) -> float:
        """Forward price of the underlying asset."""

        return s * self.dividend_discount(th) / self.mkt.discount_factor(th)

    def d1(self, th: float, s: float, v: float) -> float:
        """Black-Scholes ``d1`` auxiliary term."""

        return (
            np.log(self.forward_price(th, s) / self.k) + v / 2 * th
        ) / (v * th) ** 0.5

    def d2(self, th: float, s: float, v: float) -> float:
        """Black-Scholes ``d2`` auxiliary term."""

        return self.d1(th, s, v) - (v * th) ** 0.5

    def call(self, th: float, s: float, v: float) -> float:
        """Price of the European call option."""

        n1 = spst.norm.cdf(self.d1(th, s, v))
        n2 = spst.norm.cdf(self.d2(th, s, v))
        return self.mkt.discount_factor(th) * (
            self.forward_price(th, s) * n1 - self.k * n2
        )

    def call_payoff(self, s: float) -> float:
        """Intrinsic value of a call option at spot price ``s``."""

        return np.maximum(0.0, s - self.k)

    def call_delta(self, th: float, s: float, v: float) -> float:
        """Delta of the call option."""

        n1 = spst.norm.cdf(self.d1(th, s, v))
        return self.dividend_discount(th) * n1

    def put(self, th: float, s: float, v: float) -> float:
        """Price of the European put option."""

        n1 = spst.norm.cdf(-self.d1(th, s, v))
        n2 = spst.norm.cdf(-self.d2(th, s, v))
        return self.mkt.discount_factor(th) * (
            self.k * n2 - self.forward_price(th, s) * n1
        )

    def put_payoff(self, s: float) -> float:
        """Intrinsic value of a put option at spot price ``s``."""

        return np.maximum(0.0, self.k - s)

    def put_delta(self, th: float, s: float, v: float) -> float:
        """Delta of the put option."""

        n1 = spst.norm.cdf(-self.d1(th, s, v))
        return -self.dividend_discount(th) * n1

    def vega(self, th: float, s: float, v: float) -> float:
        """Vega of the option."""

        dn2 = spst.norm.pdf(self.d2(th, s, v))
        return self.k * self.mkt.discount_factor(th) * dn2 * th ** 0.5
