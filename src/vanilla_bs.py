import numpy as np
import pydantic as pyd
import scipy.stats as spst

import src.market as mk
import CONFIG as CFG


class EuropeanOptionBs:

    def __init__(self,
                 k: float,
                 q: float,
                 mkt: mk.Market):
        self.k: float = k
        self.q: float = q
        self.mkt: mk.Market = mkt

    @property
    def r(self):
        return self.mkt.r

    def Q(self, th):
        return np.exp(-self.q*th)

    def F(self, th, s):
        return s*self.Q(th)/self.mkt.D(th)

    def d1(self, th, s, v):
        return (np.log(self.F(th, s)/self.k) + v/2*th)/(v*th)**0.5

    def d2(self, th, s, v):
        return self.d1(th, s, v) - (v*th)**0.5

    def call(self, th, s, v):
        n1 = spst.norm.cdf(self.d1(th, s, v))
        n2 = spst.norm.cdf(self.d2(th, s, v))
        return self.mkt.D(th)*(self.F(th, s)*n1 - self.k*n2)

    def call_payoff(self, s):
        return np.maximum(0, s - self.k)

    def call_delta(self, th, s, v):
        n1 = spst.norm.cdf(self.d1(th, s, v))
        return self.Q(th)*n1

    def put(self, th, s, v):
        n1 = spst.norm.cdf(- self.d1(th, s, v))
        n2 = spst.norm.cdf(- self.d2(th, s, v))
        return self.mkt.D(th)*(self.k*n2 - self.F(th, s)*n1)

    def put_payoff(self, s):
        return np.maximum(0, self.k - s)

    def put_delta(self, th, s, v):
        n1 = spst.norm.cdf(-self.d1(th, s, v))
        return -self.Q(th)*n1

    def vega(self, th, s, v):
        dn2 = spst.norm.pdf(self.d2(th, s, v))
        return self.k*self.mkt.D(th)*dn2*th**0.5
