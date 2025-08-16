"""Heston model dynamics utilities.

This module contains only numerical code and is free of any side effects or UI
logic so that it can be reused from both Streamlit and command line
interfaces.
"""

import pydantic as pyd
import numpy as np
import CONFIG as CFG


class DynamicsParametersHeston(pyd.BaseModel):
    r: float
    q: float
    kappa: float
    theta: float
    sig: float
    rho: float

    def cir_number(self) -> float:
        """Return the Cox–Ingersoll–Ross (CIR) parameter."""
        return 2 * self.kappa * self.theta / self.sig ** 2

    def cir_message(self) -> str:
        """Human readable message about the CIR parameter."""
        return f"CIR Parameter (must be greater than 1): {self.cir_number():.2f}"

    def mean_variance(self, th, v):
        x = self.kappa*th + CFG.EPS
        return -np.expm1(-x)/x*(v - self.theta) + self.theta

    # def mean_variance(self, th, v):
    #     return v + th*0

    # def mean_variance(self, th, v):
    #     return (alp.CIRProcess(theta=self.kappa,
    #                            mu=self.theta,
    #                            sigma=self.sig,
    #                            initial=v)
    #             .get_marginal(t=th + CFG.EPS)
    #             .mean()
    #             )

    def A(self, x, y) -> list[list]:
        '''
        Covariance matrix appearing in feynman kac formula.
        '''
        return [[x**2*y, self.rho*self.sig*x*y],
                [self.rho*self.sig*x*y, self.sig**2*y]]

    def dA(self, x, y) -> list:
        '''
        Divergence of A
        '''
        return [2*x*y + self.rho*self.sig*x,
                self.rho*self.sig*y + self.sig**2]

    def b(self, x, y) -> list:
        '''
        Drift vector in feynman kac formula
        '''
        return [(self.r - self.q)*x,
                self.kappa*(self.theta - y)]

    # def b(self, x, y) -> list:
    #     '''
    #     Drift vector in feynman kac formula
    #     '''
    #     return [self.r - self.q + 0*x,
    #             self.kappa*(self.theta - y)]
