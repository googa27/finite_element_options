import pydantic as pyd
import streamlit as st
import numpy as np
import aleatory.processes as alp

from .config import EPS


class DynamicsParametersHeston(pyd.BaseModel):
    r: float
    q: float
    kappa: float
    theta: float
    sig: float
    rho: float

    def cir_number(self) -> float:
        return 2*self.kappa*self.theta/self.sig**2

    def write_cir(self) -> None:
        with st.sidebar:
            st.write(f"CIR Parameter (must be greater than 1): {self.cir_number():.2f}")

    def mean_variance(self, th, v):
        x = self.kappa*th + EPS
        return -np.expm1(-x)/x*(v - self.theta) + self.theta

    # def mean_variance(self, th, v):
    #     return v + th*0

    # def mean_variance(self, th, v):
    #     return (alp.CIRProcess(theta=self.kappa,
    #                            mu=self.theta,
    #                            sigma=self.sig,
    #                            initial=v)
    #             .get_marginal(t=th + EPS)
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
