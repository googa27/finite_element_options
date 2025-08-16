import streamlit as st
import numpy as np
import scipy.stats as spst
import aleatory.processes as alp

from src.core.dynamics_heston import DynamicsParametersHeston
from src.core.market import Market
from src.core.vanilla_bs import EuropeanOptionBs
from src.core.mesh import create_rectangular_mesh


class Sidebar:

    def __init__(self):
        self._make_sidebar()

    def _make_sidebar(self):
        with st.sidebar:
            st.title('Option Parameters')

            self.is_american = st.checkbox('american', value=False)
            self.is_call = st.checkbox('call', value=False)

            k = st.slider('Strike', 0.1, 2., 0.4)
            T = st.slider('Maturity', 0., 5., 1.)

            st.title('Market Parameters')
            r = st.slider('r', -0.1, 0.5, 0.03)

            st.title('Boundary Conditions')
            self.dirichlet_bcs = st.multiselect('Dirichlet Boundary Conditions',
                                                ['s_min', 's_max',
                                                 'v_min', 'v_max'],
                                                [])

            st.title('Discretization Parameters')

            mesh_refine = st.slider('mesh refine', 1, 10, 5)
            # mesh_adaptive = st.slider('mesh adaptive', 1, 10, 3)
            nt = st.slider('nt', 1, 1000, 100)
            neg_log_alpha = st.slider(r'neg log - $\alpha$', 1., 3., 1.)
            self.lam = st.slider(r'$\lambda$', 0., 1., 0.5)

            st.title('Dynamics Parameters')
            q = st.slider(r'$q$', 0., 1., 0.03)
            kappa = st.slider(r'$\kappa$', 0., 1., 0.5)
            theta = st.slider(r'$\theta$', 0., 1., 0.5)
            sig = st.slider(r'$\sigma$', 0., 0.8, 0.2)
            rho = st.slider(r'$\rho$', -1., 1., 0.5)

            v_max = st.slider(r'$v_{max}$', theta, 5*theta, 2*theta)

            v_dist = (alp.CIRProcess(theta=kappa,
                                     mu=theta,
                                     sigma=sig,
                                     initial=v_max)
                      .get_marginal(T))

            alpha = 10**(-neg_log_alpha)
            z_a = spst.norm.ppf(1 - alpha)

            v_threshold = v_dist.ppf(1 - alpha)

            r1 = r - q - v_threshold/2
            r2 = r - q + v_threshold/2

            x_max = z_a*np.sqrt(v_threshold*T) - min(r1, r2)*T
            s_max = k*np.exp(x_max)

        self.dh = DynamicsParametersHeston(r=r,
                                           q=q,
                                           kappa=kappa,
                                           theta=theta,
                                           sig=sig,
                                           rho=rho)
        self.mkt = Market(r=r)
        self.bsopt = EuropeanOptionBs(k,
                                      self.dh.q,
                                      self.mkt)
        self.t = np.linspace(0, T, nt)
        self.mesh = create_rectangular_mesh(s_max, v_max, mesh_refine)
