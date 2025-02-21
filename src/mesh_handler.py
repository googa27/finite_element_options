import streamlit as st
import skfem as fem
import skfem.helpers as fh
import numpy as np
import aleatory.processes as alp
import scipy.stats as spst

import CONFIG as CFG


class MeshHandler:

    def __init__(self, dynh, bsopt, opt_param):
        self.init_refine = st.slider('mesh refine', 1, 5, 2)

        neg_log_alpha = st.slider(r'neg log - $\alpha$', 1., 3., 1.)
        self.lam = st.slider(r'$\lambda$', 0., 1., 0.5)
        v_max = st.slider(r'$v_{max}$', theta, 5*theta, 2*theta)

        v_dist = (alp.CIRProcess(theta=kappa,
                                 mu=theta,
                                 sigma=sig,
                                 initial=v_max)
                  .get_marginal(T))

        alpha = 10**(-self.neg_log_alpha)
        z_a = spst.norm.ppf(1 - alpha)

        v_threshold = v_dist.ppf(1 - alpha)

        r1 = r - q - v_threshold/2
        r2 = r - q + v_threshold/2

        x_max = z_a*np.sqrt(v_threshold*T) - min(r1, r2)*T
        self.s_max = k*np.exp(x_max)

    def _eval_estimator(m, u):
        fbasis = [fem.InteriorFacetBasis(m, CFG.ELEM, side=i) for i in [0, 1]]
        w = {'u' + str(i + 1): fbasis[i].interpolate(u) for i in [0, 1]}

        @fem.Functional
        def edge_jump(w):
            h = w.h
            n = w.n
            dw1 = fh.grad(w['u1'])
            dw2 = fh.grad(w['u2'])
            return h * ((dw1[0] - dw2[0]) * n[0] +
                        (dw1[1] - dw2[1]) * n[1]) ** 2

        eta_E = edge_jump.elemental(fbasis[0], **w)

        tmp = np.zeros(m.facets.shape[1])
        np.add.at(tmp, fbasis[0].find, eta_E)
        eta_E = np.sum(.5 * tmp[m.t2f], axis=0)

        return eta_E

    def mesh_init(self, params):
        self.mesh = (fem.MeshTri()
                     .init_tensor(x=np.linspace(0, self.s_max, 2),
                                  y=np.linspace(0, self.v_max, 2)
                                  )
                     .refined(init_refine)
                     .with_boundaries({'s_min': lambda x: x[0] == 0,
                                      's_max': lambda x: x[0] == self.s_max,
                                       'v_min': lambda x: x[1] == 0,
                                       'v_max': lambda x: x[1] == self.v_max
                                       }
                                      )
                     )

    def adapt_mesh(m, u):
        pass
