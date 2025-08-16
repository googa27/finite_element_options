import streamlit as st
import numpy as np
import skfem as fem
import skfem.helpers as fhl
import skfem.visuals as femv
import matplotlib.pyplot as plt
import aleatory.processes as alp

import src.sidebar as sdb
import src.forms as frm
import src.plots as splt
import CONFIG as CFG

from skfem.visuals.matplotlib import plot as femplot

prm = sdb.Sidebar()

dynh = prm.dh
mkt = prm.mkt
bsopt = prm.bsopt
t = prm.t
dt = t[1] - t[0]

dynh.write_cir()


def initialize_value_array(t, Vh, bsopt, is_call):
    v_tsv = np.empty((t.shape[0], Vh.N))
    v_tsv[0] = Vh.project(lambda x: bsopt.call_payoff(x[0])*is_call
                          + bsopt.put_payoff(x[0])*(not is_call))
    return v_tsv


# splt.plot_mean_variance(t, dynh)

#################
### MAIN LOOP ###
#################


def solve_system(Th):
    Vh = fem.CellBasis(Th, CFG.ELEM)
    dVh = fem.FacetBasis(Th, CFG.ELEM)

    v_tsv = initialize_value_array(t, Vh, bsopt, is_call=prm.is_call)

    forms = frm.Forms(prm)
    I = forms.id_bil().assemble(Vh)
    L = forms.l_bil().assemble(Vh)
    A = I - prm.lam*dt*L
    B = A + dt*L

    for i, th_i in enumerate(t[:-1]):
        b_previous = B@v_tsv[i]
        b_inhom = (prm.lam * forms.b_lin().assemble(dVh, th=th_i + dt)
                   + (1 - prm.lam) * forms.b_lin().assemble(dVh, th=th_i)
                   )
        b = b_previous + dt*(b_inhom)

        if prm.dirichlet_bcs:
            # Project boundary values and enforce Dirichlet conditions
            u_dirichlet = Vh.project(
                lambda x: bsopt.call(
                    th_i, x[0], dynh.mean_variance(th_i, x[1])
                )
                if prm.is_call
                else bsopt.put(
                    th_i, x[0], dynh.mean_variance(th_i, x[1])
                )
            )
            A, b = fem.enforce(
                A, b, x=u_dirichlet, D=Vh.get_dofs(prm.dirichlet_bcs)
            )

        v_tsv[i + 1] = fem.solve(A, b)

        if prm.is_american:
            v_tsv[i + 1] = np.maximum(v_tsv[i + 1], v_tsv[0])

    return v_tsv


Th = prm.mesh
Vh = fem.CellBasis(Th, CFG.ELEM)
with st.sidebar:
    print(f'Number of degrees of freedom: {Vh.N}')


v_tsv = solve_system(Th)

#####################
### END MAIN LOOP ###
#####################

u = v_tsv[-1]
u_s = Vh.project(Vh.interpolate(u).grad[0])
u_ss = Vh.project(Vh.interpolate(u_s).grad[0])
u_v = Vh.project(Vh.interpolate(u).grad[1])

plot_data: dict = {'Option Value': u,
                   'Delta': u_s,
                   'Gamma': u_ss,
                   'Variance Vega': u_v}

for title, f_sv in plot_data.items():
    splt.plot_2d(Vh, f_sv, title)
