import streamlit as st
import numpy as np
import skfem as fem

import src.sidebar as sdb
import src.plots as splt
import CONFIG as CFG

from skfem.visuals.matplotlib import plot as femplot
from src.space.solver import SpaceSolver
from src.time.stepper import ThetaScheme

prm = sdb.Sidebar()

dynh = prm.dh
mkt = prm.mkt
bsopt = prm.bsopt
t = prm.t

space = SpaceSolver(prm.mesh, dynh, bsopt, is_call=prm.is_call)
stepper = ThetaScheme(theta=prm.lam)

with st.sidebar:
    st.write(dynh.cir_message())

v_tsv = stepper.solve(
    t,
    space,
    dirichlet_bcs=prm.dirichlet_bcs,
    is_american=prm.is_american,
)


Th = space.mesh
Vh = space.Vh
with st.sidebar:
    st.write(f'Number of degrees of freedom: {Vh.N}')

# use solution from earlier call to the time stepper
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
