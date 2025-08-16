import streamlit as st
import numpy as np

import src.sidebar as sdb
import src.plots as splt
from src.space import SpaceSolver
from src.time import ThetaScheme

prm = sdb.Sidebar()
prm.dh.write_cir()

space = SpaceSolver(prm)
t = prm.t
dt = t[1] - t[0]
stepper = ThetaScheme(prm.lam)

with st.sidebar:
    print(f'Number of degrees of freedom: {space.Vh.N}')

v_tsv = np.empty((t.shape[0], space.Vh.N))
v_tsv[0] = space.v0

for i, th_i in enumerate(t[:-1]):
    v_next = stepper.step(space, v_tsv[i], th_i, dt)
    if prm.is_american:
        v_next = np.maximum(v_next, v_tsv[0])
    v_tsv[i + 1] = v_next

u = v_tsv[-1]
u_s = space.Vh.project(space.Vh.interpolate(u).grad[0])
u_ss = space.Vh.project(space.Vh.interpolate(u_s).grad[0])
u_v = space.Vh.project(space.Vh.interpolate(u).grad[1])

plot_data = {
    'Option Value': u,
    'Delta': u_s,
    'Gamma': u_ss,
    'Variance Vega': u_v,
}

for title, f_sv in plot_data.items():
    splt.plot_2d(space.Vh, f_sv, title)
