import matplotlib.pyplot as plt
import streamlit as st
import skfem.visuals as femv

from skfem.visuals.matplotlib import plot as femplot


def plot_mean_variance(t, dynh):

    fig, ax = plt.subplots()
    ax.plot(t, dynh.mean_variance(t, 2*dynh.theta))
    ax.hlines(y=dynh.theta,
              xmin=t[0], xmax=t[-1],
              linestyles='--',
              color='black',
              linewidth=3,
              label='Theta')
    ax.hlines(y=2*dynh.theta,
              xmin=t[0], xmax=t[-1],
              linestyles='--',
              color='red',
              linewidth=3,
              label='v0')
    ax.set_ylabel('mean variance')
    ax.set_xlabel('time to maturity')
    ax.set_title('Heston mean variance')
    ax.legend()
    st.pyplot(fig)


def plot_2d(Vh, f_sv, title: str) -> None:
    fig, ax = plt.subplots()
    femv.matplotlib.plot(Vh, f_sv, ax=ax, colorbar=True)
    ax.set_xlabel('underlying')
    ax.set_ylabel('variance')
    ax.set_title(title)
    st.pyplot(fig)
