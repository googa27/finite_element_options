import streamlit as st
import numpy as np


class Market:

    def __init__(self, r: float):
        self.r = r

    def D(self, th: float):
        return np.exp(-self.r*th)
