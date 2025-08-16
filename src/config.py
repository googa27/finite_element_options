"""Shared configuration constants for the finite element solvers."""

import skfem as fem

# Numerical tolerance used throughout the spatial discretisation
EPS: float = 1e-10

# Default finite element used when constructing basis functions
ELEM: fem.Element = fem.ElementTriP2()

