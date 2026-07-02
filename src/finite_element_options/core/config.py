"""Project-wide numerical configuration."""

from dataclasses import dataclass
import skfem as fem


@dataclass
class Config:
    """Configuration parameters for numerical schemes.

    Attributes
    ----------
    eps:
        Small positive number to avoid division by zero.
    elem:
        Finite element used for spatial discretization.
    """
    eps: float = 1e-10
    elem: fem.Element = fem.ElementTriP2()
