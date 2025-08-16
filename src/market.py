"""Market module.

Provides a simple representation of a financial market.  The class is
implemented as a dataclass to clearly express its data attributes and to keep
the model immutable once created.  This tiny abstraction allows other
components to depend on the ``Market`` interface rather than concrete
implementations, following the Dependency Inversion Principle of SOLID.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Market:
    """Container for market parameters.

    Attributes
    ----------
    r:
        Constant risk-free interest rate.
    """

    r: float

    def D(self, th: float) -> float:
        """Discount factor for maturity ``th``.

        Parameters
        ----------
        th:
            Time to maturity.

        Returns
        -------
        float
            Present value discount factor.
        """

        return np.exp(-self.r * th)
