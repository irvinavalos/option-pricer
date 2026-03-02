import numpy as np
from numpy.typing import ArrayLike

from bspx.numeric_utils import to_f64
from bspx.types import _F64


def forward_price(
    S: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
) -> _F64:
    """
    Risk-neutral forward price of the underlying asset:
    F = S * exp((r - q) * T)

    Parameters:
        S:  Asset price
        T:  Time to expiration (in years)
        r:  Annual risk-free rate
        q:  Dividend yield (defaults to 0 for non-dividend paying assets)
    """
    (S_, T_, r_, q_) = map(to_f64, (S, T, r, q))
    return S_ * np.exp((r_ - q_) * T_)
