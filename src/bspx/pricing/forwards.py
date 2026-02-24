from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

_F64 = NDArray[np.float64]

_to_f64: Callable[[ArrayLike], _F64] = partial(np.asarray, dtype=np.float64)


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
    (S_, T_, r_, q_) = map(_to_f64, (S, T, r, q))
    return S_ * np.exp((r_ - q_) * T_)
