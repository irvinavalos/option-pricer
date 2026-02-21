from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bspx.constants import CALENDAR_DAYS_PER_YEAR
from bspx.types import OptionType, PricingFunction

_F64 = NDArray[np.float64]

_to_f64: Callable[[ArrayLike], _F64] = partial(np.asarray, dtype=np.float64)

BUMP = 1e-4


def delta_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> _F64:
    _S = _to_f64(S)
    h = _S * BUMP

    price_up = pricing_func(_S + h, K, T, r, vol, option_type)
    price_down = pricing_func(_S - h, K, T, r, vol, option_type)

    return (price_up - price_down) / (2 * h)


def theta_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> _F64:
    _T = _to_f64(T)
    h = _T * BUMP

    price_up = pricing_func(S, K, _T + h, r, vol, option_type)
    price_down = pricing_func(S, K, _T - h, r, vol, option_type)

    return -(price_up - price_down) / (2 * h * CALENDAR_DAYS_PER_YEAR)


def gamma_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> _F64:
    _S = _to_f64(S)
    h = _S * BUMP

    price_up = pricing_func(_S + h, K, T, r, vol, option_type)
    price_down = pricing_func(_S - h, K, T, r, vol, option_type)
    price_curr = pricing_func(_S, K, T, r, vol, option_type)

    return (price_up + price_down - 2 * price_curr) / np.square(h)


def vega_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> _F64:
    _vol = _to_f64(vol)
    h = _vol * BUMP

    price_up = pricing_func(S, K, T, r, _vol + h, option_type)
    price_down = pricing_func(S, K, T, r, _vol - h, option_type)

    return (price_up - price_down) / (2 * h * 100)


def rho_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> _F64:
    _r = _to_f64(r)
    h = np.maximum(np.abs(_r), 0.01) * BUMP

    price_up = pricing_func(S, K, T, _r + h, vol, option_type)
    price_down = pricing_func(S, K, T, _r - h, vol, option_type)

    return (price_up - price_down) / (2 * h * 100)
