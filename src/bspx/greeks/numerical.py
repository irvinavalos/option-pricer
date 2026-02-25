from collections.abc import Callable
from functools import partial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bspx.types import DayCount, OptionType, PricingFunction

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
    S_ = _to_f64(S)
    h = S_ * BUMP

    price_up = pricing_func(S_ + h, K, T, r, vol, option_type)
    price_down = pricing_func(S_ - h, K, T, r, vol, option_type)

    return (price_up - price_down) / (2 * h)


def theta_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
    day_count: DayCount = DayCount.CALENDAR,
) -> _F64:
    T_ = _to_f64(T)
    h = T_ * BUMP

    price_up = pricing_func(S, K, T_ + h, r, vol, option_type)
    price_down = pricing_func(S, K, T_ - h, r, vol, option_type)

    return -(price_up - price_down) / (2 * h * day_count)


def gamma_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
) -> _F64:
    S_ = _to_f64(S)
    h = S_ * BUMP

    price_up = pricing_func(S_ + h, K, T, r, vol, "call")
    price_down = pricing_func(S_ - h, K, T, r, vol, "call")
    price_curr = pricing_func(S_, K, T, r, vol, "call")

    return (price_up + price_down - 2 * price_curr) / np.square(h)


def vega_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
) -> _F64:
    vol_ = _to_f64(vol)
    h = vol_ * BUMP

    price_up = pricing_func(S, K, T, r, vol_ + h, "call")
    price_down = pricing_func(S, K, T, r, vol_ - h, "call")

    return (price_up - price_down) / (2 * h)


def rho_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> _F64:
    r_ = _to_f64(r)
    h = np.maximum(np.abs(r_), 0.01) * BUMP

    price_up = pricing_func(S, K, T, r_ + h, vol, option_type)
    price_down = pricing_func(S, K, T, r_ - h, vol, option_type)

    return (price_up - price_down) / (2 * h)
