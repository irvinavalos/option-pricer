import numpy as np
from numpy.typing import ArrayLike

from bspx.numeric_utils import to_f64
from bspx.types import _F64, DayCount, OptionType, PricingFunction

# TODO: Add Automatic Differentiation (AD) via PyTorch or JAX for increased performance
# on Greeks computation.

BUMP = 1e-4
MIN_BUMP = 1e-6
CALL_OPTION = "call"


def _to_f64_with_step(x: ArrayLike) -> tuple[_F64, _F64]:
    x = to_f64(x)
    return x, np.maximum(np.abs(x) * BUMP, MIN_BUMP)


def delta_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> _F64:
    S_, h = _to_f64_with_step(S)
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
    T_, h = _to_f64_with_step(T)
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
    S_, h = _to_f64_with_step(S)

    # Note: Gamma is identical for call and put options due to Put-Call parity
    price_up = pricing_func(S_ + h, K, T, r, vol, CALL_OPTION)
    price_down = pricing_func(S_ - h, K, T, r, vol, CALL_OPTION)
    price_curr = pricing_func(S_, K, T, r, vol, CALL_OPTION)
    return (price_up + price_down - 2 * price_curr) / np.square(h)


def vega_fd(
    pricing_func: PricingFunction,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
) -> _F64:
    vol_, h = _to_f64_with_step(vol)

    # Note: Vega is identical for call and put options due to Put-Call parity
    price_up = pricing_func(S, K, T, r, vol_ + h, CALL_OPTION)
    price_down = pricing_func(S, K, T, r, vol_ - h, CALL_OPTION)
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
    r_, h = _to_f64_with_step(r)
    price_up = pricing_func(S, K, T, r_ + h, vol, option_type)
    price_down = pricing_func(S, K, T, r_ - h, vol, option_type)
    return (price_up - price_down) / (2 * h)
