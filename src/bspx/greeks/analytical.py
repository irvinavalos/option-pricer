import numpy as np
from numpy.typing import NDArray

from bspx.instruments import Greeks
from bspx.pricing import BlackScholesState
from bspx.types import DayCount, OptionType

_F64 = NDArray[np.float64]


def delta(state: BlackScholesState, option_type: OptionType) -> _F64:
    match option_type:
        case "call":
            return state.cdf_d1
        case "put":
            return state.cdf_d1 - 1


def theta(
    state: BlackScholesState,
    option_type: OptionType,
    day_count: DayCount = DayCount.CALENDAR,
) -> _F64:
    decay = (-state.S * state.pdf_d1 * state.vol) / (2 * state.sqrt_t)
    discount = state.K * state.discount

    match option_type:
        case "call":
            return (decay - state.r * discount * state.cdf_d2) / day_count
        case "put":
            return (decay + state.r * discount * state.cdf_nd2) / day_count


def gamma(state: BlackScholesState) -> _F64:
    return state.pdf_d1 / (state.S * state.vol_sqrt_t)


def vega(state: BlackScholesState) -> _F64:
    return state.S * state.sqrt_t * state.pdf_d1


def rho(state: BlackScholesState, option_type: OptionType) -> _F64:
    scale = state.K * state.T * state.discount

    match option_type:
        case "call":
            return scale * state.cdf_d2
        case "put":
            return -scale * state.cdf_nd2


def calculate_greeks(
    state: BlackScholesState,
    option_type: OptionType,
    day_count: DayCount = DayCount.CALENDAR,
) -> Greeks:
    return Greeks(
        delta=delta(state, option_type),
        theta=theta(state, option_type, day_count),
        gamma=gamma(state),
        vega=vega(state),
        rho=rho(state, option_type),
    )
