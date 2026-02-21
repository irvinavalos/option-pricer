from bspx.constants import TRADING_DAYS_PER_YEAR
from bspx.instruments import Greeks
from bspx.pricing import BlackScholesState
from bspx.types import ArrayLike, OptionType


def delta(state: BlackScholesState, option_type: OptionType) -> ArrayLike:
    match option_type:
        case "call":
            return state.cdf_d1
        case "put":
            return state.cdf_d1 - 1


def theta(state: BlackScholesState, option_type: OptionType) -> ArrayLike:
    decay = (-state.S * state.pdf_d1 * state.vol) / (2 * state.sqrt_t)
    discount = state.K * state.discount

    match option_type:
        case "call":
            return (decay - state.r * discount * state.cdf_d2) / TRADING_DAYS_PER_YEAR
        case "put":
            return (decay + state.r * discount * state.cdf_nd2) / TRADING_DAYS_PER_YEAR


def gamma(state: BlackScholesState) -> ArrayLike:
    return state.pdf_d1 / (state.S * state.vol_sqrt_t)


def vega(state: BlackScholesState) -> ArrayLike:
    return state.S * state.sqrt_t * state.pdf_d1 / 100


def rho(state: BlackScholesState, option_type: OptionType) -> ArrayLike:
    scale = state.K * state.T * state.discount

    match option_type:
        case "call":
            return scale * state.cdf_d2 / 100
        case "put":
            return -scale * state.cdf_nd2 / 100


def calculate_greeks(
    state: BlackScholesState,
    option_type: OptionType,
) -> Greeks:
    return Greeks(
        delta=delta(state, option_type),
        theta=theta(state, option_type),
        gamma=gamma(state),
        vega=vega(state),
        rho=rho(state, option_type),
    )
