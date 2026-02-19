from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from bspx.types import ArrayLike, OptionType


def _validate_inputs(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
) -> None:
    if np.any(np.asarray(S) <= 0):
        raise ValueError(f"Error: Asset price 'S' must be positive\n Got: {S}")
    if np.any(np.asarray(K) <= 0):
        raise ValueError(f"Error: Strike price 'K' must be positive\n Got: {K}")
    if np.any(np.asarray(T) <= 0):
        raise ValueError(f"Error: Time to maturity 'T' must be positive\n Got: {T}")
    if np.any(np.asarray(vol) <= 0):
        raise ValueError(f"Error: Volatility 'vol' must be positive\n Got: {vol}")


@dataclass(slots=True, frozen=True)
class BlackScholesState:
    """
    Parameters:

        S:          Asset price
        K:          Strike price
        T:          Time to expiration (in years)
        r:          Risk-free rate (annualized)
        vol:        Volatility (annualized)

        d1:         Black Scholes d1 component
        d2:         Black-Scholes d2 component
        cdf_d1:     N(d1) -- Call delta
        cdf_d2:     N(d2) -- Risk-neutral probability of finishing in the money (ITM)
        cdf_nd1:    N(-d1) -- Put delta magnitude
        cdf_nd2:    N(-d2) -- Complement of d2
        pdf_d1:     f(d1) -- PDF of the Standard Normal distribution

        sqrt_t:     sqrt(T) -- Root time
        discount:   exp{-rT} -- Discount factor
        vol_sqrt_t: vol * sqrt(T) -- Total volatility
    """

    S: ArrayLike
    K: ArrayLike
    T: ArrayLike
    r: ArrayLike
    vol: ArrayLike

    d1: ArrayLike
    d2: ArrayLike
    cdf_d1: ArrayLike
    cdf_d2: ArrayLike
    cdf_nd1: ArrayLike
    cdf_nd2: ArrayLike
    pdf_d1: ArrayLike

    sqrt_t: ArrayLike
    discount: ArrayLike
    vol_sqrt_t: ArrayLike


def build_black_scholes_state(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
) -> BlackScholesState:
    _validate_inputs(S, K, T, vol)

    sqrt_t = np.sqrt(T)
    vol_sqrt_t = vol * sqrt_t

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    return BlackScholesState(
        S=S,
        K=K,
        T=T,
        r=r,
        vol=vol,
        d1=d1,
        d2=d2,
        cdf_d1=norm.cdf(d1),
        cdf_d2=norm.cdf(d2),
        pdf_d1=norm.pdf(d1),
        cdf_nd1=norm.cdf(-d1),
        cdf_nd2=norm.cdf(-d2),
        sqrt_t=sqrt_t,
        discount=np.exp(-r * T),
        vol_sqrt_t=vol_sqrt_t,
    )


def call_price(state: BlackScholesState) -> ArrayLike:
    # c = S * N(d1) - K * e^(-rT) * N(d2)
    return state.S * state.cdf_d1 - state.K * state.discount * state.cdf_d2


def put_price(state: BlackScholesState) -> ArrayLike:
    # p = K * e^(-rT) * N(-d2) - S * N(-d1)
    return state.K * state.discount * state.cdf_nd2 - state.S * state.cdf_nd1


def black_scholes_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> ArrayLike:
    state = build_black_scholes_state(S, K, T, r, vol)

    match option_type:
        case "call":
            return call_price(state)
        case "put":
            return put_price(state)
