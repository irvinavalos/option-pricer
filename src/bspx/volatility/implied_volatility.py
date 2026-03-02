from typing import cast

import numpy as np
from scipy.optimize import brentq, newton

from bspx.greeks import AnalyticalBackend
from bspx.pricing import build_black_scholes_state, forward_price
from bspx.types import OptionType


def _objective_func(
    vol: float,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
) -> float:
    """Price of model - Market price for a specific volatility"""
    state = build_black_scholes_state(S, K, T, r, vol)
    price = state.call_price() if option_type == "call" else state.put_price()

    return float(price) - market_price


def _manaster_koehler(S: float, K: float, T: float, r: float) -> float:
    F_ = float(forward_price(S, T, r))

    return np.sqrt(2 * np.abs(np.log(F_ / K)) / T)


def _vega_scalar(
    vol: float,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> float:
    _ = (market_price, option_type)
    state = build_black_scholes_state(S, K, T, r, vol)
    backend = AnalyticalBackend(state)
    return float(backend.vega())


def _check_no_arbitrage(
    market_price: float, S: float, K: float, T: float, r: float, option_type: OptionType
) -> None:
    df = np.exp(-r * T)

    match option_type:
        case "call":
            lower_bound = max(S - K * df, 0.0)
            upper_bound = S
        case "put":
            lower_bound = max(K * df - S, 0.0)
            upper_bound = K * df

    if market_price < lower_bound or market_price > upper_bound:
        raise ValueError(
            f"Market Price {market_price:.4f} is outside no arbitrage bounds "
            f"[{lower_bound:.4f}, {upper_bound:.4f}] IV is undefined"
        )


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Computing implied volatility using Newton-Raphson with Manaster-Koehler startpoint

    - Uses analytical Vega as the derivative
    - Manaster-Koehler result ensures the initial volatility lies in a well behaved region of the Black-Scholes pricing function
    - Raises ValueError if convergence fails, use 'implied_vol_brent' as fallback
    """
    _check_no_arbitrage(market_price, S, K, T, r, option_type)

    vol_0 = _manaster_koehler(S, K, T, r)
    root = newton(
        func=_objective_func,
        x0=vol_0,
        fprime=_vega_scalar,
        args=(market_price, S, K, T, r, option_type),
        tol=tol,
        maxiter=max_iter,
        full_output=False,
    )

    return cast(float, root)


def implied_vol_brent(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    vol_lo: float = 1e-4,
    vol_hi: float = 10.0,
) -> float:
    """
    Computing implied volatility using Brent's method

    - Guaranteed to converge on interval [vol_lo, vol_hi]
    - Slower than Newton-Raphson but is able to handle cases where vega tends to zero
    """
    _check_no_arbitrage(market_price, S, K, T, r, option_type)

    root = brentq(
        f=_objective_func,
        a=vol_lo,
        b=vol_hi,
        args=(market_price, S, K, T, r, option_type),
        xtol=tol,
        full_output=False,
    )

    return cast(float, root)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
) -> float:
    try:
        return implied_vol_newton(market_price, S, K, T, r, option_type)
    except RuntimeError:
        return implied_vol_brent(market_price, S, K, T, r, option_type)
