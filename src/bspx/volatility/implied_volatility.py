from typing import cast

import numpy as np
from scipy.optimize import brentq, newton

from bspx.greeks.analytical import vega
from bspx.pricing.black_scholes_model import build_black_scholes_state
from bspx.pricing.forwards import forward_price
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
    _F = float(forward_price(S, T, r))
    return np.sqrt(2 * np.abs(np.log(_F / K)) / T)


def _vega_scalar(
    vol: float,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> float:
    state = build_black_scholes_state(S, K, T, r, vol)
    return float(vega(state))


def _check_no_arbitrage(
    market_price: float, S: float, K: float, T: float, r: float, option_type: OptionType
) -> None:
    discount = K * np.exp(-r * T)
    intrinsic = (
        max(S - discount, 0.0) if option_type == "call" else max(discount - S, 0.0)
    )
    if market_price < intrinsic:
        raise ValueError(
            f"Market Price: {market_price:.4f} violates no arbitrage bounds"
            f"Intrinsic Value: {intrinsic:.4f} undefined"
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
    try:
        root = brentq(
            f=_objective_func,
            a=vol_lo,
            b=vol_hi,
            args=(market_price, S, K, T, r, option_type),
            xtol=tol,
            full_output=False,
        )
        return cast(float, root)
    except RuntimeError:
        return implied_vol_brent(market_price, S, K, T, r, option_type, tol)
