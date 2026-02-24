from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from bspx.types import OptionType

_F64 = NDArray[np.float64]

_to_f64: Callable[[ArrayLike], _F64] = partial(np.asarray, dtype=np.float64)


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
    if np.any(np.asarray(T) < 0):
        raise ValueError(f"Error: Time to maturity 'T' must be nonnegative\n Got: {T}")
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

    S: _F64
    K: _F64
    T: _F64
    r: _F64
    vol: _F64

    d1: _F64
    d2: _F64
    cdf_d1: _F64
    cdf_d2: _F64
    cdf_nd1: _F64
    cdf_nd2: _F64
    pdf_d1: _F64

    sqrt_t: _F64
    discount: _F64
    vol_sqrt_t: _F64

    @classmethod
    def build(
        cls, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, vol: ArrayLike
    ) -> "BlackScholesState":
        S_, K_, T_, r_, vol_ = map(_to_f64, (S, K, T, r, vol))
        _validate_inputs(S_, K_, T_, vol_)

        sqrt_t = np.sqrt(T_)
        vol_sqrt_t = vol_ * sqrt_t
        safe_vol_sqrt_t = np.where(T_ > 0, vol_sqrt_t, 1.0)

        d1 = (np.log(S_ / K_) + (r_ + 0.5 * np.square(vol_)) * T_) / safe_vol_sqrt_t
        d1 = np.where(T_ > 0, d1, np.where(S_ >= K_, 1e10, 1e-10))
        d2 = d1 - vol_sqrt_t

        return cls(
            S=S_,
            K=K_,
            T=T_,
            r=r_,
            vol=vol_,
            d1=d1,
            d2=d2,
            cdf_d1=norm.cdf(d1),
            cdf_d2=norm.cdf(d2),
            pdf_d1=norm.pdf(d1),
            cdf_nd1=norm.cdf(-d1),
            cdf_nd2=norm.cdf(-d2),
            sqrt_t=sqrt_t,
            discount=np.exp(-r_ * T_),
            vol_sqrt_t=vol_sqrt_t,
        )

    def call_price(self) -> _F64:
        payoff = np.maximum(self.S - self.K * self.discount, 0.0)
        price = self.S * self.cdf_d1 - self.K * self.discount * self.cdf_d2
        return np.where(self.T > 0, price, payoff)

    def put_price(self) -> _F64:
        payoff = np.maximum(self.K * self.discount - self.S, 0.0)
        price = self.K * self.discount * self.cdf_nd2 - self.S * self.cdf_nd1
        return np.where(self.T > 0, price, payoff)


def build_black_scholes_state(*args, **kwargs) -> BlackScholesState:
    return BlackScholesState.build(*args, **kwargs)


def black_scholes_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    vol: ArrayLike,
    option_type: OptionType = "call",
) -> _F64:
    state = BlackScholesState.build(S, K, T, r, vol)

    match option_type:
        case "call":
            return state.call_price()
        case "put":
            return state.put_price()
