from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy.stats.norm as jnorm
from jax import Array
from jax.typing import ArrayLike

_MIN_VEGA = 1e-6

_MIN_VOL = 1e-5
_MAX_VOL = 5.0


def _vectorize_inputs(*args: ArrayLike) -> list[Array]:
    return [jnp.asarray(arg, dtype=jnp.float32) for arg in args]


def _check_market_inputs(
    S: Array,
    K: Array,
    t: Array,
    r: Array,
    sigma: Array | None,
    q: Array,
    market_price: Array | None = None,
) -> None:
    if jnp.any(S <= 0):
        raise ValueError("S must be positive")
    if jnp.any(K <= 0):
        raise ValueError("K must be positive")
    if jnp.any(t <= 0):
        raise ValueError("t must be positive")
    if jnp.any(r < 0):
        raise ValueError("r must be nonnegative")
    if sigma is not None and jnp.any(sigma < 0):
        raise ValueError("sigma must be nonnegative")
    if jnp.any(q < 0):
        raise ValueError("q must be nonnegative")
    if market_price is not None and jnp.any(market_price <= 0):
        raise ValueError("Market price must positive")


def _compute_d1(
    S: Array,
    K: Array,
    t: Array,
    r: Array,
    sigma: Array,
    q: Array,
) -> Array:
    return (jnp.log(S / K) + (r - q + jnp.square(sigma) / 2) * t) / (
        sigma * jnp.sqrt(t)
    )


def _price(
    S: Array,
    K: Array,
    t: Array,
    r: Array,
    sigma: Array,
    q: Array,
    option_type: Literal["call", "put"] = "call",
    check_inputs: bool = False,
) -> Array:
    if check_inputs:
        _check_market_inputs(S, K, t, r, sigma, q)

    d1 = _compute_d1(S, K, t, r, sigma, q)
    d2 = d1 - sigma * jnp.sqrt(t)

    match option_type:
        case "call":
            return (S * jnp.exp(-q * t) * jnorm.cdf(d1)) - (
                K * jnp.exp(-r * t) * jnorm.cdf(d2)
            )
        case "put":
            return (K * jnp.exp(-r * t) * jnorm.cdf(-d2)) - (
                S * jnp.exp(-q * t) * jnorm.cdf(-d1)
            )


def _manaster_koehler_init(S: Array, K: Array, t: Array, r: Array, q: Array) -> Array:
    return jnp.sqrt(jnp.abs((2 / jnp.sqrt(t)) * jnp.log(S / K) + (r - q) * t) * (1 / t))


class BlackScholes:
    def __init__(self) -> None: ...

    def price(
        self,
        S: ArrayLike,
        K: ArrayLike,
        t: ArrayLike,
        r: ArrayLike,
        sigma: ArrayLike,
        q: ArrayLike = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> Array:
        S, K, t, r, sigma, q = _vectorize_inputs(S, K, t, r, sigma, q)

        return _price(S, K, t, r, sigma, q, option_type, check_inputs=True)

    def delta(
        self,
        S: ArrayLike,
        K: ArrayLike,
        t: ArrayLike,
        r: ArrayLike,
        sigma: ArrayLike,
        q: ArrayLike = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> Array:
        S, K, t, r, sigma, q = _vectorize_inputs(S, K, t, r, sigma, q)

        _check_market_inputs(S, K, t, r, sigma, q)

        d1 = _compute_d1(S, K, t, r, sigma, q)

        match option_type:
            case "call":
                return jnp.exp(-q * t) * jnorm.cdf(d1)
            case "put":
                return jnp.exp(-q * t) * (jnorm.cdf(d1) - 1)

    def theta(
        self,
        S: ArrayLike,
        K: ArrayLike,
        t: ArrayLike,
        r: ArrayLike,
        sigma: ArrayLike,
        q: ArrayLike = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> Array:
        S, K, t, r, sigma, q = _vectorize_inputs(S, K, t, r, sigma, q)

        _check_market_inputs(S, K, t, r, sigma, q)

        d1 = _compute_d1(S, K, t, r, sigma, q)
        d2 = d1 - sigma * jnp.sqrt(t)

        match option_type:
            case "call":
                return (
                    -(S * sigma * jnp.exp(-q * t) * jnorm.pdf(d1) / (2 * jnp.sqrt(t)))
                    - r * K * jnp.exp(-r * t) * jnorm.cdf(d2)
                    + q * S * jnp.exp(-q * t) * jnorm.cdf(d1)
                )
            case "put":
                return (
                    -(S * sigma * jnp.exp(-q * t) * jnorm.pdf(d1) / (2 * jnp.sqrt(t)))
                    + r * K * jnp.exp(-r * t) * jnorm.cdf(-d2)
                    - q * S * jnp.exp(-q * t) * jnorm.cdf(-d1)
                )

    def gamma(
        self,
        S: ArrayLike,
        K: ArrayLike,
        t: ArrayLike,
        r: ArrayLike,
        sigma: ArrayLike,
        q: ArrayLike = 0.0,
    ) -> Array:
        S, K, t, r, sigma, q = _vectorize_inputs(S, K, t, r, sigma, q)

        _check_market_inputs(S, K, t, r, sigma, q)

        d1 = _compute_d1(S, K, t, r, sigma, q)

        return jnp.exp(-q * t) * jnorm.pdf(d1) / (S * sigma * jnp.sqrt(t))

    def vega(
        self,
        S: ArrayLike,
        K: ArrayLike,
        t: ArrayLike,
        r: ArrayLike,
        sigma: ArrayLike,
        q: ArrayLike = 0.0,
    ) -> Array:
        S, K, t, r, sigma, q = _vectorize_inputs(S, K, t, r, sigma, q)

        _check_market_inputs(S, K, t, r, sigma, q)

        d1 = _compute_d1(S, K, t, r, sigma, q)

        return S * jnp.exp(-q * t) * jnp.sqrt(t) * jnorm.pdf(d1)

    def rho(
        self,
        S: ArrayLike,
        K: ArrayLike,
        t: ArrayLike,
        r: ArrayLike,
        sigma: ArrayLike,
        q: ArrayLike = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> Array:
        S, K, t, r, sigma, q = _vectorize_inputs(S, K, t, r, sigma, q)

        _check_market_inputs(S, K, t, r, sigma, q)

        d1 = _compute_d1(S, K, t, r, sigma, q)
        d2 = d1 - sigma * jnp.sqrt(t)

        match option_type:
            case "call":
                return K * t * jnp.exp(-r * t) * jnorm.cdf(d2)
            case "put":
                return -K * t * jnp.exp(-r * t) * jnorm.cdf(-d2)

    @jax.jit
    def iv_solver(
        self,
        market_price: ArrayLike,
        S: ArrayLike,
        K: ArrayLike,
        t: ArrayLike,
        r: ArrayLike,
        q: ArrayLike = 0.0,
        option_type: Literal["call", "put"] = "call",
        num_iter: int = 200,
        tol: float = 1e-7,
    ) -> Array:
        S, K, t, r, q, market_price = _vectorize_inputs(S, K, t, r, q, market_price)

        _check_market_inputs(
            S=S, K=K, t=t, r=r, sigma=None, q=q, market_price=market_price
        )

        sigma = _manaster_koehler_init(S, K, t, r, q)

        for _ in range(num_iter):
            price = _price(S, K, t, r, sigma, q, option_type)

            f = price - market_price

            d1 = _compute_d1(S, K, t, r, sigma, q)

            vega = S * jnp.exp(-q * t) * jnp.sqrt(t) * jnorm.pdf(d1)
            vega = jnp.where(jnp.abs(vega) < _MIN_VEGA, _MIN_VEGA, vega)

            sigma_next = sigma - (f / vega)
            sigma_next = jnp.clip(sigma_next, _MIN_VOL, _MAX_VOL)

            if jnp.all(jnp.abs(sigma_next - sigma) < tol):
                return sigma_next

            sigma = sigma_next

        return sigma
