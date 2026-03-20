import jax.numpy as jnp


class BinomialTree:
    def __init__(self) -> None: ...

    def price(self, r, q, sigma, t):
        u = jnp.exp(sigma * jnp.sqrt(t))
        d = jnp.exp(-sigma * jnp.sqrt(t))
        p = (jnp.exp((r - q) * t) - d) / (u - d)
