from dataclasses import dataclass

from bspx.types import ArrayLike


@dataclass(slots=True)
class OptionPrice:
    call: ArrayLike
    put: ArrayLike

    def __repr__(self) -> str:
        return f"OptionPrice(call={self.call}, put={self.put})"


@dataclass(slots=True)
class Greeks:
    delta: ArrayLike
    theta: ArrayLike
    gamma: ArrayLike
    vega: ArrayLike
    rho: ArrayLike
