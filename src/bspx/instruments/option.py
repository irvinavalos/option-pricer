from dataclasses import dataclass

from bspx.types import ArrayLike


@dataclass(slots=True, frozen=True)
class OptionPrice:
    call: ArrayLike
    put: ArrayLike

    def __repr__(self) -> str:
        return f"OptionPrice(call={self.call}, put={self.put})"


@dataclass(slots=True, frozen=True)
class Greeks:
    delta: ArrayLike
    theta: ArrayLike
    gamma: ArrayLike
    vega: ArrayLike
    rho: ArrayLike

    def __repr__(self) -> str:
        return f"Greeks(delta={self.delta},theta={self.theta}, gamma={self.gamma}, vega={self.vega}, rho={self.rho})"
