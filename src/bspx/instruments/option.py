from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class OptionPrice:
    call: NDArray[np.float64]
    put: NDArray[np.float64]

    def __repr__(self) -> str:
        return f"OptionPrice(call={self.call}, put={self.put})"


@dataclass(slots=True, frozen=True)
class Greeks:
    delta: NDArray[np.float64]
    theta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    vega: NDArray[np.float64]
    rho: NDArray[np.float64]

    def __repr__(self) -> str:
        return f"Greeks(delta={self.delta},theta={self.theta}, gamma={self.gamma}, vega={self.vega}, rho={self.rho})"
