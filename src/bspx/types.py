from enum import IntEnum
from typing import Callable, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

_F64 = NDArray[np.float64]

type OptionType = Literal["call", "put"]


class DayCount(IntEnum):
    TRADING = 252
    CALENDAR = 365


type PricingFunction = Callable[
    [
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        OptionType,
    ],
    _F64,
]


@runtime_checkable
class PricingModel(Protocol):
    def call_price(self) -> _F64: ...

    def put_price(self) -> _F64: ...
