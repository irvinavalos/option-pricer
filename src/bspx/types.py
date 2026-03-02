from collections.abc import Callable
from enum import IntEnum, StrEnum
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

type _F64 = NDArray[np.float64]
type OptionType = Literal["call", "put"]


class DayCount(IntEnum):
    TRADING = 252
    CALENDAR = 365

    @property
    def min_time_to_expiry(self) -> float:
        return 1 / self.value


class DiffMethod(StrEnum):
    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"
    AUTOMATIC = "automatic"


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


@runtime_checkable
class GreeksBackend(Protocol):
    method: DiffMethod

    def delta(self, option_type: OptionType = "call") -> _F64: ...
    def theta(
        self, option_type: OptionType = "call", day_count: DayCount = DayCount.TRADING
    ) -> _F64: ...
    def gamma(self) -> _F64: ...
    def vega(self) -> _F64: ...
    def rho(self, option_type: OptionType = "call") -> _F64: ...
