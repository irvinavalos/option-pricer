from enum import IntEnum
from typing import Callable, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    NDArray[np.float64],
]


class OptionTestCase(NamedTuple):
    name: str
    S: ArrayLike
    K: ArrayLike
    T: ArrayLike
    r: ArrayLike
    vol: ArrayLike
    expected_call: ArrayLike
    expected_put: ArrayLike
    source: str
    notes: str | None = None


class GreekTestCase(NamedTuple):
    name: str
    S: ArrayLike
    K: ArrayLike
    T: ArrayLike
    r: ArrayLike
    vol: ArrayLike
    expected_delta_call: ArrayLike
    expected_delta_put: ArrayLike | None
    expected_theta_call_calendar: ArrayLike
    expected_theta_call_trading: ArrayLike
    expected_theta_put_calendar: ArrayLike | None
    expected_theta_put_trading: ArrayLike | None
    expected_gamma: ArrayLike
    expected_vega: ArrayLike
    expected_rho_call: ArrayLike
    expected_rho_put: ArrayLike | None
    source: str
    notes: str | None = None
