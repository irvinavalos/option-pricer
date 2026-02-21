from typing import Callable, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

type OptionType = Literal["call", "put"]

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
