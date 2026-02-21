from typing import Any, Callable, Literal, NamedTuple, TypeGuard

import numpy as np

type OptionType = Literal["call", "put"]

type Scalar = int | float | np.number
type Array = np.ndarray
type ArrayLike = Scalar | Array


# TODO: add error checking if type None is passed in
def is_array(val: Any) -> TypeGuard[Array]:
    return isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number)


type PricingFunction = Callable[
    [
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        OptionType,
    ],
    ArrayLike,
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
