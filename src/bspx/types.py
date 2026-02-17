from typing import Any, Literal, TypeGuard

import numpy as np

type OptionType = Literal["call", "put"]

type Scalar = int | float | np.number
type Array = np.ndarray
type ArrayLike = Scalar | Array


def is_array(val: Any) -> TypeGuard[Array]:
    return isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number)
