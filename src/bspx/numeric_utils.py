from collections.abc import Callable
from functools import partial

import numpy as np
from numpy.typing import ArrayLike

from bspx.types import _F64

# Ensures that input of type ArrayLike (float, list, ndarray, etc.) is converted to
# np.float64 array
to_f64: Callable[[ArrayLike], _F64] = partial(np.asarray, dtype=np.float64)
