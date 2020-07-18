from typing import Optional

import numpy as np
from numba import njit


@njit(inline="always")
def similarity(x: np.array, y: np.array) -> int:
    """Cantidad de valores en común entre x e y.

    x[i] != y[i] si ambos son 1.
    """
    out = 0
    for xi, yi in zip(x, y):
        if 1 != xi == yi:
            out += 1
    return out


@njit(inline="always")
def find_first_zero(x: np.array) -> Optional[int]:
    for i in range(x.size):
        if x[i] == 0:
            return i


@njit(inline="always")
def find_last_zero(x: np.array) -> Optional[int]:
    i = find_first_zero(x[::-1])
    if i is not None:
        return x.size - 1 - i
