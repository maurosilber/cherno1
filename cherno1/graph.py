import numpy as np
from numba import njit

from .common import similarity


@njit
def similarity_matrix(repeats):
    n = len(repeats)
    M = np.empty((n, n), dtype=np.uint8)
    for i, x in enumerate(repeats):
        for j, y in enumerate(repeats[: i + 1]):
            M[i, j] = M[j, i] = similarity(x, y)
    return M


def adjacency_matrix(repeats, threshold):
    sm = similarity_matrix(repeats)
    sm = sm >= threshold
    sm[np.diag_indices_from(sm)] = False
    return sm
