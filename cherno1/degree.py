import numpy as np
from numba import njit, prange

from .common import similarity


@njit(inline="always")
def _degree_for_x(data, x, degree_x):
    for y in data:
        d = similarity(x, y)
        degree_x[d] += 1

    # Remove link with itself
    d = similarity(x, x)
    degree_x[d] -= 1


@njit(parallel=True)
def _batch_update(data, data_batch, degree_batch):
    for i in prange(len(data_batch)):
        _degree_for_x(data, data_batch[i], degree_batch[i])


def degree_by_similarity(data, progress=False, batch_size=16):
    n, k = data.shape
    degree = np.zeros((n, k + 1), dtype=np.min_scalar_type(n))

    if progress:
        import tqdm

        for i in tqdm.trange(n):
            if i % batch_size == 0:
                _batch_update(data, data[i:][:batch_size], degree[i:][:batch_size])
    else:
        _batch_update(data, data, degree)

    return degree.astype(np.min_scalar_type(degree.max()))
