import numpy as np
from numba import njit, prange

from .common import find_first_equal, similarity


@njit(inline="always")
def _degree_by_similarity_for_repeat(repeats, repeat, degree_repeat):
    for y in repeats:
        d = similarity(repeat, y)
        degree_repeat[d] += 1

    # Remove link with itself
    d = similarity(repeat, repeat)
    degree_repeat[d] -= 1


@njit(parallel=True)
def _batch_update(repeats, repeats_batch, degrees_batch):
    for i in prange(len(repeats_batch)):
        _degree_by_similarity_for_repeat(repeats, repeats_batch[i], degrees_batch[i])


def degree_by_similarity(repeats, progress=False, batch_size=16):
    n, k = repeats.shape
    degree = np.zeros((n, k + 1), dtype=np.min_scalar_type(n))

    if progress:
        import tqdm

        for i in tqdm.trange(n):
            if i % batch_size == 0:
                _batch_update(
                    repeats, repeats[i:][:batch_size], degree[i:][:batch_size]
                )
    else:
        _batch_update(repeats, repeats, degree)

    return degree.astype(np.min_scalar_type(degree.max()))


def degree_upto_similarity(repeats, *args, **kwargs):
    degree = degree_by_similarity(repeats, *args, **kwargs)
    return np.cumsum(degree[:, ::-1], axis=1)[:, ::-1]


@njit
def _degree_by_similarity_for_array(array, degree):
    for repeat in array:
        _degree_by_similarity_for_repeat(array, repeat, degree)
    degree //= 2


def degree_by_similarity_by_array(repeats, array_map):
    n = array_map.max() + 1
    _, k = repeats.shape
    degree = np.zeros((n, k + 1), dtype=int)

    order = np.argsort(array_map)
    x = array_map[order]

    for i in range(n):
        end = find_first_equal(x, i + 1)
        if end > 1:
            array = repeats[order[:end]]
            _degree_by_similarity_for_array(array, degree[i])
        x, order = x[end:], order[end:]

    return degree.astype(np.min_scalar_type(degree.max()))


@njit(inline="always")
def degree_by_similarity_between_supernodes(sn1, sn2):
    degree = np.zeros(sn1.shape[1] + 1, dtype=np.int32)
    for x in sn1:
        for y in sn2:
            d = similarity(x, y)
            degree[d] += 1
    return degree
