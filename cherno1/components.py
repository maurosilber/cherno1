import numpy as np
from numba import njit
from numba.typed import List

from .common import find_first_equal, find_last_equal, similarity


@njit
def find_component(repeats, components, label, threshold):
    check = List([0])
    components[0] = label
    while check:
        x = repeats[check.pop()]
        for j in range(1, len(components)):
            if components[j] > 0:
                continue
            elif similarity(x, repeats[j]) >= threshold:
                components[j] = label
                check.append(j)


@njit
def _find_components(data, components, threshold):
    label = 0
    while (start := find_first_equal(components)) is not None:
        end = find_last_equal(components) + 1
        label += 1
        find_component(data[start:end], components[start:end], label, threshold)
    return label


def find_components(data, threshold):
    n = len(data)
    components = np.zeros(n, np.min_scalar_type(n))
    max_label = _find_components(data, components, threshold)
    return components.astype(np.min_scalar_type(max_label))


def find_components_by_array(repeats, array_map):
    n = array_map.max() + 1
    k = repeats.shape[1] + 1
    min_similarity = -np.ones((n, k), dtype=np.int8)

    order = np.argsort(array_map)
    x = array_map[order]

    import tqdm

    end = 0
    for i in tqdm.trange(n):
        x, order = x[end:], order[end:]
        end = find_first_equal(x, i + 1)
        array_repeats = repeats[order[:end]]
        array_n = len(array_repeats)

        if array_n == 1:
            continue
        elif array_n == 2:
            s = similarity(*array_repeats)
            min_similarity[i, :s] = 1
        else:
            for s in reversed(range(k)):
                components = np.zeros(array_n, int)
                max_label = _find_components(array_repeats, components, s)
                min_similarity[i, s] = max_label

    return min_similarity
