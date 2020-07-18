import numpy as np
from numba import njit

from .common import find_first_zero, find_last_zero, similarity


@njit(inline="always")
def find_component(data, components, label, threshold):
    check = {0}
    components[0] = label
    while check:
        x = data[check.pop()]
        for j in range(1, len(components)):
            if components[j] > 0:
                continue
            elif similarity(x, data[j]) >= threshold:
                components[j] = label
                check.add(j)


@njit
def _find_components(data, components, threshold):
    label = 0
    while (start := find_first_zero(components)) is not None:
        end = find_last_zero(components) + 1
        label += 1
        find_component(data[start:end], components[start:end], label, threshold)
    return label


def find_components(data, threshold):
    n = len(data)
    components = np.zeros(n, np.min_scalar_type(n))
    max_label = _find_components(data, components, threshold)
    return components.astype(np.min_scalar_type(max_label))
