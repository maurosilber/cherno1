import numpy as np
from numba import njit
from numba.typed import List

from .common import find_first_equal


@njit
def _project(repeats, arrays, projection, label):
    checked_r, checked_a = {0}, {0}
    check_r, check_a, projection[0] = List(repeats[:1]), List(arrays[:1]), label
    while check_r or check_a:
        if len(check_r) > 0:
            ri = check_r.pop()
            checked_r.add(ri)
        else:
            ri = -1
        if len(check_a) > 0:
            ai = check_a.pop()
            checked_a.add(ai)
        else:
            ai = -1

        for i in range(projection.size):
            if projection[i] > 0:
                continue
            else:
                if repeats[i] == ri:
                    projection[i] = label
                    if arrays[i] not in checked_a:
                        check_a.append(arrays[i])
                if arrays[i] == ai:
                    projection[i] = label
                    if repeats[i] not in checked_r:
                        check_r.append(repeats[i])


def project(repeats, arrays):
    projection = np.zeros_like(repeats)
    label = 0
    while (i := find_first_equal(projection, 0)) is not None:
        label += 1
        _project(repeats[i:], arrays[i:], projection[i:], label)
    return projection
