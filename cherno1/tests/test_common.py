import numpy as np
from cherno1.common import find_first_zero, find_last_zero, similarity
from ward import test


def similarity_numpy(x, y):
    return ((x == y) & (x != 1) & (y != 1)).sum()


def find_first_zero_numpy(x):
    x = x == 0
    if not x.any():
        return None
    else:
        return x.argmax()


@test("Similarity")
def _():
    x = np.array([3, 4, 5, 5, 4])
    assert similarity(x, x) == x.size

    for _ in range(100):
        x, y = np.random.randint(0, 5, size=(2, 33))
        assert similarity(x, y) == similarity_numpy(x, y)


@test("Similarity excludes 1")
def _():
    x = np.array([3, 4, 5, 5, 4])
    y = np.array([1, 2, 7, 6, 4])
    assert similarity(x, y) == 1

    x = np.array([1, 4, 5, 5, 4])
    y = np.array([1, 2, 7, 6, 4])
    assert similarity(x, y) == 1

    x = np.array([1, 4, 5, 5, 4])
    y = np.array([3, 2, 7, 6, 4])
    assert similarity(x, y) == 1


@test("Find first zero")
def _():
    x = np.ones(10, dtype=int)
    assert find_first_zero(x) is None

    for _ in range(1000):
        size = np.random.randint(100)
        x = np.random.randint(0, 10, size=size)
        assert find_first_zero(x) == find_first_zero_numpy(x)


@test("Find last zero")
def _():
    x = np.ones(10, dtype=int)
    assert find_last_zero(x) is None

    for i in range(10):
        x[i] = 0
        assert find_last_zero(x) == i
