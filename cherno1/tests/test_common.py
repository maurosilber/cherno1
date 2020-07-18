import timeit

import numpy as np
from cherno1.common import find_first_zero, similarity
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


@test("Benchmark: Similarity")
def _():
    x, y = np.random.randint(0, 22, size=(2, 33)).astype(np.uint8)
    globals = {"similarity": similarity, "x": x, "y": y}
    t_fast = timeit.timeit(
        "similarity(x, y)", setup="similarity(x, y)", globals=globals, number=1000
    )
    globals = {"similarity_numpy": similarity_numpy, "x": x, "y": y}
    t_slow = timeit.timeit("similarity_numpy(x, y)", globals=globals, number=1000)
    assert t_fast < t_slow
    print(f"x{t_slow/t_fast:.1f}")


@test("Find first zero")
def _():
    x = np.ones(10, dtype=int)
    assert find_first_zero(x) is None

    for _ in range(1000):
        size = np.random.randint(100)
        x = np.random.randint(0, 10, size=size)
        assert find_first_zero(x) == find_first_zero_numpy(x)


@test("Benchmark: Find first zero")
def _():
    x = np.ones(10_000, dtype=int)
    for i in np.linspace(0, x.size - 1, 5, dtype=int)[::-1]:
        x[i] = 0
        globals = {"func": find_first_zero, "x": x}
        t_fast = timeit.timeit("func(x)", setup="func(x)", globals=globals, number=1000)
        globals["func"] = find_first_zero_numpy
        t_slow = timeit.timeit("func(x)", setup="func(x)", globals=globals, number=1000)
        assert t_fast < t_slow
        print(f"x{t_slow/t_fast:.1f}")
