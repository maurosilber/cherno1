import numpy as np
from cherno1.common import find_first_zero, similarity
from cherno1.components import find_component, find_components
from ward import each, fixture, test


def find_component_slow(data, threshold):
    component, check = set(), {0}
    while check:
        i = check.pop()
        component.add(i)
        x = data[i]
        for j in range(1, len(data)):
            if similarity(x, data[j]) >= threshold:
                if j not in component:
                    check.add(j)
    return component


def find_components_slow(data, threshold):
    components = np.zeros(len(data), np.int32)
    label = 0
    while (i := find_first_zero(components)) is not None:
        component = find_component_slow(data[i:], threshold)
        label += 1
        for c in component:
            components[i:][c] = label
    return components


@fixture
def repeats():
    return np.load("cherno1/tests/repeats.npy")


@test("Find component at threshold={threshold}")
def _(data=repeats, threshold=each(15, 20, 25)):
    component = find_component_slow(data, threshold)

    components = np.zeros(len(data), np.int16)
    find_component(data, components, label=1, threshold=threshold)

    assert set(np.nonzero(components)[0]) == component


@test("Find components at threshold={threshold}")
def _(data=repeats, threshold=each(15, 20, 25)):
    data = data[:10]

    components_slow = find_components_slow(data, threshold)
    components = find_components(data, threshold)
    assert (components == components_slow).all()
