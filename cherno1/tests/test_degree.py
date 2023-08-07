import networkx as nx
import numpy as np
import pandas as pd
from ward import each, fixture, test

from cherno1.common import similarity
from cherno1.degree import degree_by_similarity


def build_graph(data, threshold):
    g = nx.Graph()
    for i, x in enumerate(data):
        g.add_node(i)
        for j, y in enumerate(data[:i]):
            if similarity(x, y) == threshold:
                g.add_edge(i, j)
    return g


@fixture
def repeats():
    return np.load("cherno1/tests/repeats.npy")[:100]


@test("Degree distribution")
def _(data=repeats):
    degrees = degree_by_similarity(data)

    for s, d in enumerate(degrees.T):
        g = build_graph(data, s)
        degree = pd.DataFrame(g.degree).sort_values(0)[1].values
        assert np.all(d == degree)


@test("Degree distribution (batch size: {batch_size})")
def _(data=repeats, batch_size=each(1, 2, 3)):
    degrees = degree_by_similarity(data)
    batched_degrees = degree_by_similarity(data, progress=True, batch_size=batch_size)
    assert np.all(batched_degrees == degrees)
