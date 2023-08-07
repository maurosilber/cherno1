import networkx as nx
import numpy as np
from ward import fixture, test

from cherno1.common import similarity
from cherno1.graph import adjacency_matrix


def build_graph(data, threshold):
    g = nx.Graph()
    for i, x in enumerate(data):
        g.add_node(i)
        for j, y in enumerate(data[:i]):
            if similarity(x, y) >= threshold:
                g.add_edge(i, j)
    return g


@fixture
def repeats():
    return np.load("cherno1/tests/repeats.npy")[:3]


@test("Adjacency matrix")
def _(data=repeats):
    for s in range(1, 34):
        g = build_graph(data, s)
        adj = nx.to_numpy_array(g)
        assert np.all(adjacency_matrix(data, s) == adj)
