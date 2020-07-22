import networkx as nx
import numpy as np
import tqdm
from numba import njit

from .common import similarity


@njit
def similarity_matrix(repeats):
    n = len(repeats)
    M = np.empty((n, n), dtype=np.uint8)
    for i, x in enumerate(repeats):
        for j, y in enumerate(repeats[: i + 1]):
            M[i, j] = M[j, i] = similarity(x, y)
    return M


def adjacency_matrix(repeats, threshold):
    sm = similarity_matrix(repeats)
    sm = sm >= threshold
    sm[np.diag_indices_from(sm)] = False
    return sm


@njit
def _edges_for_repeat(repeats, repeat, threshold):
    out = np.zeros(len(repeats), dtype=np.bool_)
    for i, r in enumerate(repeats):
        out[i] = similarity(repeat, r) >= threshold
    return out


def build_graph(repeats, threshold, progress=False):
    graph = nx.Graph()
    for i, repeat in enumerate(tqdm.tqdm(repeats, disable=not progress)):
        edge = _edges_for_repeat(repeats[i:], repeat, threshold)
        graph.add_edges_from([(i, j) for j in np.nonzero(edge)[0]])
    return graph
