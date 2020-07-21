"""
Calculate in-array degree distribution for a given similarity threshold.

OUTPUT: degree array of (N array, k similarity level)

So, array[i,j] is the number of internal edges for array i at similarity j.
"""

import numpy as np
from cherno1.degree import degree_by_similarity_by_array


def calc_degree_by_array():
    array_map = np.load("../data/repeats_map.npy")
    repeats = np.load("../data/internal_repeats_num_ali.npy")
    return degree_by_similarity_by_array(repeats, array_map)
