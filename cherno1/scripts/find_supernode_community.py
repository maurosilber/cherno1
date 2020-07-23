from functools import partial

import click
import numpy as np
import tqdm
from cherno1.common import castdown
from cherno1.graph import build_igraph
from p_tqdm import p_umap


def func_joined(sn_repeats, sn, min_similarity, progress=False):
    degree = np.zeros((len(sn_repeats), 34), dtype=int)
    partitions = np.zeros((len(sn_repeats), 34), dtype=int)
    for s in tqdm.trange(min_similarity, 33, disable=not progress):
        graph = build_igraph(sn_repeats, s)
        degree[:, s] = graph.degree()
        partitions[:, s] = graph.community_infomap().membership

    np.save(f"degree_{sn}.npy", castdown(degree))
    np.save(f"partitions_{sn}.npy", castdown(partitions))


def func_separated(sn_repeats, sn, min_similarity, progress=False):
    for s in tqdm.trange(33, min_similarity, -1, disable=not progress):
        graph = build_igraph(sn_repeats, s)
        degree = graph.degree()
        partitions = graph.community_infomap().membership
        np.save(f"degree_{sn}_{s}.npy", castdown(degree))
        np.save(f"partitions_{sn}_{s}.npy", castdown(partitions))


@click.command()
@click.argument("repeat_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("supernode_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("supernodes", nargs=-1, type=click.INT)
@click.option(
    "--min-similarity", type=click.INT, default=10, help="Upto this similarity.",
)
@click.option("--separated", is_flag=True)
def cli(repeat_file, supernode_file, supernodes, min_similarity, separated):
    repeats = np.load(repeat_file)["repeats"]
    supernode_array = np.load(supernode_file)["25"]
    if not supernodes:
        supernodes = np.unique(supernode_array)
    func = func_separated if separated else func_joined
    supernode_repeats = (repeats[supernode_array == sn] for sn in supernodes)
    p_umap(partial(func, min_similarity=min_similarity), supernode_repeats, supernodes)


if __name__ == "__main__":
    cli()
